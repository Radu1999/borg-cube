"""Utilities for retaining training model checkpoints."""
import os
import random
from typing import Any

import torch
from tqdm import tqdm


class TrainingState:
    """Persist training counters and randomness alongside a model checkpoint."""

    def __init__(self, model, path, optimizer, scheduler, loader, resume=False):
        self.model = model
        self.path = path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = loader
        self.epoch = 0
        self.batch = 0
        self.global_step = 0
        self.total_loss = 0.0
        self.best_score = -1.0
        self.generator = torch.Generator().manual_seed(model.config.seed)
        self.loader.generator = self.generator
        self.loader.sampler.generator = self.generator
        self.epoch_rng = None
        self.epoch_generator = None
        if resume:
            state = torch.load(
                os.path.join(path, "last", "training_state.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if state["batches_per_epoch"] != len(loader):
                raise ValueError("Cannot resume with a different number of training batches")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            for name in ("epoch", "batch", "global_step", "total_loss", "best_score",
                         "epoch_rng", "epoch_generator"):
                setattr(self, name, state[name])
            self.generator.set_state(state["generator"])
            self.restore_rng(state["rng"])

    @staticmethod
    def capture_rng():
        return {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def restore_rng(state):
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def progress(self, epoch, label):
        self.epoch = epoch
        if self.batch:
            current_rng = self.capture_rng()
            self.generator.set_state(self.epoch_generator)
            self.restore_rng(self.epoch_rng)
            batches = iter(self.loader)
            for _ in range(self.batch):
                next(batches)
            self.restore_rng(current_rng)
        else:
            self.epoch_rng = self.capture_rng()
            self.epoch_generator = self.generator.get_state()
            batches = iter(self.loader)
        return tqdm(
            batches, total=len(self.loader), initial=self.batch,
            desc=f"[{label}] Epoch {epoch + 1}",
        )

    def step(self, loss):
        self.batch += 1
        self.global_step += 1
        self.total_loss += loss

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "batch": self.batch,
            "global_step": self.global_step,
            "total_loss": self.total_loss,
            "best_score": self.best_score,
            "batches_per_epoch": len(self.loader),
            "rng": self.capture_rng(),
            "generator": self.generator.get_state(),
            "epoch_rng": self.epoch_rng,
            "epoch_generator": self.epoch_generator,
        }

    def finish_epoch(self, score):
        self.epoch += 1
        self.batch = 0
        self.total_loss = 0.0
        previous_best = self.best_score
        self.best_score = max(score, previous_best)
        save_training_models(self.model, self.path, score, previous_best, self.state_dict())

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is KeyboardInterrupt:
            checkpoint = os.path.join(self.path, "last")
            self.model.save(checkpoint)
            torch.save(self.state_dict(), os.path.join(checkpoint, "training_state.pt"))
        return False


def save_training_models(
    model: Any, path: str, score: float, best_score: float, training_state=None
) -> float:
    """Save the latest model and update the best model when its score improves."""
    model.save(os.path.join(path, "last"))
    if training_state is not None:
        torch.save(training_state, os.path.join(path, "last", "training_state.pt"))
    if score > best_score:
        model.save(os.path.join(path, "best"))
        if training_state is not None:
            torch.save(training_state, os.path.join(path, "best", "training_state.pt"))
        return score
    return best_score
