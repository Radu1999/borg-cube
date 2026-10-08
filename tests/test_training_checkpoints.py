"""Tests for retaining latest and best training models."""
import ast
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from src.models.checkpoints import TrainingState, save_training_models


class _FakeModel:
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "score"), "w", encoding="utf-8") as output:
            output.write(str(self.score))


class TestTrainingCheckpoints(unittest.TestCase):

    def test_each_trainer_uses_training_state(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for component in ("tokenizer", "tagger", "parser", "lemmatizer"):
            with self.subTest(component=component):
                path = os.path.join(root, "src", "models", f"{component}.py")
                with open(path, encoding="utf-8") as source:
                    tree = ast.parse(source.read())
                training = next(
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "train_model"
                )
                contexts = [
                    item.context_expr
                    for node in ast.walk(training) if isinstance(node, ast.With)
                    for item in node.items
                ]
                self.assertTrue(any(
                    isinstance(context, ast.Call)
                    and isinstance(context.func, ast.Name)
                    and context.func.id == "TrainingState"
                    for context in contexts
                ), f"{component} does not manage training with TrainingState")
                state_calls = {
                    node.func.attr for node in ast.walk(training)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "state"
                }
                self.assertTrue({"progress", "step", "finish_epoch"}.issubset(state_calls))

    def test_restores_optimizer_scheduler_counters_and_rng(self):
        model = _FakeModel()
        model.config = SimpleNamespace(seed=42)
        model.score = 0.7
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loader = DataLoader(torch.arange(4), batch_size=2, shuffle=True)
        with tempfile.TemporaryDirectory() as path:
            state = TrainingState(model, path, optimizer, scheduler, loader)
            parameter.sum().backward()
            optimizer.step()
            scheduler.step()
            state.step(0.5)
            state.finish_epoch(0.7)
            expected_random = torch.rand(3)
            restored_optimizer = torch.optim.AdamW([parameter], lr=9.0)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1)
            restored = TrainingState(
                model, path, restored_optimizer, restored_scheduler, loader, resume=True,
            )
            self.assertEqual(restored.epoch, 1)
            self.assertEqual(restored.batch, 0)
            self.assertEqual(restored.global_step, 1)
            self.assertEqual(restored.best_score, 0.7)
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(restored_optimizer.state[parameter]["step"].item(), 1)
            torch.testing.assert_close(torch.rand(3), expected_random)

    def test_retains_last_and_best_only(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as path:
            model.score = 0.7
            best_score = save_training_models(model, path, model.score, -1.0)
            model.score = 0.6
            best_score = save_training_models(model, path, model.score, best_score)

            self.assertEqual(set(os.listdir(path)), {"last", "best"})
            with open(os.path.join(path, "last", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.6")
            with open(os.path.join(path, "best", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.7")

    def test_updates_best_when_mean_score_improves(self):
        model = _FakeModel()
        model.score = 0.5
        with tempfile.TemporaryDirectory() as path:
            best_score = save_training_models(model, path, 0.5, 0.4)

            self.assertEqual(best_score, 0.5)
            with open(os.path.join(path, "best", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.5")


if __name__ == "__main__":
    unittest.main()
