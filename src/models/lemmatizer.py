"""Lemmatizer model: edit-script classification."""
from __future__ import annotations
from dataclasses import replace

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import (
    LemmatizerDataset,
    _build_vocab,
    _compute_edit_script,
)
from src.models.base import BorgBaseModel
from src.models.inference import sentence_batches
from src.models.evaluation import (
    average_f1,
    evaluate_predictions,
    print_validation_metrics,
)
from src.models.checkpoints import save_training_models


def _apply_edit_script(form: str, script: str) -> str:
    """Reconstruct a lemma from a form and an edit script."""
    try:
        parts = script.split(":")
        prefix_keep = int(parts[0][1:])  # k<n>
        suffix_strip = int(parts[1][1:])  # s<n>
        suffix_add = parts[2][1:] if len(parts) > 2 else ""  # a<str>
    except (IndexError, ValueError):
        return form.lower()

    stem = form.lower()[:prefix_keep]

    if suffix_strip > 0 and len(form) - suffix_strip > prefix_keep:
        pass  # strip is relative to the original tail already

    lemma = stem + suffix_add
    return lemma if lemma else form.lower()


class LemmatizerModel(BorgBaseModel):
    """Classifies each token into a lemma edit-script."""

    def __init__(
        self,
        config: BorgConfig,
        upos_vocab: Optional[Dict[str, int]] = None,
        script_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(config, "lemmatizer")

        self.upos_vocab = upos_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        self.script_vocab = script_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        self._build_heads()

    def _build_heads(self) -> None:
        upos_emb_dim = 32

        self.upos_embedding = nn.Embedding(
            len(self.upos_vocab),
            upos_emb_dim,
            padding_idx=0,
        )

        self.classifier = nn.Linear(
            self.hidden_size + upos_emb_dim,
            len(self.script_vocab),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        upos_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(
            input_ids,
            attention_mask,
        )  # (B, L, H)

        upos_emb = self.upos_embedding(
            upos_ids
        )  # (B, L, E)

        concat = torch.cat(
            [hidden, upos_emb],
            dim=-1,
        )  # (B, L, H+E)

        return self.classifier(
            concat
        )  # (B, L, num_scripts)

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "upos_vocab": self.upos_vocab,
            "script_vocab": self.script_vocab,
            "upos_embedding": self.upos_embedding.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def _set_extras(
        self,
        extras: Dict[str, Any],
    ) -> None:
        self.upos_vocab = extras["upos_vocab"]
        self.script_vocab = extras["script_vocab"]

        self._build_heads()

        self.upos_embedding.load_state_dict(
            extras["upos_embedding"]
        )
        self.classifier.load_state_dict(
            extras["classifier"]
        )

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
    ) -> "LemmatizerModel":
        device = config.resolve_device()
        device_type = torch.device(device).type

        torch.manual_seed(config.seed)

        all_upos = [
            t.upos
            for s in train_sentences
            for t in s.regular_tokens()
        ]
        upos_vocab = _build_vocab(all_upos)

        all_scripts = [
            _compute_edit_script(
                t.form,
                t.lemma,
            )
            for s in train_sentences
            for t in s.regular_tokens()
        ]
        script_vocab = _build_vocab(all_scripts)

        model = LemmatizerModel(
            config,
            upos_vocab,
            script_vocab,
        ).to(device)

        train_ds = LemmatizerDataset(
            train_sentences,
            config.model_name,
            config.max_seq_length,
            upos_vocab,
            script_vocab,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        total_steps = (
            len(train_loader)
            * config.num_epochs
        )

        warmup_steps = int(
            total_steps
            * config.warmup_ratio
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        best_score = -1.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"[Lemmatizer] Epoch {epoch + 1}",
            )

            for batch in progress:
                input_ids = batch[
                    "input_ids"
                ].to(device)

                attention_mask = batch[
                    "attention_mask"
                ].to(device)

                upos_ids = batch[
                    "upos_ids"
                ].to(device)

                script_labels = batch[
                    "script_labels"
                ].to(device)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device_type,
                    dtype=config.dtype,
                    enabled=device_type == "cuda",
                ):
                    logits = model(
                        input_ids,
                        attention_mask,
                        upos_ids,
                    )

                    loss = loss_fn(
                        logits.view(
                            -1,
                            len(script_vocab),
                        ),
                        script_labels.view(-1),
                    )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=(
                        f"{total_loss / max(progress.n, 1):.4f}"
                    ),
                    lr=(
                        f"{scheduler.get_last_lr()[0]:.2e}"
                    ),
                )

            avg_loss = (
                total_loss
                / len(train_loader)
            )

            metrics = evaluate_predictions(
                dev_sentences,
                model.predict(dev_sentences),
            )

            print(f"  loss={avg_loss:.4f}")

            print_validation_metrics(
                metrics,
                ["LEMMA"],
            )

            score = average_f1(
                metrics,
                ["LEMMA"],
            )

            best_score = save_training_models(
                model,
                model_path,
                score,
                best_score,
            )

        return model

    # ------------------------------------------------------------------
    def predict(
        self,
        sentences: List[Sentence],
        *,
        show_progress: bool = True,
    ) -> List[Sentence]:
        device = self.config.resolve_device()
        device_type = torch.device(device).type

        self.eval()
        self.to(device)

        inv_script = {
            v: k
            for k, v in self.script_vocab.items()
        }

        results = list(sentences)
        for batch in sentence_batches(self, sentences, show_progress=show_progress):
            upos_ids = torch.zeros_like(batch.input_ids)
            for row, word_ids in enumerate(batch.word_ids):
                token_upos = [
                    self.upos_vocab.get(token.upos, self.upos_vocab["<UNK>"])
                    for token in batch.tokens[row]
                ]
                upos_ids[row, :len(word_ids)] = torch.tensor(
                    [token_upos[wid] if wid is not None else 0 for wid in word_ids],
                    dtype=torch.long,
                )

            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=self.config.dtype,
                enabled=device_type == "cuda",
            ):
                logits = self(
                    batch.input_ids.to(device),
                    batch.attention_mask.to(device),
                    upos_ids.to(device),
                )
                script_preds = logits.argmax(-1).cpu().tolist()

            for row, index in enumerate(batch.indices):
                sent = sentences[index]
                word_scripts = {
                    wid: inv_script.get(script_preds[row][pos], "k0:s0:a")
                    for wid, pos in batch.word_positions[row].items()
                }
                new_sent = Sentence(comments=sent.comments)
                for tok in sent.tokens:
                    if tok.is_multiword() or tok.is_empty():
                        new_sent.tokens.append(tok)
                        continue
                    script = word_scripts.get(tok.id - 1, "k0:s0:a")
                    new_sent.tokens.append(
                        replace(tok, lemma=_apply_edit_script(tok.form, script))
                    )
                results[index] = new_sent

        return results