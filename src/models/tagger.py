"""Tagger model: UPOS / XPOS / FEATS classification."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import TaggerDataset, _build_vocab, _feats_to_str
from src.models.base import BorgBaseModel
from src.models.evaluation import average_f1, evaluate_predictions, print_validation_metrics
from src.models.checkpoints import save_training_models


class TaggerModel(BorgBaseModel):
    """Multi-head sequence labeller for UPOS, XPOS, and FEATS."""

    def __init__(
        self,
        config: BorgConfig,
        upos_vocab: Optional[Dict[str, int]] = None,
        xpos_vocab: Optional[Dict[str, int]] = None,
        feats_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(config, "tagger")
        self.upos_vocab = upos_vocab or {"<PAD>": 0, "<UNK>": 1}
        self.xpos_vocab = xpos_vocab or {"<PAD>": 0, "<UNK>": 1}
        self.feats_vocab = feats_vocab or {"<PAD>": 0, "<UNK>": 1}
        self._build_heads()

    def _build_heads(self) -> None:
        self.upos_head = nn.Linear(self.hidden_size, len(self.upos_vocab))
        self.xpos_head = nn.Linear(self.hidden_size, len(self.xpos_vocab))
        self.feats_head = nn.Linear(self.hidden_size, len(self.feats_vocab))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple:
        hidden = self.encode(input_ids, attention_mask)  # (B, L, H)
        return self.upos_head(hidden), self.xpos_head(hidden), self.feats_head(hidden)

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "upos_vocab": self.upos_vocab,
            "xpos_vocab": self.xpos_vocab,
            "feats_vocab": self.feats_vocab,
            "upos_head": self.upos_head.state_dict(),
            "xpos_head": self.xpos_head.state_dict(),
            "feats_head": self.feats_head.state_dict(),
        }

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        self.upos_vocab = extras["upos_vocab"]
        self.xpos_vocab = extras["xpos_vocab"]
        self.feats_vocab = extras["feats_vocab"]
        self._build_heads()
        self.upos_head.load_state_dict(extras["upos_head"])
        self.xpos_head.load_state_dict(extras["xpos_head"])
        self.feats_head.load_state_dict(extras["feats_head"])

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
    ) -> "TaggerModel":
        device = config.resolve_device()
        torch.manual_seed(config.seed)

        # Build vocabularies from training data
        all_upos = [t.upos for s in train_sentences for t in s.regular_tokens()]
        all_xpos = [t.xpos for s in train_sentences for t in s.regular_tokens()]
        all_feats = [_feats_to_str(t) for s in train_sentences for t in s.regular_tokens()]
        upos_vocab = _build_vocab(all_upos)
        xpos_vocab = _build_vocab(all_xpos)
        feats_vocab = _build_vocab(all_feats)

        model = TaggerModel(config, upos_vocab, xpos_vocab, feats_vocab).to(device)

        train_ds = TaggerDataset(
            train_sentences, config.model_name, config.max_seq_length,
            upos_vocab, xpos_vocab, feats_vocab,
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        inv_upos = {v: k for k, v in upos_vocab.items()}
        best_score = -1.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0
            progress = tqdm(train_loader, desc=f"[Tagger] Epoch {epoch + 1}")
            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                upos_lbl = batch["upos_labels"].to(device)
                xpos_lbl = batch["xpos_labels"].to(device)
                feats_lbl = batch["feats_labels"].to(device)

                u_logits, x_logits, f_logits = model(input_ids, attention_mask)
                upos_loss = loss_fn(u_logits.view(-1, len(upos_vocab)), upos_lbl.view(-1))
                xpos_loss = loss_fn(x_logits.view(-1, len(xpos_vocab)), xpos_lbl.view(-1))
                feats_loss = loss_fn(f_logits.view(-1, len(feats_vocab)), feats_lbl.view(-1))
                loss = upos_loss + xpos_loss + feats_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    upos_loss=f"{upos_loss.item():.4f}",
                    xpos_loss=f"{xpos_loss.item():.4f}",
                    feats_loss=f"{feats_loss.item():.4f}",
                    avg_loss=f"{total_loss / max(progress.n, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            avg_loss = total_loss / len(train_loader)

            metrics = evaluate_predictions(dev_sentences, model.predict(dev_sentences))
            print(f"  loss={avg_loss:.4f}")
            print_validation_metrics(metrics, ["UPOS", "XPOS", "FEATS"])
            score = average_f1(metrics, ["UPOS", "XPOS", "FEATS"])
            best_score = save_training_models(model, model_path, score, best_score)

        return model

    # ------------------------------------------------------------------
    def predict(self, sentences: List[Sentence]) -> List[Sentence]:
        device = self.config.resolve_device()
        self.eval()
        self.to(device)

        inv_upos = {v: k for k, v in self.upos_vocab.items()}
        inv_xpos = {v: k for k, v in self.xpos_vocab.items()}
        inv_feats = {v: k for k, v in self.feats_vocab.items()}

        results: List[Sentence] = []
        hf_tok = self.hf_tokenizer

        for sent in sentences:
            tokens = sent.regular_tokens()
            if not tokens:
                results.append(sent)
                continue
            forms = [t.form for t in tokens]

            encoding = hf_tok(
                forms,
                is_split_into_words=True,
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                u_logits, x_logits, f_logits = self(input_ids, attention_mask)

            u_preds = u_logits.squeeze(0).argmax(-1).cpu().tolist()
            x_preds = x_logits.squeeze(0).argmax(-1).cpu().tolist()
            f_preds = f_logits.squeeze(0).argmax(-1).cpu().tolist()

            word_to_pos: Dict[int, tuple] = {}
            for i, wid in enumerate(word_ids):
                if wid is None or wid in word_to_pos:
                    continue
                word_to_pos[wid] = (
                    inv_upos.get(u_preds[i], "_"),
                    inv_xpos.get(x_preds[i], "_"),
                    inv_feats.get(f_preds[i], "_"),
                )

            new_sent = Sentence(comments=sent.comments)
            for tok in sent.tokens:
                if tok.is_multiword() or tok.is_empty():
                    new_sent.tokens.append(tok)
                    continue
                tid = tok.id - 1  # 0-based index
                upos, xpos, feats_str = word_to_pos.get(tid, ("_", "_", "_"))
                new_tok = Token(
                    id=tok.id, form=tok.form, lemma=tok.lemma,
                    upos=upos, xpos=xpos, feats=feats_str,
                    head=tok.head, deprel=tok.deprel, deps=tok.deps, misc=tok.misc,
                )
                new_sent.tokens.append(new_tok)
            results.append(new_sent)

        return results
