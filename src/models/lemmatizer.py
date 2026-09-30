"""Lemmatizer model: edit-script classification."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import LemmatizerDataset, _build_vocab, _compute_edit_script
from src.models.base import BorgBaseModel


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
        self.upos_vocab = upos_vocab or {"<PAD>": 0, "<UNK>": 1}
        self.script_vocab = script_vocab or {"<PAD>": 0, "<UNK>": 1}
        self._build_heads()

    def _build_heads(self) -> None:
        upos_emb_dim = 32
        self.upos_embedding = nn.Embedding(len(self.upos_vocab), upos_emb_dim, padding_idx=0)
        self.classifier = nn.Linear(self.hidden_size + upos_emb_dim, len(self.script_vocab))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        upos_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)  # (B, L, H)
        upos_emb = self.upos_embedding(upos_ids)         # (B, L, E)
        concat = torch.cat([hidden, upos_emb], dim=-1)   # (B, L, H+E)
        return self.classifier(concat)                   # (B, L, num_scripts)

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "upos_vocab": self.upos_vocab,
            "script_vocab": self.script_vocab,
            "upos_embedding": self.upos_embedding.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        self.upos_vocab = extras["upos_vocab"]
        self.script_vocab = extras["script_vocab"]
        self._build_heads()
        self.upos_embedding.load_state_dict(extras["upos_embedding"])
        self.classifier.load_state_dict(extras["classifier"])

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
    ) -> "LemmatizerModel":
        device = config.resolve_device()
        torch.manual_seed(config.seed)

        all_upos = [t.upos for s in train_sentences for t in s.regular_tokens()]
        upos_vocab = _build_vocab(all_upos)
        all_scripts = [
            _compute_edit_script(t.form, t.lemma)
            for s in train_sentences
            for t in s.regular_tokens()
        ]
        script_vocab = _build_vocab(all_scripts)

        model = LemmatizerModel(config, upos_vocab, script_vocab).to(device)

        train_ds = LemmatizerDataset(
            train_sentences, config.model_name, config.max_seq_length,
            upos_vocab, script_vocab,
        )
        dev_ds = LemmatizerDataset(
            dev_sentences, config.model_name, config.max_seq_length,
            upos_vocab, script_vocab,
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=config.eval_batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        best_acc = -1.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0
            progress = tqdm(train_loader, desc=f"[Lemmatizer] Epoch {epoch + 1}")
            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                upos_ids = batch["upos_ids"].to(device)
                script_labels = batch["script_labels"].to(device)

                logits = model(input_ids, attention_mask, upos_ids)
                loss = loss_fn(logits.view(-1, len(script_vocab)), script_labels.view(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{total_loss / progress.n:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            avg_loss = total_loss / len(train_loader)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    upos_ids = batch["upos_ids"].to(device)
                    script_labels = batch["script_labels"].to(device)
                    logits = model(input_ids, attention_mask, upos_ids)
                    preds = logits.argmax(-1)
                    mask = script_labels != -100
                    correct += (preds[mask] == script_labels[mask]).sum().item()
                    total += mask.sum().item()

            acc = correct / max(total, 1)
            print(f"  loss={avg_loss:.4f}  dev_script_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                model.save(model_path)

        return model

    # ------------------------------------------------------------------
    def predict(self, sentences: List[Sentence]) -> List[Sentence]:
        device = self.config.resolve_device()
        self.eval()
        self.to(device)

        inv_script = {v: k for k, v in self.script_vocab.items()}
        hf_tok = self.hf_tokenizer
        results: List[Sentence] = []

        for sent in sentences:
            tokens = sent.regular_tokens()
            if not tokens:
                results.append(sent)
                continue
            forms = [t.form for t in tokens]
            upos_list = [t.upos for t in tokens]

            encoding = hf_tok(
                forms,
                is_split_into_words=True,
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            seq_len = encoding["input_ids"].size(1)

            upos_ids_list = [0] * seq_len
            for i, wid in enumerate(word_ids):
                if wid is not None and wid < len(upos_list):
                    upos_ids_list[i] = self.upos_vocab.get(upos_list[wid], self.upos_vocab["<UNK>"])

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            upos_ids = torch.tensor([upos_ids_list], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = self(input_ids, attention_mask, upos_ids)
            script_preds = logits.squeeze(0).argmax(-1).cpu().tolist()

            word_scripts: Dict[int, str] = {}
            for i, wid in enumerate(word_ids):
                if wid is not None and wid not in word_scripts:
                    word_scripts[wid] = inv_script.get(script_preds[i], "k0:s0:a")

            new_sent = Sentence(comments=sent.comments)
            for tok in sent.tokens:
                if tok.is_multiword() or tok.is_empty():
                    new_sent.tokens.append(tok)
                    continue
                tid = tok.id - 1
                script = word_scripts.get(tid, "k0:s0:a")
                lemma = _apply_edit_script(tok.form, script)
                new_tok = Token(
                    id=tok.id, form=tok.form, lemma=lemma,
                    upos=tok.upos, xpos=tok.xpos, feats=tok.feats,
                    head=tok.head, deprel=tok.deprel, deps=tok.deps, misc=tok.misc,
                )
                new_sent.tokens.append(new_tok)
            results.append(new_sent)

        return results
