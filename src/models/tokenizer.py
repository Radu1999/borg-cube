"""Tokenizer model: sentence and token boundary detection."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import TokenizerDataset
from eval.conll18_ud_eval import EvalResult
from src.models.base import BorgBaseModel
from src.models.checkpoints import TrainingState
from src.models.evaluation import (
    average_f1,
    evaluate_predictions,
    print_validation_metrics,
    tokenizer_validation_text,
)


_LABELS = {0: "C", 1: "T", 2: "S"}  # Continuation / Token-start / Sentence-start


def _mwt_surface_metrics(
    gold_sentences: List[Sentence],
    predicted_sentences: List[Sentence],
    text: str,
) -> EvalResult:
    def positive_spans(sentences: List[Sentence], *, gold: bool) -> set[tuple[int, int]]:
        spans = set()
        cursor = 0
        for sentence in sentences:
            for token in sentence.surface_tokens():
                start = text.find(token.form, cursor)
                if start < 0:
                    continue
                end = start + len(token.form)
                is_mwt = token.is_multiword() if gold else token.is_mwt
                if is_mwt:
                    spans.add((start, end))
                cursor = end
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
        return spans

    gold_spans = positive_spans(gold_sentences, gold=True)
    predicted_spans = positive_spans(predicted_sentences, gold=False)
    true_positives = len(gold_spans & predicted_spans)
    false_positives = len(predicted_spans - gold_spans)
    false_negatives = len(gold_spans - predicted_spans)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return EvalResult(
        precision=precision,
        recall=recall,
        f1=f_score,
        aligned_count=true_positives,
        gold_count=len(gold_spans),
        system_count=len(predicted_spans),
    )


class TokenizerModel(BorgBaseModel):
    """Predicts sentence/token boundaries at sub-word level."""

    NUM_LABELS = 3  # CONTINUATION, TOKEN_START, SENTENCE_START

    def __init__(self, config: BorgConfig):
        super().__init__(config, "tokenizer")
        self.classifier = nn.Linear(self.hidden_size, self.NUM_LABELS)
        self.mwt_classifier = nn.Linear(self.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask)  # (B, L, H)
        return self.classifier(hidden), self.mwt_classifier(hidden)

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {
            "classifier": self.classifier.state_dict(),
            "mwt_classifier": self.mwt_classifier.state_dict(),
        }

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        # `BorgBaseModel.load` reconstructs instances via `BorgBaseModel.__init__`
        # rather than `TokenizerModel.__init__`, so `self.classifier` may not
        # exist yet. Rebuild it before loading the saved weights.
        self.classifier = nn.Linear(self.hidden_size, self.NUM_LABELS)
        self.classifier.load_state_dict(extras["classifier"])
        self.mwt_classifier = nn.Linear(self.hidden_size, 2)
        if "mwt_classifier" in extras:
            self.mwt_classifier.load_state_dict(extras["mwt_classifier"])
        else:
            nn.init.zeros_(self.mwt_classifier.weight)
            nn.init.zeros_(self.mwt_classifier.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
        *,
        resume: bool = False,
    ) -> "TokenizerModel":
        device = config.resolve_device()
        device_type = torch.device(device).type

        torch.manual_seed(config.seed)

        if resume:
            model = TokenizerModel.load(os.path.join(model_path, "last"), config).to(device)
        else:
            model = TokenizerModel(config).to(device)

        train_ds = TokenizerDataset(
            train_sentences,
            config.model_name,
            config.max_seq_length,
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

        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        with TrainingState(
            model, model_path, optimizer, scheduler, train_loader, resume=resume,
        ) as state:
            for epoch in range(state.epoch, config.num_epochs):
                model.train()
                progress = state.progress(epoch, "Tokenizer")

                for batch in progress:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    mwt_labels = batch["mwt_labels"].to(device)

                    optimizer.zero_grad()

                    with torch.autocast(
                        device_type=device_type,
                        dtype=config.dtype,
                        enabled=device_type == "cuda",
                    ):
                        boundary_logits, mwt_logits = model(
                            input_ids,
                            attention_mask,
                        )

                        loss = loss_fn(
                            boundary_logits.view(-1, TokenizerModel.NUM_LABELS),
                            labels.view(-1),
                        ) + loss_fn(
                            mwt_logits.view(-1, 2),
                            mwt_labels.view(-1),
                        )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    optimizer.step()
                    scheduler.step()

                    state.step(loss.item())

                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg_loss=f"{state.total_loss / state.batch:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

                avg_loss = state.total_loss / len(train_loader)

                validation_text = tokenizer_validation_text(dev_sentences)
                predicted_sentences = model.predict(validation_text)

                metrics = evaluate_predictions(
                    dev_sentences,
                    predicted_sentences,
                )

                print(f"  loss={avg_loss:.4f}")

                metrics["MWT"] = _mwt_surface_metrics(
                    dev_sentences,
                    predicted_sentences,
                    validation_text,
                )
                print_validation_metrics(
                    metrics,
                    ["Tokens", "Sentences", "MWT"],
                )

                score = average_f1(
                    metrics,
                    ["Tokens", "Sentences"],
                )

                state.finish_epoch(score)

        return model

    # ------------------------------------------------------------------
    def predict(self, text: str, *, show_progress: bool = True) -> List[Sentence]:
        """Segment *text* into sentences and tokens, preserving whitespace."""
        device = self.config.resolve_device()
        device_type = torch.device(device).type

        self.eval()
        self.to(device)

        hf_tok = self.hf_tokenizer

        encoding = hf_tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        max_seq_length = min(
            self.config.max_seq_length,
            self.encoder.config.max_position_embeddings,
        )

        special_tokens = hf_tok.num_special_tokens_to_add(pair=False)
        window_size = max_seq_length - special_tokens

        if window_size < 1:
            raise ValueError(
                "max_seq_length must leave room for at least one token"
            )

        overlap = min(window_size // 4, window_size - 1)
        window_step = window_size - overlap

        score_sums = torch.zeros(
            (len(token_ids), self.NUM_LABELS)
        )
        mwt_score_sums = torch.zeros((len(token_ids), 2))
        score_counts = torch.zeros(len(token_ids))
        batch_size = self.config.eval_batch_size
        if batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if device_type == "mps":
            batch_size = 1
        window_starts = range(0, len(token_ids), window_step)

        with torch.inference_mode():
            for batch_start in tqdm(
                range(0, len(window_starts), batch_size),
                desc="[TokenizerModel] Predict",
                unit="batch",
                disable=not show_progress or not token_ids,
            ):
                starts = window_starts[batch_start:batch_start + batch_size]
                windows = [
                    token_ids[start:start + window_size]
                    for start in starts
                ]
                model_inputs = [
                    hf_tok.prepare_for_model(
                        window,
                        add_special_tokens=True,
                        return_attention_mask=True,
                    )
                    for window in windows
                ]
                input_ids = pad_sequence(
                    [torch.tensor(item["input_ids"], dtype=torch.long) for item in model_inputs],
                    batch_first=True,
                    padding_value=hf_tok.pad_token_id,
                ).to(device)
                attention_mask = pad_sequence(
                    [torch.tensor(item["attention_mask"], dtype=torch.long) for item in model_inputs],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                with torch.autocast(
                    device_type=device_type,
                    dtype=self.config.dtype,
                    enabled=device_type == "cuda",
                ):
                    model_outputs = self(
                        input_ids,
                        attention_mask,
                    )
                    boundary_logits, mwt_logits = model_outputs
                    boundary_logits = boundary_logits.float().cpu()
                    mwt_logits = mwt_logits.float().cpu()

                for row, (start, window) in enumerate(zip(starts, windows)):
                    try:
                        special_mask = hf_tok.get_special_tokens_mask(
                            window,
                            already_has_special_tokens=False,
                        )
                    except AssertionError:
                        special_mask = hf_tok.get_special_tokens_mask(
                            model_inputs[row]["input_ids"],
                            already_has_special_tokens=True,
                        )
                    content_positions = [
                        index for index, is_special in enumerate(special_mask)
                        if not is_special
                    ]
                    score_sums[start:start + len(window)] += boundary_logits[row, content_positions]
                    mwt_score_sums[start:start + len(window)] += mwt_logits[row, content_positions]
                    score_counts[start:start + len(window)] += 1

        scores = score_sums / score_counts.unsqueeze(1)
        mwt_scores = mwt_score_sums / score_counts.unsqueeze(1)
        character_positions = []
        subword_indices = []
        for index, (start, end) in enumerate(offset_mapping):
            while start < end and text[start].isspace():
                start += 1
            if start < end:
                character_positions.append(start)
                subword_indices.append(index)

        positions, inverse = torch.unique(
            torch.tensor(character_positions, dtype=torch.long),
            sorted=True,
            return_inverse=True,
        )
        character_scores = scores.new_zeros((len(positions), self.NUM_LABELS))
        character_scores.index_add_(0, inverse, scores[subword_indices])
        labels = character_scores.argmax(-1).tolist()
        character_mwt_scores = mwt_scores.new_zeros((len(positions), 2))
        character_mwt_scores.index_add_(0, inverse, mwt_scores[subword_indices])
        mwt_labels = character_mwt_scores.argmax(-1).tolist()

        first_character = next(
            (index for index, char in enumerate(text) if not char.isspace()),
            None,
        )
        if first_character is None:
            return []

        first_mwt_label = mwt_labels[0] if mwt_labels else 0
        boundaries = [(first_character, TokenizerDataset.SENTENCE_START, first_mwt_label)]
        boundaries.extend(
            (position, label, mwt_label)
            for position, label, mwt_label in zip(
                positions.tolist()[1:], labels[1:], mwt_labels[1:]
            )
            if label != TokenizerDataset.CONTINUATION
        )

        sentences: List[Sentence] = []
        current_sentence = Sentence()
        for index, (start, label, is_mwt) in enumerate(boundaries):
            if label == TokenizerDataset.SENTENCE_START and current_sentence.tokens:
                sentences.append(current_sentence)
                current_sentence = Sentence()

            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
            token_text = text[start:end]
            current_sentence.tokens.append(
                Token(
                    id=len(current_sentence.tokens) + 1,
                    form="".join(token_text.split()),
                    space_after=token_text[-1].isspace(),
                    is_mwt=bool(is_mwt),
                )
            )

        if current_sentence.tokens:
            sentences.append(current_sentence)

        return sentences