"""Tokenizer model: sentence and token boundary detection."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import TokenizerDataset
from src.models.base import BorgBaseModel
from src.models.checkpoints import save_training_models
from src.models.evaluation import (
    average_f1,
    evaluate_predictions,
    print_validation_metrics,
    tokenizer_validation_text,
)


_LABELS = {0: "C", 1: "T", 2: "S"}  # Continuation / Token-start / Sentence-start


class TokenizerModel(BorgBaseModel):
    """Predicts sentence/token boundaries at sub-word level."""

    NUM_LABELS = 3  # CONTINUATION, TOKEN_START, SENTENCE_START

    def __init__(self, config: BorgConfig):
        super().__init__(config, "tokenizer")
        self.classifier = nn.Linear(self.hidden_size, self.NUM_LABELS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)  # (B, L, H)
        return self.classifier(hidden)  # (B, L, num_labels)

    # ------------------------------------------------------------------
    def _get_extras(self) -> Dict[str, Any]:
        return {"classifier": self.classifier.state_dict()}

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        # `BorgBaseModel.load` reconstructs instances via `BorgBaseModel.__init__`
        # rather than `TokenizerModel.__init__`, so `self.classifier` may not
        # exist yet. Rebuild it before loading the saved weights.
        self.classifier = nn.Linear(self.hidden_size, self.NUM_LABELS)
        self.classifier.load_state_dict(extras["classifier"])

    # ------------------------------------------------------------------
    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
    ) -> "TokenizerModel":
        device = config.resolve_device()
        device_type = torch.device(device).type

        torch.manual_seed(config.seed)

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

        best_score = -1.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"[Tokenizer] Epoch {epoch + 1}",
            )

            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device_type,
                    dtype=config.dtype,
                    enabled=device_type == "cuda",
                ):
                    logits = model(
                        input_ids,
                        attention_mask,
                    )  # (B, L, C)

                    loss = loss_fn(
                        logits.view(-1, TokenizerModel.NUM_LABELS),
                        labels.view(-1),
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
                    avg_loss=f"{total_loss / max(progress.n, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            avg_loss = total_loss / len(train_loader)

            predicted_sentences = model.predict(
                tokenizer_validation_text(dev_sentences)
            )

            metrics = evaluate_predictions(
                dev_sentences,
                predicted_sentences,
            )

            print(f"  loss={avg_loss:.4f}")

            print_validation_metrics(
                metrics,
                ["Tokens", "Sentences"],
            )

            score = average_f1(
                metrics,
                ["Tokens", "Sentences"],
            )

            best_score = save_training_models(
                model,
                model_path,
                score,
                best_score,
            )

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
        score_counts = torch.zeros(len(token_ids))
        batch_size = self.config.eval_batch_size
        if batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
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
                    logits = self(
                        input_ids,
                        attention_mask,
                    ).float().cpu()

                for row, (start, window) in enumerate(zip(starts, windows)):
                    special_mask = hf_tok.get_special_tokens_mask(
                        window,
                        already_has_special_tokens=False,
                    )
                    content_positions = [
                        index for index, is_special in enumerate(special_mask)
                        if not is_special
                    ]
                    score_sums[start:start + len(window)] += logits[row, content_positions]
                    score_counts[start:start + len(window)] += 1

        preds = (score_sums / score_counts.unsqueeze(1)).argmax(-1).tolist()

        sentences: List[Sentence] = []
        current_sentence: Optional[Sentence] = None
        current_form_chars: List[str] = []
        current_token_id = 1

        # Track the end of the last processed subword to detect whitespace.
        last_end = 0
        pending_whitespace = False

        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                continue  # special token

            label = preds[i]
            raw_subword = text[start:end]

            # Some tokenizers (e.g. SentencePiece-based ones used by
            # DeBERTa-v3) include leading whitespace inside the offset span
            # of a sub-word that starts a new word. Split that whitespace
            # out so `subword` only holds the actual sub-word content.
            subword = raw_subword.lstrip()

            leading_ws = raw_subword[
                : len(raw_subword) - len(subword)
            ]

            # Detect whitespace before this token/subword: either a gap
            # between the previous subword and this one, or whitespace
            # embedded at the start of this subword's own offset span.
            whitespace_before = (
                text[last_end:start] + leading_ws
            )

            if not subword:
                pending_whitespace = (
                    pending_whitespace
                    or bool(whitespace_before)
                    or raw_subword.isspace()
                )

                last_end = end
                continue

            if pending_whitespace:
                whitespace_before = " " + whitespace_before

            pending_whitespace = False

            if label == TokenizerDataset.SENTENCE_START:
                # Flush any pending token from the previous sentence.
                if (
                    current_form_chars
                    and current_sentence is not None
                ):
                    # The token that just ended has space_after if
                    # whitespace precedes this new sentence.
                    has_space = len(whitespace_before) > 0

                    current_sentence.tokens.append(
                        Token(
                            id=current_token_id,
                            form="".join(current_form_chars),
                            space_after=has_space,
                        )
                    )

                # Flush old sentence.
                if (
                    current_sentence is not None
                    and current_sentence.tokens
                ):
                    sentences.append(current_sentence)

                current_sentence = Sentence()
                current_form_chars = [subword]
                current_token_id = 1

            elif label == TokenizerDataset.TOKEN_START:
                if current_sentence is None:
                    current_sentence = Sentence()
                    current_token_id = 1

                if current_form_chars:
                    # Token ends here. It has space_after if whitespace
                    # exists before this new token.
                    has_space = len(whitespace_before) > 0

                    current_sentence.tokens.append(
                        Token(
                            id=current_token_id,
                            form="".join(current_form_chars),
                            space_after=has_space,
                        )
                    )

                    current_token_id += 1

                current_form_chars = [subword]

            else:  # CONTINUATION
                if current_sentence is None:
                    current_sentence = Sentence()
                    current_token_id = 1

                # If we have a continuation but there was whitespace
                # before it, this is technically a model error
                # (continuation should be adjacent), but we handle it
                # by appending the subword.
                current_form_chars.append(subword)

            last_end = end

        # Flush remaining.
        if (
            current_form_chars
            and current_sentence is not None
        ):
            # Check if there's whitespace at the very end of the text.
            has_space = (
                pending_whitespace
                or len(text[last_end:]) > 0
            )

            current_sentence.tokens.append(
                Token(
                    id=current_token_id,
                    form="".join(current_form_chars),
                    space_after=has_space,
                )
            )

        if (
            current_sentence is not None
            and current_sentence.tokens
        ):
            sentences.append(current_sentence)

        return sentences