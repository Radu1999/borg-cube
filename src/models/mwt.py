"""Sequence-to-sequence multiword-token expansion model."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import MWTDataset
from src.models.base import BorgBaseModel
from src.models.checkpoints import TrainingState
from src.models.inference import sentence_batches


def _mwt_precision_recall(
    expected: List[str],
    predicted: List[str],
) -> tuple[float, float]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gold, guess in zip(expected, predicted):
        gold_is_mwt = len(gold.split()) > 1
        guess_is_mwt = len(guess.split()) > 1
        correct_expansion = gold_is_mwt and guess == gold
        true_positives += correct_expansion
        false_positives += guess_is_mwt and not correct_expansion
        false_negatives += gold_is_mwt and not correct_expansion

    predicted_mwt_count = true_positives + false_positives
    gold_mwt_count = true_positives + false_negatives
    precision = true_positives / predicted_mwt_count if predicted_mwt_count else 0.0
    recall = true_positives / gold_mwt_count if gold_mwt_count else 0.0
    return precision, recall


class MWTModel(BorgBaseModel):
    """Expand each surface form into one or more syntactic words."""

    MAX_OUTPUT_LENGTH = 128
    CHAR_EMBEDDING_SIZE = 64

    def __init__(
        self,
        config: BorgConfig,
        output_vocab: Optional[Dict[str, int]] = None,
        input_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(config, "mwt")
        self.output_vocab = output_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }
        self.input_vocab = input_vocab or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }
        self._build_heads()

    def _build_heads(self) -> None:
        char_emb_dim = 64
        lstm_size = 200
        hidden_size = self.encoder.config.hidden_size
        self.input_embedding = nn.Embedding(
            len(self.input_vocab),
            char_emb_dim,
            padding_idx=self.input_vocab["<PAD>"],
        )
        self.input_encoder = nn.LSTM(
            input_size=hidden_size + char_emb_dim,
            hidden_size=lstm_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.output_embedding = nn.Embedding(
            len(self.output_vocab),
            char_emb_dim,
            padding_idx=self.output_vocab["<PAD>"],
        )
        self.output_decoder = nn.LSTM(
            input_size=lstm_size * 2 + char_emb_dim,
            hidden_size=lstm_size * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_size * 2,
            num_heads=1,
            batch_first=True,
            kdim=lstm_size * 2,
            vdim=lstm_size * 2,
        )
        self.output_logit = nn.Linear(
            lstm_size * 2,
            len(self.output_vocab),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_char_ids: torch.Tensor,
        output_char_ids: Optional[torch.Tensor] = None,
        max_output_length: int = MAX_OUTPUT_LENGTH,
    ) -> Dict[str, Any]:
        hidden = self.encode(input_ids, attention_mask)
        concat = hidden[:, 1:-1, :]
        concat = concat.reshape((concat.shape[0] * concat.shape[1], -1))
        input_char_ids = input_char_ids.reshape(
            input_char_ids.shape[0] * input_char_ids.shape[1],
            input_char_ids.shape[2],
        )
        output_char_ids = (
            output_char_ids.reshape(
                output_char_ids.shape[0] * output_char_ids.shape[1],
                output_char_ids.shape[2],
            )
            if output_char_ids is not None
            else None
        )

        input_char_emb = self.input_embedding(input_char_ids)
        concat = concat.unsqueeze(1).repeat(1, input_char_emb.shape[1], 1)
        input_embedding = torch.cat([concat, input_char_emb], dim=-1)
        encoder_states, _ = self.input_encoder(input_embedding)

        output_strings = [""] * input_char_ids.shape[0]
        finished = [False] * len(output_strings)
        inverse_output_vocab = {value: key for key, value in self.output_vocab.items()}
        output_logits = []
        last_output_embeddings = torch.zeros(
            (
                input_char_ids.shape[0],
                1,
                input_char_emb.shape[2] + encoder_states.shape[2],
            ),
            device=encoder_states.device,
        )
        decoder_hidden = None
        if output_char_ids is not None:
            output_char_ids = torch.clamp(output_char_ids, min=0)
            max_output_length = output_char_ids.shape[1]

        for step in range(max_output_length):
            decoder_output, decoder_hidden = self.output_decoder(
                last_output_embeddings,
                decoder_hidden,
            )
            logits = self.output_logit(decoder_output)
            output_logits.append(logits)
            last_output_char_ids = logits.argmax(-1)
            if output_char_ids is not None:
                output_char_emb = self.output_embedding(output_char_ids[:, step:step + 1])
            else:
                output_char_emb = self.output_embedding(last_output_char_ids)
                for row, char_id in enumerate(last_output_char_ids[:, 0].tolist()):
                    if char_id == self.output_vocab["<EOS>"]:
                        finished[row] = True
                    elif not finished[row] and char_id not in (
                        self.output_vocab["<PAD>"],
                        self.output_vocab["<SOS>"],
                        self.output_vocab["<UNK>"],
                    ):
                        output_strings[row] += inverse_output_vocab.get(char_id, "")
                if all(finished):
                    break
            attended, _ = self.attention(
                query=decoder_output,
                key=encoder_states,
                value=encoder_states,
            )
            last_output_embeddings = torch.cat([output_char_emb, attended], dim=-1)

        return {
            "logits": torch.cat(output_logits, dim=1),
            "strings": output_strings,
        }

    def _get_extras(self) -> Dict[str, Any]:
        return {
            "input_vocab": self.input_vocab,
            "output_vocab": self.output_vocab,
            "input_embedding": self.input_embedding.state_dict(),
            "input_encoder": self.input_encoder.state_dict(),
            "output_embedding": self.output_embedding.state_dict(),
            "output_decoder": self.output_decoder.state_dict(),
            "attention": self.attention.state_dict(),
            "output_logit": self.output_logit.state_dict(),
        }

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        self.input_vocab = extras["input_vocab"]
        self.output_vocab = extras["output_vocab"]
        self._build_heads()
        for name in (
            "input_embedding",
            "input_encoder",
            "output_embedding",
            "output_decoder",
            "attention",
            "output_logit",
        ):
            getattr(self, name).load_state_dict(extras[name])

    @staticmethod
    def train_model(
        train_sentences: List[Sentence],
        dev_sentences: List[Sentence],
        config: BorgConfig,
        model_path: str,
        *,
        resume: bool = False,
    ) -> "MWTModel":
        device = config.resolve_device()
        device_type = torch.device(device).type
        torch.manual_seed(config.seed)

        if resume:
            model = MWTModel.load(os.path.join(model_path, "last"), config).to(device)
            train_ds = MWTDataset(
                train_sentences,
                config.model_name,
                config.max_seq_length,
                model.output_vocab,
                model.input_vocab,
            )
        else:
            train_ds = MWTDataset(
                train_sentences,
                config.model_name,
                config.max_seq_length,
            )
            model = MWTModel(
                config,
                train_ds.output_vocab,
                train_ds.input_vocab,
            ).to(device)
        if not len(train_ds):
            raise ValueError("The MWT training data contains no surface tokens")

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * config.warmup_ratio),
            total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        dev_ds = MWTDataset(
            dev_sentences,
            config.model_name,
            config.max_seq_length,
            model.output_vocab,
            model.input_vocab,
        )

        with TrainingState(
            model,
            model_path,
            optimizer,
            scheduler,
            train_loader,
            resume=resume,
        ) as state:
            for epoch in range(state.epoch, config.num_epochs):
                model.train()
                progress = state.progress(epoch, "MWT")
                for batch in progress:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    input_char_ids = batch["input_char_ids"].to(device)
                    output_char_ids = batch["output_char_ids"].to(device)
                    optimizer.zero_grad()
                    with torch.autocast(
                        device_type=device_type,
                        dtype=config.dtype,
                        enabled=device_type == "cuda",
                    ):
                        outputs = model(
                            input_ids,
                            attention_mask,
                            input_char_ids,
                            output_char_ids,
                        )
                        loss = loss_fn(
                            outputs["logits"].view(-1, outputs["logits"].shape[-1]),
                            output_char_ids.view(-1),
                        )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    state.step(loss.item())
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg_loss=f"{state.total_loss / max(progress.n, 1):.4f}",
                    )

                model.eval()
                if any(sentence.surface_tokens() for sentence in dev_sentences):
                    expected = [
                        target
                        for sentence in dev_sentences
                        for token in sentence.surface_tokens()
                        for target in [
                            MWTDataset._expansion_targets(sentence).get(token.id, token.form)
                        ]
                    ]
                    predicted = [
                        expansion
                        for sentence_expansions in model._predict_expansions(
                            dev_sentences,
                            show_progress=False,
                        )
                        for expansion in sentence_expansions
                    ]
                    precision, recall = _mwt_precision_recall(expected, predicted)
                    score = sum(
                        prediction == target
                        for prediction, target in zip(predicted, expected)
                    ) / len(expected)
                else:
                    score = 0.0
                    precision = 0.0
                    recall = 0.0
                state.finish_epoch(score)
                print(f"  exact expansion accuracy={score:.4f}")
                print(f"  MWT precision={precision:.4f} recall={recall:.4f}")

        return model

    def _predict_expansions(
        self,
        sentences: List[Sentence],
        *,
        show_progress: bool = True,
    ) -> List[List[str]]:
        device = self.config.resolve_device()
        device_type = torch.device(device).type
        predictions = [
            [token.form for token in sentence.surface_tokens()]
            for sentence in sentences
        ]
        for batch in sentence_batches(self, sentences, show_progress=show_progress):
            if batch.input_char_ids is None:
                raise RuntimeError("MWT inference requires input character IDs")
            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=self.config.dtype,
                enabled=device_type == "cuda",
            ):
                outputs = self(
                    batch.input_ids.to(device),
                    batch.attention_mask.to(device),
                    batch.input_char_ids.to(device),
                )["strings"]
            sequence_width = batch.input_ids.shape[1] - 2
            for row, sentence_index in enumerate(batch.indices):
                for word_id, position in batch.word_positions[row].items():
                    predictions[sentence_index][word_id] = outputs[
                        row * sequence_width + position - 1
                    ]
        return predictions

    def predict(
        self,
        sentences: List[Sentence],
        *,
        show_progress: bool = True,
    ) -> List[Sentence]:
        self.eval()
        self.to(self.config.resolve_device())
        predicted_by_sentence = self._predict_expansions(
            sentences,
            show_progress=show_progress,
        )

        results = []
        for sentence_index, sentence in enumerate(sentences):
            if any(token.is_multiword() for token in sentence.tokens):
                results.append(sentence)
                continue
            new_tokens = []
            next_id = 1
            word_id = 0
            for token_index, token in enumerate(sentence.tokens):
                if token.is_multiword() or token.is_empty():
                    new_tokens.append(token)
                    continue
                expansion = predicted_by_sentence[sentence_index][word_id]
                word_id += 1
                parts = expansion.split()
                if len(parts) < 2:
                    new_tokens.append(replace(token, id=next_id))
                    next_id += 1
                    continue

                end_id = next_id + len(parts) - 1
                new_tokens.append(Token(
                    id=f"{next_id}-{end_id}",
                    form=token.form,
                    misc=token.misc,
                    space_after=token.space_after,
                ))
                for part_index, part in enumerate(parts):
                    new_tokens.append(Token(
                        id=next_id + part_index,
                        form=part,
                        space_after=(
                            token.space_after
                            if part_index == len(parts) - 1
                            else True
                        ),
                    ))
                next_id = end_id + 1
            results.append(Sentence(tokens=new_tokens, comments=sentence.comments))
        return results