"""Character-level multiword-token expansion model."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.data.dataset import MWTDataset
from src.models.checkpoints import TrainingState


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


class MWTModel(nn.Module):
    """Expand tokenizer-identified MWT tokens using characters only."""

    CHAR_EMBEDDING_SIZE = 64
    HIDDEN_SIZE = 128
    MAX_OUTPUT_LENGTH = 128

    def __init__(
        self,
        config: BorgConfig,
        output_vocab: Optional[Dict[str, int]] = None,
        input_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.config = config
        self.component = "mwt"
        specials = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.output_vocab = output_vocab or specials.copy()
        self.input_vocab = input_vocab or specials.copy()
        self._build_heads()

    def _build_heads(self) -> None:
        embedding_size = self.CHAR_EMBEDDING_SIZE
        hidden_size = self.HIDDEN_SIZE
        self.input_embedding = nn.Embedding(
            len(self.input_vocab), embedding_size, padding_idx=self.input_vocab["<PAD>"]
        )
        self.input_encoder = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        encoder_size = hidden_size * 2
        self.encoder_projection = nn.Linear(encoder_size, encoder_size, bias=False)
        self.query_projection = nn.Linear(encoder_size, encoder_size, bias=False)
        self.attention_score = nn.Linear(encoder_size, 1, bias=False)
        self.output_embedding = nn.Embedding(
            len(self.output_vocab), embedding_size,
            padding_idx=self.output_vocab["<PAD>"],
        )
        self.output_decoder = nn.LSTM(
            embedding_size + encoder_size,
            encoder_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_logit = nn.Linear(encoder_size * 2, len(self.output_vocab))

    def _encode(self, input_char_ids: torch.Tensor):
        mask = input_char_ids.ne(self.input_vocab["<PAD>"])
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        embedded = self.input_embedding(input_char_ids)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.input_encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=input_char_ids.shape[1]
        )
        return encoded, mask, self.encoder_projection(encoded)

    def _decode_step(self, previous_ids, hidden, encoded, mask, projected):
        query = self.query_projection(hidden[0][-1]).unsqueeze(1)
        scores = self.attention_score(torch.tanh(projected + query)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoded)
        decoder_input = torch.cat(
            [self.output_embedding(previous_ids), context], dim=-1
        )
        decoder_output, hidden = self.output_decoder(decoder_input, hidden)
        logits = self.output_logit(torch.cat([decoder_output, context], dim=-1))
        return logits, hidden

    def forward(
        self,
        input_char_ids: torch.Tensor,
        output_char_ids: Optional[torch.Tensor] = None,
        max_output_length: int = MAX_OUTPUT_LENGTH,
    ) -> Dict[str, Any]:
        encoded, mask, projected = self._encode(input_char_ids)
        batch_size = input_char_ids.shape[0]
        hidden_size = self.HIDDEN_SIZE * 2
        hidden = (
            encoded.new_zeros((1, batch_size, hidden_size)),
            encoded.new_zeros((1, batch_size, hidden_size)),
        )

        if output_char_ids is not None:
            decoder_inputs = output_char_ids[:, :-1].clamp(min=0)
            logits = []
            for step in range(decoder_inputs.shape[1]):
                step_logits, hidden = self._decode_step(
                    decoder_inputs[:, step:step + 1], hidden,
                    encoded, mask, projected,
                )
                logits.append(step_logits)
            return {"logits": torch.cat(logits, dim=1)}

        previous_ids = torch.full(
            (batch_size, 1), self.output_vocab["<SOS>"],
            dtype=torch.long, device=input_char_ids.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_char_ids.device)
        strings = [""] * batch_size
        inverse_vocab = {index: char for char, index in self.output_vocab.items()}
        for _ in range(max_output_length):
            logits, hidden = self._decode_step(
                previous_ids, hidden, encoded, mask, projected
            )
            previous_ids = logits[:, 0].argmax(-1, keepdim=True)
            for row, char_id in enumerate(previous_ids[:, 0].tolist()):
                if char_id == self.output_vocab["<EOS>"]:
                    finished[row] = True
                elif not finished[row] and char_id not in (
                    self.output_vocab["<PAD>"],
                    self.output_vocab["<SOS>"],
                    self.output_vocab["<UNK>"],
                ):
                    strings[row] += inverse_vocab.get(char_id, "")
            if finished.all():
                break
        return {"strings": strings}

    def _get_extras(self) -> Dict[str, Any]:
        return {
            "input_vocab": self.input_vocab,
            "output_vocab": self.output_vocab,
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "borg_config.json"), "w") as output:
            values = self.config.to_dict()
            values["component"] = self.component
            values["model_type"] = "character_mwt_v1"
            json.dump(values, output, indent=2)
        torch.save(
            {**self._get_extras(), "model_state": self.state_dict()},
            os.path.join(path, "extras.pt"),
        )

    @classmethod
    def load(cls, path: str, config: Optional[BorgConfig] = None) -> "MWTModel":
        with open(os.path.join(path, "borg_config.json")) as source:
            values = json.load(source)
        if values.get("model_type") != "character_mwt_v1":
            raise ValueError(
                "This MWT checkpoint uses the previous BERT-conditioned model; "
                "train a new character-only MWT model."
            )
        values.pop("component", None)
        values.pop("model_type", None)
        checkpoint_config = BorgConfig.from_dict(values)
        extras = torch.load(
            os.path.join(path, "extras.pt"), map_location="cpu", weights_only=False
        )
        model = cls(
            config or checkpoint_config,
            output_vocab=extras["output_vocab"],
            input_vocab=extras["input_vocab"],
        )
        model.load_state_dict(extras["model_state"])
        return model

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
        torch.manual_seed(config.seed)

        if resume:
            model = MWTModel.load(os.path.join(model_path, "last"), config).to(device)
            train_ds = MWTDataset(
                train_sentences, model.output_vocab, model.input_vocab
            )
        else:
            train_ds = MWTDataset(train_sentences)
            model = MWTModel(
                config, train_ds.output_vocab, train_ds.input_vocab
            ).to(device)
        if not len(train_ds):
            raise ValueError("The MWT training data contains no MWT tokens")

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        def linear_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(
                0.0,
                (total_steps - step) / max(total_steps - warmup_steps, 1),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        dev_ds = MWTDataset(
            dev_sentences, model.output_vocab, model.input_vocab
        )

        with TrainingState(
            model, model_path, optimizer, scheduler, train_loader, resume=resume,
        ) as state:
            for epoch in range(state.epoch, config.num_epochs):
                model.train()
                progress = state.progress(epoch, "MWT")
                for batch in progress:
                    input_char_ids = batch["input_char_ids"].to(device)
                    output_char_ids = batch["output_char_ids"].to(device)
                    optimizer.zero_grad()
                    outputs = model(input_char_ids, output_char_ids)
                    targets = output_char_ids[:, 1:]
                    loss = loss_fn(
                        outputs["logits"].reshape(-1, outputs["logits"].shape[-1]),
                        targets.reshape(-1),
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
                expected = [expansion for _, expansion in dev_ds.examples]
                predicted = [
                    expansion for _, expansion in model._predict_pairs(
                        [form for form, _ in dev_ds.examples], show_progress=False
                    )
                ]
                precision, recall = _mwt_precision_recall(expected, predicted)
                score = (
                    sum(prediction == target for prediction, target in zip(predicted, expected))
                    / len(expected)
                    if expected else 0.0
                )
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
        results = [
            [token.form for token in sentence.surface_tokens()]
            for sentence in sentences
        ]
        candidates = []
        for sentence_index, sentence in enumerate(sentences):
            for token_index, token in enumerate(sentence.surface_tokens()):
                if token.is_mwt:
                    candidates.append((sentence_index, token_index, token.form))

        batch_size = self.config.eval_batch_size
        if batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        for start in tqdm(
            range(0, len(candidates), batch_size),
            desc="[MWTModel] Predict",
            unit="batch",
            disable=not show_progress or not candidates,
        ):
            batch = candidates[start:start + batch_size]
            char_sequences = [
                MWTDataset.encode_chars(form, self.input_vocab)
                for _, _, form in batch
            ]
            input_char_ids = pad_sequence(
                char_sequences,
                batch_first=True,
                padding_value=self.input_vocab["<PAD>"],
            ).to(device)
            with torch.inference_mode():
                expansions = self(input_char_ids)["strings"]
            for (sentence_index, token_index, _), expansion in zip(batch, expansions):
                results[sentence_index][token_index] = expansion
        return results

    def _predict_pairs(
        self,
        forms: List[str],
        *,
        show_progress: bool = True,
    ) -> List[tuple[str, str]]:
        device = self.config.resolve_device()
        batch_size = self.config.eval_batch_size
        if batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        predictions = []
        for start in tqdm(
            range(0, len(forms), batch_size),
            desc="[MWTModel] Validate",
            unit="batch",
            disable=not show_progress or not forms,
        ):
            chunk = forms[start:start + batch_size]
            inputs = pad_sequence(
                [MWTDataset.encode_chars(form, self.input_vocab) for form in chunk],
                batch_first=True,
                padding_value=self.input_vocab["<PAD>"],
            ).to(device)
            with torch.inference_mode():
                expansions = self(inputs)["strings"]
            predictions.extend(zip(chunk, expansions))
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
            sentences, show_progress=show_progress
        )

        results = []
        for sentence_index, sentence in enumerate(sentences):
            if any(token.is_multiword() for token in sentence.tokens):
                results.append(sentence)
                continue
            new_tokens = []
            next_id = 1
            surface_index = 0
            for token in sentence.tokens:
                if token.is_empty():
                    new_tokens.append(token)
                    continue
                expansion = predicted_by_sentence[sentence_index][surface_index]
                surface_index += 1
                parts = expansion.split()
                if not token.is_mwt or len(parts) < 2:
                    new_tokens.append(replace(token, id=next_id, is_mwt=False))
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
                            token.space_after if part_index == len(parts) - 1 else True
                        ),
                    ))
                next_id = end_id + 1
            results.append(Sentence(tokens=new_tokens, comments=sentence.comments))
        return results