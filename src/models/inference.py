"""Shared, right-padded batches for sentence-level inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.conllu import Sentence, Token
from src.models.base import BorgBaseModel


@dataclass
class SentenceBatch:
    indices: List[int]
    tokens: List[List[Token]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    word_ids: List[List[int | None]]
    word_positions: List[Dict[int, int]]


def sentence_batches(
    model: BorgBaseModel,
    sentences: List[Sentence],
) -> Iterator[SentenceBatch]:
    batch_size = model.config.eval_batch_size
    if batch_size < 1:
        raise ValueError("eval_batch_size must be positive")

    examples = [
        (index, sentence.regular_tokens())
        for index, sentence in enumerate(sentences)
    ]
    examples = sorted(
        (example for example in examples if example[1]),
        key=lambda example: sum(len(token.form) for token in example[1]),
    )
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        encoding = model.hf_tokenizer(
            [[token.form for token in tokens] for _, tokens in chunk],
            is_split_into_words=True,
            max_length=model.config.max_seq_length,
            truncation=True,
            padding=False,
        )
        word_ids = [
            encoding.word_ids(batch_index=index)
            for index in range(len(chunk))
        ]
        word_positions = []
        for ids in word_ids:
            positions: Dict[int, int] = {}
            for position, word_id in enumerate(ids):
                if word_id is not None:
                    positions.setdefault(word_id, position)
            word_positions.append(positions)

        # ROOT is position zero in the parser, irrespective of tokenizer padding_side.
        yield SentenceBatch(
            indices=[index for index, _ in chunk],
            tokens=[tokens for _, tokens in chunk],
            input_ids=pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in encoding["input_ids"]],
                batch_first=True,
                padding_value=model.hf_tokenizer.pad_token_id,
            ),
            attention_mask=pad_sequence(
                [torch.tensor(mask, dtype=torch.long) for mask in encoding["attention_mask"]],
                batch_first=True,
                padding_value=0,
            ),
            word_ids=word_ids,
            word_positions=word_positions,
        )
