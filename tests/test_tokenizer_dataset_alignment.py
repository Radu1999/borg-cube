"""Regression test for sub-word label alignment in TokenizerDataset.

Some fast tokenizers (e.g. the SentencePiece-based DeBERTa-v3 tokenizer)
include the leading whitespace character inside the offset span of a
sub-word that starts a new word (e.g. offset (1, 3) for " B" rather than
(2, 3) for "B"). If label alignment naively looks up the label of
``text[start]`` it picks up the whitespace's CONTINUATION label instead of
the real TOKEN_START/SENTENCE_START label of the word, silently corrupting
almost every training example.
"""
import torch

from src.data.conllu import Sentence, Token
from src.data.dataset import TokenizerDataset


def _make_dataset(text, char_labels, offset_mapping):
    ds = TokenizerDataset.__new__(TokenizerDataset)
    ds.max_length = len(offset_mapping)

    seq_len = len(offset_mapping)

    class FakeTokenizer:
        def __call__(self, _text, **kwargs):
            return {
                "input_ids": torch.zeros((1, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
                "offset_mapping": torch.tensor([offset_mapping]),
            }

    ds.hf_tok = FakeTokenizer()
    ds.examples = [(text, char_labels)]
    return ds


def test_label_alignment_skips_leading_whitespace_in_subword_offset():
    # Reconstructed text: "A B"
    #   index 0: 'A' -> SENTENCE_START
    #   index 1: ' ' -> CONTINUATION (space appended between tokens)
    #   index 2: 'B' -> TOKEN_START
    text = "A B"
    char_labels = [
        TokenizerDataset.SENTENCE_START,
        TokenizerDataset.CONTINUATION,
        TokenizerDataset.TOKEN_START,
    ]

    # Simulate a SentencePiece-style tokenizer where the second sub-word's
    # offset (1, 3) covers " B" (leading space included), not just "B".
    offset_mapping = [(0, 0), (0, 1), (1, 3), (0, 0)]

    ds = _make_dataset(text, char_labels, offset_mapping)
    item = ds[0]
    labels = item["labels"].tolist()

    assert labels[0] == -100  # special token
    assert labels[1] == TokenizerDataset.SENTENCE_START
    assert labels[2] == TokenizerDataset.TOKEN_START  # not CONTINUATION
    assert labels[3] == -100  # special token


def test_training_labels_use_multiword_surface_tokens():
    sentences = [
        Sentence(tokens=[
            Token(id="1-2", form="du", space_after=False),
            Token(id=1, form="de"),
            Token(id=2, form="le"),
            Token(id="2.1", form="empty"),
            Token(id=3, form="."),
        ]),
        Sentence(tokens=[Token(id=1, form="Next")]),
    ]
    dataset = TokenizerDataset.__new__(TokenizerDataset)
    dataset.max_length = 64
    dataset.examples = []

    dataset._build_examples(sentences)

    assert dataset.examples == [("du. Next ", [2, 0, 1, 0, 2, 0, 0, 0, 0])]


if __name__ == "__main__":
    test_label_alignment_skips_leading_whitespace_in_subword_offset()
    test_training_labels_use_multiword_surface_tokens()
    print("Alignment tests passed!")
