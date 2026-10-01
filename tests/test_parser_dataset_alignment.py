"""Regression test for HEAD label alignment in ParserDataset.

The arc classifier scores pairs of *sub-word sequence positions*, so gold
``head`` values (1-based CoNLL-U word indices, 0 == ROOT) must be translated
into sub-word positions before being used as training targets. Previously
the raw word index was used directly as the label, which only happened to
be correct when every word tokenized into exactly one sub-word; as soon as
any word split into multiple sub-words (very common with real tokenizers)
every head label after that point was silently wrong, preventing the parser
from learning anything.
"""
import torch

from src.data.conllu import Sentence, Token
from src.data.dataset import ParserDataset


class FakeEncoding(dict):
    def __init__(self, data, word_ids_list):
        super().__init__(data)
        self._word_ids_list = word_ids_list

    def word_ids(self, batch_index=0):
        return self._word_ids_list


class FakeTokenizer:
    """Simulates a tokenizer that splits the 2nd word into two sub-words.

    Sentence: "Run fastly now" (3 words, word2 "fastly" splits into two
    sub-word pieces "fast" + "##ly").

    Sequence layout: [CLS] Run fast ##ly now [SEP]
    Positions:          0   1    2    3    4   5
    word_ids:        None   0    1    1    2  None
    """

    def __call__(self, forms, **kwargs):
        seq_len = 6
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        word_ids_list = [None, 0, 1, 1, 2, None]
        return FakeEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            word_ids_list,
        )


def _make_dataset(sentences):
    ds = ParserDataset.__new__(ParserDataset)
    ds.hf_tok = FakeTokenizer()
    ds.max_length = 16
    ds.deprel_vocab = {"<PAD>": 0, "<UNK>": 1, "root": 2, "advmod": 3, "obj": 4}
    ds.sentences = sentences
    return ds


def test_head_labels_point_to_subword_positions_not_word_indices():
    # CoNLL-U: word1 "Run" (head=0, ROOT); word2 "fastly" (head=1, modifies
    # "Run"); word3 "now" (head=1, modifies "Run" too).
    sent = Sentence(tokens=[
        Token(id=1, form="Run", head=0, deprel="root"),
        Token(id=2, form="fastly", head=1, deprel="advmod"),
        Token(id=3, form="now", head=1, deprel="advmod"),
    ])

    ds = _make_dataset([sent])
    item = ds[0]
    head_labels = item["head_labels"].tolist()

    # word0 ("Run") is at position 1 and attaches to ROOT -> position 0.
    assert head_labels[1] == 0
    # word1 ("fastly") first sub-word is at position 2 and attaches to
    # word0, whose first sub-word is at position 1 (NOT word index 1).
    assert head_labels[2] == 1
    # word2 ("now") is at position 4 (after the 2 sub-words of "fastly")
    # and also attaches to word0 at position 1.
    assert head_labels[4] == 1

    # Non-first-subword / special-token positions remain ignored.
    assert head_labels[0] == -100  # [CLS]
    assert head_labels[3] == -100  # "##ly" continuation sub-word
    assert head_labels[5] == -100  # [SEP]


def test_head_label_ignored_when_head_word_is_truncated():
    # word3 "now" points to a head word (id=4) that doesn't exist in the
    # tokenized sentence (e.g. truncated by max_length). The label must be
    # left as -100 (ignored by the loss) rather than silently defaulting to
    # ROOT, which would inject a wrong training signal.
    sent = Sentence(tokens=[
        Token(id=1, form="Run", head=0, deprel="root"),
        Token(id=2, form="fastly", head=1, deprel="advmod"),
        Token(id=3, form="now", head=4, deprel="advmod"),
    ])

    ds = _make_dataset([sent])
    item = ds[0]
    head_labels = item["head_labels"].tolist()
    deprel_labels = item["deprel_labels"].tolist()

    assert head_labels[4] == -100
    assert deprel_labels[4] == -100


if __name__ == "__main__":
    test_head_labels_point_to_subword_positions_not_word_indices()
    print("Parser dataset alignment test passed!")
