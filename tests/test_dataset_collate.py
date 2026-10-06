import torch

from src.data.dataset import (
    LemmatizerDataset,
    ParserDataset,
    TaggerDataset,
    TokenizerDataset,
)


class FakeTokenizer:
    pad_token_id = 99


def _dataset(dataset_type):
    dataset = dataset_type.__new__(dataset_type)
    dataset.hf_tok = FakeTokenizer()
    return dataset


def _batch(*examples):
    return [dict(example) for example in examples]


def test_tokenizer_collate_pads_tokenizer_outputs():
    batch = _dataset(TokenizerDataset).collate_fn(
        _batch(
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([2, 0]),
            },
            {
                "input_ids": torch.tensor([3]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([1]),
            },
        )
    )

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].tolist() == [[1, 2], [3, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[2, 0], [1, -100]]


def test_tagger_collate_pads_tagger_outputs():
    batch = _dataset(TaggerDataset).collate_fn(
        _batch(
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "upos_labels": torch.tensor([2, 3]),
                "xpos_labels": torch.tensor([4, 5]),
                "feats_labels": torch.tensor([6, 7]),
            },
            {
                "input_ids": torch.tensor([8]),
                "attention_mask": torch.tensor([1]),
                "upos_labels": torch.tensor([9]),
                "xpos_labels": torch.tensor([10]),
                "feats_labels": torch.tensor([11]),
            },
        )
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "upos_labels",
        "xpos_labels",
        "feats_labels",
    }
    assert batch["upos_labels"].tolist() == [[2, 3], [9, -100]]
    assert batch["xpos_labels"].tolist() == [[4, 5], [10, -100]]
    assert batch["feats_labels"].tolist() == [[6, 7], [11, -100]]


def test_lemmatizer_collate_pads_inputs_and_targets():
    batch = _dataset(LemmatizerDataset).collate_fn(
        _batch(
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "upos_ids": torch.tensor([2, 3]),
                "script_labels": torch.tensor([4, 5]),
            },
            {
                "input_ids": torch.tensor([6]),
                "attention_mask": torch.tensor([1]),
                "upos_ids": torch.tensor([7]),
                "script_labels": torch.tensor([8]),
            },
        )
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "upos_ids",
        "script_labels",
    }
    assert batch["input_ids"].tolist() == [[1, 2], [6, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["upos_ids"].tolist() == [[2, 3], [7, 0]]
    assert batch["script_labels"].tolist() == [[4, 5], [8, -100]]


def test_parser_collate_pads_parser_outputs():
    batch = _dataset(ParserDataset).collate_fn(
        _batch(
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "head_labels": torch.tensor([0, 1]),
                "deprel_labels": torch.tensor([2, 3]),
            },
            {
                "input_ids": torch.tensor([4]),
                "attention_mask": torch.tensor([1]),
                "head_labels": torch.tensor([0]),
                "deprel_labels": torch.tensor([5]),
            },
        )
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "head_labels",
        "deprel_labels",
    }
    assert batch["head_labels"].tolist() == [[0, 1], [0, -100]]
    assert batch["deprel_labels"].tolist() == [[2, 3], [5, -100]]
