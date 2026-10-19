from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from src.data.conllu import Sentence, Token
from src.data.dataset import MWTDataset
from src.models.mwt import MWTModel, _mwt_precision_recall
from src.pipeline.pipeline import BorgPipeline


def test_mwt_precision_recall_requires_exact_gold_expansions():
    precision, recall = _mwt_precision_recall(
        ["In Dem", "auf das", "geht"],
        ["In Dem", "aufdas", "geh t"],
    )

    assert precision == 0.5
    assert recall == 0.5


def test_mwt_precision_recall_is_zero_when_denominators_are_empty():
    assert _mwt_precision_recall([], []) == (0.0, 0.0)
    assert _mwt_precision_recall(["ordinary"], ["ordinary"]) == (0.0, 0.0)


class StubHFTokenizer:
    pad_token_id = 0

    def __call__(self, forms, **kwargs):
        if kwargs.get("is_split_into_words"):
            if forms and isinstance(forms[0], str):
                return StubEncoding(
                    [[1, *range(4, 4 + len(forms)), 2]],
                    [[None, *range(len(forms)), None]],
                )
            sequences = []
            mappings = []
            for words in forms:
                sequences.append([1, *range(4, 4 + len(words)), 2])
                mappings.append([None, *range(len(words)), None])
            return StubEncoding(sequences, mappings)
        ids = [ord(char) for char in forms]
        return StubEncoding([[1, *ids, 2]], [[None, *([0] * len(ids)), None]])


class StubEncoding(dict):
    def __init__(self, input_ids, word_ids):
        super().__init__(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(
                [[1] * len(ids) for ids in input_ids],
            ),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


def make_mwt_sentence():
    return Sentence(
        comments=["# text = im gehe"],
        tokens=[
            Token(id="1-2", form="im"),
            Token(id=1, form="In"),
            Token(id=2, form="Dem"),
            Token(id=3, form="gehe"),
        ],
    )


def test_mwt_dataset_builds_expansion_and_identity_targets():
    targets = MWTDataset._expansion_targets(make_mwt_sentence())
    assert targets == {"1-2": "In Dem"}


def test_mwt_dataset_emits_source_character_ids():
    dataset = MWTDataset.__new__(MWTDataset)
    dataset.sentences = [make_mwt_sentence()]
    dataset.max_length = 32
    dataset.hf_tok = StubHFTokenizer()
    specials = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    dataset.input_vocab = {**specials, "g": 4, "e": 5, "h": 6, "i": 7, "m": 8, "t": 9}
    dataset.output_vocab = {**specials, **{char: index + 4 for index, char in enumerate(" DemInght")}}

    first = dataset[0]
    assert first["input_ids"].tolist() == [1, 4, 5, 2]
    assert len(first["input_char_ids"]) == 2
    assert len(first["output_char_ids"]) == 2
    batch = dataset.collate_fn([first])
    assert batch["input_char_ids"].shape[:2] == (1, 2)
    assert batch["output_char_ids"].shape[:2] == (1, 2)
    assert "upos_ids" not in batch


def test_mwt_forward_encodes_sentences_once_and_aligns_token_slots():
    model = MWTModel.__new__(MWTModel)
    nn.Module.__init__(model)
    model.output_vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "a": 4}
    model.input_vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "a": 4}
    model.encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
    model._build_heads()
    model.encode = Mock(return_value=torch.randn(2, 4, 8))

    outputs = model(
        torch.tensor([[1, 4, 4, 2], [1, 4, 2, 0]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        torch.tensor([
            [[2, 4, 3], [2, 4, 3]],
            [[2, 4, 3], [2, 3, 0]],
        ]),
        torch.tensor([
            [[2, 4, 3], [2, 4, 3]],
            [[2, 4, 3], [2, 3, -100]],
        ]),
    )

    assert model.encode.call_count == 1
    assert outputs["logits"].shape[0] == 4
    assert outputs["logits"].shape[-1] == len(model.output_vocab)


def test_mwt_prediction_inserts_range_and_component_rows():
    model = MWTModel.__new__(MWTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(resolve_device=lambda: "cpu")
    model._predict_expansions = Mock(return_value=[["In Dem", "gehe"]])
    sentence = Sentence(
        comments=["# text = im gehe"],
        tokens=[Token(id=1, form="im"), Token(id=2, form="gehe", space_after=False)],
    )

    result = model.predict([sentence], show_progress=False)[0]

    assert [(token.id, token.form) for token in result.tokens] == [
        ("1-2", "im"),
        (1, "In"),
        (2, "Dem"),
        (3, "gehe"),
    ]
    assert result.tokens[2].space_after is True
    assert result.tokens[3].space_after is False
    assert result.comments == sentence.comments


def test_mwt_prediction_preserves_already_expanded_sentence():
    model = MWTModel.__new__(MWTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(resolve_device=lambda: "cpu")
    model._predict_expansions = Mock(return_value=[])
    sentence = make_mwt_sentence()

    result = model.predict([sentence], show_progress=False)

    assert result[0] is sentence
    model._predict_expansions.assert_called_once_with(
        [sentence],
        show_progress=False,
    )


def test_pipeline_expands_before_tagging():
    pipeline = BorgPipeline()
    pipeline.mwt_model = Mock()
    pipeline.tagger_model = Mock()
    pipeline.parser_model = None
    pipeline.lemmatizer_model = None
    expanded = [Sentence(tokens=[Token(id="1-2", form="im"), Token(id=1, form="In"), Token(id=2, form="Dem")])]
    pipeline.mwt_model.predict.return_value = expanded
    pipeline.tagger_model.predict.return_value = expanded

    pipeline.process("im")

    pipeline.tagger_model.predict.assert_called_once_with(expanded)
