"""Offline prediction regressions and measurable batching checks."""
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.models.inference import sentence_batches
from src.models.lemmatizer import LemmatizerModel, _apply_edit_script
from src.models.parser import ParserModel, greedy_decode
from src.models.tagger import TaggerModel
from src.models.tokenizer import TokenizerModel


class Encoding(dict):
    def __init__(self, input_ids, word_ids):
        super().__init__(
            input_ids=input_ids,
            attention_mask=[[1] * len(ids) for ids in input_ids],
        )
        self.ids = word_ids

    def word_ids(self, batch_index=0):
        return self.ids[batch_index]


class WordTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, forms, max_length, **kwargs):
        sequences, mappings = [], []
        for words in forms:
            ids, word_ids = [], []
            for wid, word in enumerate(words):
                token_id = sum(map(ord, word)) % 30 + 3
                pieces = [token_id, token_id + 1] if len(word) > 2 else [token_id]
                ids.extend(pieces)
                word_ids.extend([wid] * len(pieces))
            sequences.append([1, *ids[:max_length - 2], 2])
            mappings.append([None, *word_ids[:max_length - 2], None])
        return Encoding(sequences, mappings)


def make_model(model_type, batch_size):
    torch.manual_seed(42)
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.config = BorgConfig(device="cpu", max_seq_length=10, eval_batch_size=batch_size)
    model.hf_tokenizer = Mock(wraps=WordTokenizer())
    model.hf_tokenizer.pad_token_id = 0
    model.hidden_size = 4
    if model_type is TaggerModel:
        model.upos_vocab = {"<PAD>": 0, "<UNK>": 1, "NOUN": 2}
        model.xpos_vocab = {"<PAD>": 0, "<UNK>": 1, "NN": 2}
        model.feats_vocab = {"<PAD>": 0, "<UNK>": 1, "Number=Sing": 2}
    elif model_type is LemmatizerModel:
        model.upos_vocab = {"<PAD>": 0, "<UNK>": 1, "NOUN": 2}
        model.script_vocab = {"<PAD>": 0, "<UNK>": 1, "k1:s1:az": 2}
    else:
        model.deprel_vocab = {"<PAD>": 0, "<UNK>": 1, "root": 2, "dep": 3}
        model.ARC_HIDDEN = 8
        model.REL_HIDDEN = 4
    model._build_heads()

    def encode(input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        ids = input_ids.float()
        return torch.stack([ids / 30, ids.sin(), ids.cos(), ids / 60], dim=-1)

    model.encode = Mock(side_effect=encode)
    return model


def sentences():
    return [
        Sentence(
            comments=["# long"],
            tokens=[
                Token(id=1, form="Long", upos="NOUN", space_after=False, misc="X=Y"),
                Token(id="1-2", form="Longword"),
                Token(id=2, form="word", upos="UNKNOWN"),
                Token(id="2.1", form="empty"),
                *[Token(id=i, form="more", upos="NOUN") for i in range(3, 8)],
            ],
        ),
        Sentence(comments=["# empty"]),
        Sentence(tokens=[Token(id=1, form="A", upos="NOUN")]),
        Sentence(tokens=[Token(id=1, form="B"), Token(id=2, form="C")]),
        Sentence(tokens=[Token(id="1.1", form="node")]),
        Sentence(tokens=[Token(id=1, form="Last", upos="NOUN")]),
    ]


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_batched_predictions_match_single_and_preserve_metadata(model_type):
    inputs = sentences()
    original = deepcopy(inputs)
    single = make_model(model_type, 1)
    batched = make_model(model_type, 3)
    expected = single.predict(inputs)
    actual = batched.predict(inputs)
    assert actual == expected
    assert inputs == original
    assert batched.encode.call_count == 2
    assert single.encode.call_count == 4
    assert batched.hf_tokenizer.call_count == 2
    assert actual[1] is inputs[1]
    assert actual[4] is inputs[4]
    assert actual[0].tokens[0].space_after is False
    assert actual[0].tokens[1] is inputs[0].tokens[1]
    assert actual[0].tokens[3] is inputs[0].tokens[3]
    truncated = actual[0].tokens[-1]
    if model_type is TaggerModel:
        assert (truncated.upos, truncated.xpos, truncated.feats) == ("_", "_", "_")
    elif model_type is LemmatizerModel:
        assert truncated.lemma == truncated.form.lower()
    else:
        assert (truncated.head, truncated.deprel) == (0, "_")


def test_parser_selected_relation_scores_match_dense_forward():
    model = make_model(ParserModel, 3)
    inputs = sentences()
    expected = {}
    with torch.inference_mode():
        for batch in sentence_batches(model, inputs):
            arcs, relations = model(batch.input_ids, batch.attention_mask)
            for row, index in enumerate(batch.indices):
                positions = batch.word_positions[row]
                heads = greedy_decode(arcs[row], positions)
                inverse = {pos: wid for wid, pos in positions.items()}
                expected[index] = {
                    wid: (
                        0 if head == 0 else inverse[head] + 1,
                        int(relations[row, positions[wid], head].argmax(-1)),
                    )
                    for wid, head in heads.items()
                }
    shapes = []
    hook = model.rel_biaffine.register_forward_pre_hook(
        lambda _, args: shapes.append(args[0].shape)
    )
    actual = model.predict(inputs)
    hook.remove()
    inverse_relations = {value: key for key, value in model.deprel_vocab.items()}
    for index, predictions in expected.items():
        for wid, (head, relation) in predictions.items():
            token = actual[index].regular_tokens()[wid]
            assert (token.head, token.deprel) == (head, inverse_relations[relation])
    assert all(shape[1] == 1 for shape in shapes)


def test_tagger_uses_first_subword_predictions():
    model = make_model(TaggerModel, 3)
    inputs = sentences()

    def forward(input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        logits = torch.zeros((*input_ids.shape, 3))
        ids = input_ids.remainder(3)
        logits.scatter_(-1, ids.unsqueeze(-1), 10)
        return logits, logits, logits

    model.forward = Mock(side_effect=forward)
    actual = model.predict(inputs)
    inverse = {value: key for key, value in model.upos_vocab.items()}
    for batch in sentence_batches(model, inputs):
        for row, index in enumerate(batch.indices):
            for wid, pos in batch.word_positions[row].items():
                expected = inverse[int(batch.input_ids[row, pos] % 3)]
                assert actual[index].regular_tokens()[wid].upos == expected
    assert model.forward.call_count == 2


def test_lemmatizer_aligns_upos_to_every_subword_and_uses_first_piece():
    model = make_model(LemmatizerModel, 3)
    inputs = sentences()

    def forward(input_ids, attention_mask, upos_ids):
        assert torch.is_inference_mode_enabled()
        assert torch.all(upos_ids[input_ids <= 2] == 0)
        known_upos = {}
        for sentence in inputs:
            for token in sentence.regular_tokens():
                token_id = sum(map(ord, token.form)) % 30 + 3
                pieces = [token_id, token_id + 1] if len(token.form) > 2 else [token_id]
                for piece in pieces:
                    known_upos[piece] = model.upos_vocab.get(token.upos, 1)
        for row in range(input_ids.size(0)):
            for pos in range(input_ids.size(1)):
                piece = int(input_ids[row, pos])
                assert int(upos_ids[row, pos]) == known_upos.get(piece, 0)
        logits = torch.zeros((*input_ids.shape, 3))
        logits.scatter_(-1, upos_ids.unsqueeze(-1), 10)
        return logits

    model.forward = Mock(side_effect=forward)
    actual = model.predict(inputs)
    assert actual[0].tokens[0].lemma == _apply_edit_script("Long", "k1:s1:az")
    assert actual[0].tokens[2].lemma == "word"
    assert actual[2].tokens[0].lemma == "az"
    assert model.forward.call_count == 2


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_empty_input_and_invalid_batch_size(model_type):
    model = make_model(model_type, 3)
    assert model.predict([]) == []
    assert model.encode.call_count == 0
    model.config.eval_batch_size = 0
    with pytest.raises(ValueError, match="eval_batch_size"):
        model.predict(sentences())


class WindowTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, token_ids, **kwargs):
        ids = [1, *token_ids, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def get_special_tokens_mask(self, token_ids, **kwargs):
        return [1, *[0 for _ in token_ids], 1]


def make_tokenizer(batch_size):
    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = BorgConfig(device="cpu", max_seq_length=6, eval_batch_size=batch_size)
    model.encoder = torch.nn.Module()
    model.encoder.config = model.config
    model.encoder.config.max_position_embeddings = 6
    model.hf_tokenizer = WindowTokenizer()

    def forward(ids, mask):
        assert torch.is_inference_mode_enabled()
        logits = torch.zeros((*ids.shape, 3))
        # Depend on window position to exercise overlap score accumulation.
        logits[..., 0] = 1
        logits[:, 1, 2] = 2
        logits[..., 1] = (ids % 3 == 0).float() * 3
        return logits

    model.forward = Mock(side_effect=forward)
    return model


def test_tokenizer_batches_windows_and_preserves_overlap_predictions():
    single, batched = make_tokenizer(1), make_tokenizer(3)
    text = "A B C D E F G H I J"
    assert batched.predict(text) == single.predict(text)
    assert single.forward.call_count == 7
    assert batched.forward.call_count == 3
    assert batched.forward.call_args_list[0].args[0].shape == (3, 6)
    assert batched.predict("") == []
    batched.config.eval_batch_size = 0
    with pytest.raises(ValueError, match="eval_batch_size"):
        batched.predict(text)
