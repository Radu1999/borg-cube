import torch
from types import SimpleNamespace
from unittest.mock import Mock
from src.config import BorgConfig
from src.models.tokenizer import TokenizerModel, _mwt_surface_metrics
from src.data.conllu import Token, Sentence
from src.models.evaluation import evaluate_predictions, tokenizer_validation_text

def test_tokenizer_whitespace():
    # Mock config
    config = BorgConfig()
    config.model_name = "bert-base-multilingual-cased"

    model = TokenizerModel(config)

    # Use a text with known subwords
    text = "A B. C"

    def mock_forward(self, ids, mask):
        L = ids.size(1)
        res = torch.zeros((1, L, 3))
        # Offsets for "A B. C" with bert-base-multilingual-cased:
        # [CLS] A B . C [SEP]
        # Indices: 0 1 2 3 4 5
        # Labels: X 2 1 0 2 X
        if L > 1: res[0, 1, 2] = 1 # A: SENTENCE_START
        if L > 2: res[0, 2, 1] = 1 # B: TOKEN_START
        if L > 3: res[0, 3, 0] = 1 # .: CONT
        if L > 4: res[0, 4, 2] = 1 # C: SENTENCE_START
        return res, torch.zeros((*res.shape[:2], 2))

    model.forward = mock_forward.__get__(model, TokenizerModel)

    sents = model.predict(text)

    for s in sents:
        for t in s.tokens:
            print(f"Token: {t.form}, space_after: {t.space_after}")

    # Verify
    # sents[0] tokens:
    # 0: "A" (S_START) -> followed by " " -> space_after=True
    # 1: "B." (T_START + CONT) -> followed by " " -> space_after=True
    # sents[1] tokens:
    # 0: "C" (S_START) -> followed by "" -> space_after=False
    assert sents[0].tokens[0].form == "A" and sents[0].tokens[0].space_after == True
    assert sents[0].tokens[1].form == "B." and sents[0].tokens[1].space_after == True
    assert sents[1].tokens[0].form == "C" and sents[1].tokens[0].space_after == False
    print("Whitespace test passed!")


def test_tokenizer_uses_overlapping_windows():
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            assert "return_tensors" not in kwargs
            return {
                "input_ids": [ord(char) for char in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, token_ids, add_special_tokens, return_attention_mask):
            input_ids = [101, *token_ids, 102]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens):
            return [1, *([0] * len(token_ids)), 1]

    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        max_seq_length=10, resolve_device=lambda: "cpu",
        dtype=torch.bfloat16, eval_batch_size=2,
    )
    model.to = Mock(wraps=model.to)
    model.encoder = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=6)
    )
    model.hf_tokenizer = FakeTokenizer()
    observed_windows = []

    def mock_forward(input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, TokenizerModel.NUM_LABELS))
        for row in range(input_ids.size(0)):
            length = int(attention_mask[row].sum())
            observed_windows.append(input_ids[row, 1:length - 1].tolist())
            for i, token_id in enumerate(input_ids[row].tolist()):
                logits[row, i, 2 if token_id == ord("a") else 0] = 1
        return logits, torch.zeros((*logits.shape[:2], 2))

    model.forward = mock_forward
    text = "abcdefghijklmn"
    sentences = model.predict(text)
    model.to.assert_called_once_with("cpu")
    observed_ids = [token_id for window in observed_windows for token_id in window]

    assert len(observed_ids) > len(text)
    assert set(observed_ids) == {ord(char) for char in text}
    assert len(observed_windows) > 1
    assert any(
        set(left) & set(right)
        for left, right in zip(observed_windows, observed_windows[1:])
    )
    assert all(
        len(window) + 2 <= model.encoder.config.max_position_embeddings
        for window in observed_windows
    )
    assert len(sentences) == 1
    assert sentences[0].tokens[0].form == text


def test_tokenizer_handles_subword_offsets_that_include_leading_whitespace():
    # Some fast tokenizers (e.g. the SentencePiece-based DeBERTa-v3
    # tokenizer) report an offset span for a sub-word that includes the
    # whitespace preceding it, e.g. offset (1, 3) for " B" instead of (2, 3)
    # for "B". `predict` must strip that leading whitespace out of the
    # reconstructed token form and still detect the space correctly.
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [1000, 1001],
                "offset_mapping": [(0, 1), (1, 3)],
            }

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, token_ids, add_special_tokens, return_attention_mask):
            input_ids = [101, *token_ids, 102]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens):
            return [1, *([0] * len(token_ids)), 1]

    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        max_seq_length=10, resolve_device=lambda: "cpu",
        dtype=torch.bfloat16, eval_batch_size=2,
    )
    model.encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=10))
    model.hf_tokenizer = FakeTokenizer()

    def mock_forward(input_ids, attention_mask):
        logits = torch.zeros((1, input_ids.size(1), TokenizerModel.NUM_LABELS))
        for i, token_id in enumerate(input_ids[0].tolist()):
            if token_id == 1000:
                logits[0, i, 2] = 1  # SENTENCE_START
            elif token_id == 1001:
                logits[0, i, 1] = 1  # TOKEN_START
            else:
                logits[0, i, 0] = 1  # CONTINUATION
        return logits, torch.zeros((*logits.shape[:2], 2))

    model.forward = mock_forward
    sents = model.predict("A B")

    assert len(sents) == 1
    tokens = sents[0].tokens
    assert tokens[0].form == "A" and tokens[0].space_after is True
    assert tokens[1].form == "B" and tokens[1].space_after is False


def test_tokenizer_skips_whitespace_only_offsets():
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            input_ids = [1000, 1001, 1002] if text == "A B" else [1000, 1001]
            return {
                "input_ids": input_ids,
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, token_ids, add_special_tokens, return_attention_mask):
            input_ids = [101, *token_ids, 102]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens):
            return [1, *([0] * len(token_ids)), 1]

    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        max_seq_length=10, resolve_device=lambda: "cpu",
        dtype=torch.bfloat16, eval_batch_size=2,
    )
    model.encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=10))
    model.hf_tokenizer = FakeTokenizer()

    def mock_forward(input_ids, attention_mask):
        logits = torch.zeros((1, input_ids.size(1), TokenizerModel.NUM_LABELS))
        labels = {1000: 2, 1001: 1, 1002: 1}
        for i, token_id in enumerate(input_ids[0].tolist()):
            logits[0, i, labels.get(token_id, 0)] = 1
        return logits, torch.zeros((*logits.shape[:2], 2))

    model.forward = mock_forward
    sentences = model.predict("A B")

    assert len(sentences) == 1
    assert [token.form for token in sentences[0].tokens] == ["A", "B"]
    assert sentences[0].tokens[0].space_after is True

    trailing_space = model.predict("A ")
    assert [token.form for token in trailing_space[0].tokens] == ["A"]
    assert trailing_space[0].tokens[0].space_after is True


def _predict_with_offsets(text, offsets, labels, mwt_labels=None):
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {
                "input_ids": list(range(1000, 1000 + len(offsets))),
                "offset_mapping": offsets,
            }

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, token_ids, add_special_tokens, return_attention_mask):
            input_ids = [101, *token_ids, 102]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens):
            return [1, *([0] * len(token_ids)), 1]

    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        max_seq_length=64, resolve_device=lambda: "cpu",
        dtype=torch.bfloat16, eval_batch_size=2,
    )
    model.encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))
    model.hf_tokenizer = FakeTokenizer()

    def mock_forward(input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, TokenizerModel.NUM_LABELS))
        mwt_logits = torch.zeros((*input_ids.shape, 2))
        for row, ids in enumerate(input_ids.tolist()):
            for position, token_id in enumerate(ids):
                if token_id >= 1000:
                    index = token_id - 1000
                    logits[row, position, labels[index]] = 10
                    mwt_logits[row, position, mwt_labels[index] if mwt_labels else 0] = 10
        return logits, mwt_logits

    model.forward = mock_forward
    return model.predict(text, show_progress=False)


def test_tokenizer_predicts_mwt_flag_on_surface_token():
    predicted = _predict_with_offsets(
        "du pain", [(0, 2), (3, 7)], [2, 1], [1, 0],
    )

    assert [(token.form, token.is_mwt) for token in predicted[0].surface_tokens()] == [
        ("du", True),
        ("pain", False),
    ]


def test_mwt_metrics_compare_surface_spans_not_expanded_words():
    gold = [Sentence(tokens=[
        Token(id="1-2", form="du"),
        Token(id=1, form="de"),
        Token(id=2, form="le"),
        Token(id=3, form="pain"),
    ])]
    predicted = _predict_with_offsets(
        "du pain", [(0, 2), (3, 7)], [2, 1], [1, 0],
    )

    metrics = _mwt_surface_metrics(gold, predicted, "du pain")
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert (metrics.gold_count, metrics.system_count) == (1, 1)


def test_tokenizer_decodes_duplicate_boundaries_once():
    sentences = _predict_with_offsets(
        "\u4f60\u597d", [(0, 1), (0, 1), (1, 2)], [2, 2, 0],
    )

    assert len(sentences) == 1
    assert [(token.form, token.space_after) for token in sentences[0].tokens] == [
        ("\u4f60\u597d", False),
    ]


def test_tokenizer_preserves_characters_outside_subword_offsets():
    cases = [
        ("a\u0301", [(0, 1)], [2], [("a\u0301", False)]),
        (
            "\u200bA\u0301 B\u0301", [(1, 2), (3, 5)], [2, 1],
            [("\u200bA\u0301", True), ("B\u0301", False)],
        ),
        ("\u200b", [], [], [("\u200b", False)]),
    ]
    for text, offsets, labels, expected in cases:
        sentences = _predict_with_offsets(text, offsets, labels)
        assert len(sentences) == 1
        assert [(token.form, token.space_after) for token in sentences[0].tokens] == expected


def test_tokenizer_validation_preserves_surface_text():
    gold = [Sentence(tokens=[
        Token(id=1, form="\u4f60\u597d"),
        Token(id=2, form="a\u0301", space_after=False),
    ])]
    predicted = _predict_with_offsets(
        tokenizer_validation_text(gold),
        [(0, 1), (0, 1), (1, 2), (2, 4)], [2, 2, 0, 1],
    )

    metrics = evaluate_predictions(gold, predicted)

    assert metrics["Tokens"].f1 == 1.0
    assert metrics["Sentences"].f1 == 1.0


if __name__ == "__main__":
    test_tokenizer_whitespace()
