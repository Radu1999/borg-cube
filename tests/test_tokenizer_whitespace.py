import torch
from types import SimpleNamespace
from src.config import BorgConfig
from src.models.tokenizer import TokenizerModel
from src.data.conllu import Token, Sentence

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
        return res

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
        max_seq_length=10, resolve_device=lambda: "cpu"
    )
    model.encoder = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=6)
    )
    model.hf_tokenizer = FakeTokenizer()
    observed_windows = []

    def mock_forward(input_ids, attention_mask):
        observed_windows.append(input_ids[0, 1:-1].tolist())
        logits = torch.zeros((1, input_ids.size(1), TokenizerModel.NUM_LABELS))
        for i, token_id in enumerate(input_ids[0].tolist()):
            logits[0, i, 2 if token_id == ord("a") else 0] = 1
        return logits

    model.forward = mock_forward
    text = "abcdefghijklmn"
    sentences = model.predict(text)
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


if __name__ == "__main__":
    test_tokenizer_whitespace()
