import torch
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

if __name__ == "__main__":
    test_tokenizer_whitespace()
