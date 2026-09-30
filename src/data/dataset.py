"""PyTorch Dataset classes for each pipeline component."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.conllu import Sentence, Token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vocab(items: List[str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for item in sorted(set(items)):
        if item not in vocab:
            vocab[item] = len(vocab)
    return vocab


def _feats_to_str(token: Token) -> str:
    return token.feats_str() if hasattr(token, "feats_str") else str(token.feats)


# ---------------------------------------------------------------------------
# Tokenizer dataset
# ---------------------------------------------------------------------------

class TokenizerDataset(Dataset):
    """Character-level dataset for sentence/token boundary detection.

    Labels per character:
        0 = continuation (inside a token)
        1 = token-start (first char of a token, not sentence-start)
        2 = sentence-start (first char of first token in a sentence)
    """

    CONTINUATION = 0
    TOKEN_START = 1
    SENTENCE_START = 2

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
    ):
        self.max_length = max_length
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)

        # Build (text_chunk, label_list) pairs from the raw text reconstructed
        # out of CoNLL-U sentences.
        self.examples: List[Tuple[str, List[int]]] = []
        self._build_examples(sentences)

    def _build_examples(self, sentences: List[Sentence]) -> None:
        # We group sentences into chunks that fit within max_length.
        chars: List[str] = []
        labels: List[int] = []

        def flush():
            if chars:
                self.examples.append(("".join(chars), list(labels)))
                chars.clear()
                labels.clear()

        for sent in sentences:
            tokens = sent.regular_tokens()
            if not tokens:
                continue
            sentence_text = ""
            sentence_labels: List[int] = []
            for tok_idx, tok in enumerate(tokens):
                form = tok.form
                for char_idx, ch in enumerate(form):
                    if tok_idx == 0 and char_idx == 0:
                        lbl = self.SENTENCE_START
                    elif char_idx == 0:
                        lbl = self.TOKEN_START
                    else:
                        lbl = self.CONTINUATION
                    sentence_text += ch
                    sentence_labels.append(lbl)

                # Handle SpaceAfter: add actual whitespace from the Token if available
                if tok_idx < len(tokens) - 1:
                    if tok.space_after:
                        sentence_text += " "
                        sentence_labels.append(self.CONTINUATION)
                    else:
                        # No space between tokens (e.g. "don't" -> "do", "'t")
                        pass
                elif tok.space_after:
                    # Trailing space at the end of the sentence
                    sentence_text += " "
                    sentence_labels.append(self.CONTINUATION)

            # Check if adding this sentence exceeds limit; flush first if so.
            if len(chars) + len(sentence_text) > self.max_length and chars:
                flush()
            chars.extend(sentence_text)
            labels.extend(sentence_labels)

        flush()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, char_labels = self.examples[idx]
        encoding = self.hf_tok(
            text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset_mapping = encoding.pop("offset_mapping").squeeze(0)  # (seq_len, 2)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))
        if isinstance(token_type_ids, torch.Tensor) and token_type_ids.dim() > 1:
            token_type_ids = token_type_ids.squeeze(0)

        seq_len = input_ids.size(0)
        aligned_labels = torch.zeros(seq_len, dtype=torch.long)

        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:
                aligned_labels[i] = -100  # special token — ignored
            else:
                # Use the label of the first character of this sub-word
                char_idx = start
                if char_idx < len(char_labels):
                    aligned_labels[i] = char_labels[char_idx]
                else:
                    aligned_labels[i] = self.CONTINUATION

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
        }


# ---------------------------------------------------------------------------
# Tagger dataset
# ---------------------------------------------------------------------------

class TaggerDataset(Dataset):
    """Token-level dataset for UPOS/XPOS/FEATS tagging."""

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
        upos_vocab: Optional[Dict[str, int]] = None,
        xpos_vocab: Optional[Dict[str, int]] = None,
        feats_vocab: Optional[Dict[str, int]] = None,
    ):
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.max_length = max_length

        all_upos = [t.upos for s in sentences for t in s.regular_tokens()]
        all_xpos = [t.xpos for s in sentences for t in s.regular_tokens()]
        all_feats = [_feats_to_str(t) for s in sentences for t in s.regular_tokens()]

        self.upos_vocab = upos_vocab or _build_vocab(all_upos)
        self.xpos_vocab = xpos_vocab or _build_vocab(all_xpos)
        self.feats_vocab = feats_vocab or _build_vocab(all_feats)

        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]

        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        seq_len = input_ids.size(0)
        upos_labels = torch.full((seq_len,), -100, dtype=torch.long)
        xpos_labels = torch.full((seq_len,), -100, dtype=torch.long)
        feats_labels = torch.full((seq_len,), -100, dtype=torch.long)

        seen_words = set()
        for i, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            if word_id < len(tokens):
                tok = tokens[word_id]
                upos_labels[i] = self.upos_vocab.get(tok.upos, self.upos_vocab["<UNK>"])
                xpos_labels[i] = self.xpos_vocab.get(tok.xpos, self.xpos_vocab["<UNK>"])
                feats_labels[i] = self.feats_vocab.get(
                    _feats_to_str(tok), self.feats_vocab["<UNK>"]
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "upos_labels": upos_labels,
            "xpos_labels": xpos_labels,
            "feats_labels": feats_labels,
        }


# ---------------------------------------------------------------------------
# Parser dataset
# ---------------------------------------------------------------------------

class ParserDataset(Dataset):
    """Token-level dataset for dependency parsing (HEAD + DEPREL)."""

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
        deprel_vocab: Optional[Dict[str, int]] = None,
    ):
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.max_length = max_length

        all_deprels = [t.deprel for s in sentences for t in s.regular_tokens()]
        self.deprel_vocab = deprel_vocab or _build_vocab(all_deprels)
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]

        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        seq_len = input_ids.size(0)
        head_labels = torch.full((seq_len,), -100, dtype=torch.long)
        deprel_labels = torch.full((seq_len,), -100, dtype=torch.long)

        seen_words = set()
        for i, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            if word_id < len(tokens):
                tok = tokens[word_id]
                head_labels[i] = tok.head if tok.head is not None else 0
                deprel_labels[i] = self.deprel_vocab.get(
                    tok.deprel, self.deprel_vocab["<UNK>"]
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_labels": head_labels,
            "deprel_labels": deprel_labels,
        }


# ---------------------------------------------------------------------------
# Lemmatizer dataset
# ---------------------------------------------------------------------------

def _compute_edit_script(form: str, lemma: str) -> str:
    """Encode lemma as an edit script relative to the form.

    Strategy: find the longest common prefix and suffix, then record
    how many characters to strip from the end and what to append.

    Returns a string like "keep3:strip2:add+en" which is compact enough
    to classify.
    """
    form_l = form.lower()
    lemma_l = lemma.lower()

    # Longest common prefix
    prefix_len = 0
    for a, b in zip(form_l, lemma_l):
        if a == b:
            prefix_len += 1
        else:
            break

    # Characters remaining after the prefix
    form_tail = form_l[prefix_len:]
    lemma_tail = lemma_l[prefix_len:]

    # Strip suffix chars from form_tail and append lemma_tail
    strip = len(form_tail)
    add = lemma_tail

    return f"k{prefix_len}:s{strip}:a{add}"


class LemmatizerDataset(Dataset):
    """Token-level dataset for lemmatization via edit scripts."""

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
        upos_vocab: Optional[Dict[str, int]] = None,
        script_vocab: Optional[Dict[str, int]] = None,
    ):
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.max_length = max_length

        all_upos = [t.upos for s in sentences for t in s.regular_tokens()]
        self.upos_vocab = upos_vocab or _build_vocab(all_upos)

        all_scripts = [
            _compute_edit_script(t.form, t.lemma)
            for s in sentences
            for t in s.regular_tokens()
        ]
        self.script_vocab = script_vocab or _build_vocab(all_scripts)
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]

        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        seq_len = input_ids.size(0)
        upos_ids = torch.zeros(seq_len, dtype=torch.long)
        script_labels = torch.full((seq_len,), -100, dtype=torch.long)

        seen_words = set()
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            tok = tokens[word_id] if word_id < len(tokens) else None
            if tok is not None:
                upos_ids[i] = self.upos_vocab.get(tok.upos, self.upos_vocab["<UNK>"])
            if word_id not in seen_words and tok is not None:
                seen_words.add(word_id)
                script = _compute_edit_script(tok.form, tok.lemma)
                script_labels[i] = self.script_vocab.get(
                    script, self.script_vocab["<UNK>"]
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "upos_ids": upos_ids,
            "script_labels": script_labels,
        }
