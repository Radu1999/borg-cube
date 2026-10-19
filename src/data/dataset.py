"""PyTorch Dataset classes for each pipeline component."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from src.data.conllu import Sentence, Token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vocab(items: List[str], initial_vocab: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    vocab = initial_vocab.copy() if initial_vocab is not None else {"<PAD>": 0, "<UNK>": 1}
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
    """Character-level dataset for sentence/token boundary detection."""

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
        self.examples: List[Tuple[str, List[int]]] = []
        self._build_examples(sentences)

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        padding_values = {
            "input_ids": self.hf_tok.pad_token_id,
            "attention_mask": 0,
            "labels": -100,
        }
        return {
            key: pad_sequence(
                [example[key] for example in batch],
                batch_first=True,
                padding_value=padding_value,
            )
            for key, padding_value in padding_values.items()
        }

    def _build_examples(self, sentences: List[Sentence]) -> None:
        chars: List[str] = []
        labels: List[int] = []

        def flush():
            if chars:
                self.examples.append(("".join(chars), list(labels)))
                chars.clear()
                labels.clear()

        for sent in sentences:
            tokens = sent.surface_tokens()
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
            padding=False,
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
                # Use the label of the first character of this sub-word.
                # Some tokenizers (e.g. SentencePiece-based ones used by
                # DeBERTa-v3) include the leading whitespace in the offset
                # span of a sub-word that starts a new word. Skip past any
                # such leading whitespace so we don't pick up the space
                # character's CONTINUATION label instead of the real
                # TOKEN_START/SENTENCE_START label of the word.
                char_idx = start
                while char_idx < end and char_idx < len(text) and text[char_idx].isspace():
                    char_idx += 1
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
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Right-pad tagger examples to the longest sequence in this batch."""
        padding_values = {
            "input_ids": self.hf_tok.pad_token_id,
            "attention_mask": 0,
            "upos_labels": -100,
            "xpos_labels": -100,
            "feats_labels": -100,
        }
        return {
            key: pad_sequence(
                [example[key] for example in batch],
                batch_first=True,
                padding_value=padding_value,
            )
            for key, padding_value in padding_values.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]

        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
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
        self.hf_tok.padding_side = "right"

    def __len__(self) -> int:
        return len(self.sentences)

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Right-pad parser examples to the longest sequence in this batch."""
        padding_values = {
            "input_ids": self.hf_tok.pad_token_id,
            "attention_mask": 0,
            "head_labels": -100,
            "deprel_labels": -100,
        }
        return {
            key: pad_sequence(
                [example[key] for example in batch],
                batch_first=True,
                padding_value=padding_value,
            )
            for key, padding_value in padding_values.items()
        }


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]

        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        seq_len = input_ids.size(0)
        head_labels = torch.full((seq_len,), -100, dtype=torch.long)
        deprel_labels = torch.full((seq_len,), -100, dtype=torch.long)

        # Map word-id -> first sub-word position. The arc classifier scores
        # pairs of *sequence positions* (not word indices), and position 0
        # (the CLS token) stands in for the virtual ROOT node. Gold
        # ``tok.head`` values are 1-based CoNLL-U word indices (0 == ROOT),
        # so they must be translated into sequence positions before being
        # used as classification targets — otherwise the labels are
        # meaningless whenever any word is split into multiple sub-words.
        word_positions: Dict[int, int] = {}
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in word_positions:
                word_positions[word_id] = i

        seen_words = set()
        for i, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            if word_id < len(tokens):
                tok = tokens[word_id]
                head_pos: Optional[int] = None
                if tok.head is None:
                    # Unknown head (e.g. malformed CoNLL-U data) -> leave
                    # the label ignored (-100) rather than silently
                    # injecting a ROOT target, same treatment as the
                    # truncated-head case below.
                    pass
                elif tok.head == 0:
                    head_pos = 0  # ROOT -> CLS position
                else:
                    # `tok.head` is the gold token's 1-based CoNLL-U index
                    # into `tokens` (`regular_tokens()`), and `word_id`
                    # (from `encoding.word_ids()`, single-example call) is
                    # the matching 0-based index into that same
                    # `forms`/`tokens` list passed to
                    # `is_split_into_words=True` — so `tok.head - 1` is
                    # exactly the `word_id` of the head token. The head
                    # word may have been truncated out of the sequence
                    # (max_length); in that case there's no valid target
                    # position, so leave the label ignored (-100) instead
                    # of silently mislabeling it as ROOT.
                    head_pos = word_positions.get(tok.head - 1)
                if head_pos is not None:
                    head_labels[i] = head_pos
                    deprel_labels[i] = self.deprel_vocab.get(
                        tok.deprel, self.deprel_vocab["<UNK>"]
                    )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_labels": head_labels,
            "deprel_labels": deprel_labels,
        }


class LemmatizerDataset(Dataset):
    """Token-level dataset for lemmatization via edit scripts."""

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
        upos_vocab: Optional[Dict[str, int]] = None,
        input_vocab: Optional[Dict[str, int]] = None,
        output_vocab: Optional[Dict[str, int]] = None
    ):
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.max_length = max_length

        all_upos = [t.upos for s in sentences for t in s.regular_tokens()]
        self.upos_vocab = upos_vocab or _build_vocab(all_upos)

        all_input_chars = [c for s in sentences for t in s.regular_tokens() for c in t.form]
        all_output_chars = [c for s in sentences for t in s.regular_tokens() for c in t.lemma]

        self.input_vocab = input_vocab or _build_vocab(all_input_chars, initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3})
        self.output_vocab = output_vocab or _build_vocab(all_output_chars, initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}) 
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Right-pad lemmatizer examples to the longest sequence in this batch."""
        padding_values = {
            "input_ids": self.hf_tok.pad_token_id,
            "attention_mask": 0,
            "upos_ids": 0,
            "input_char_ids": torch.tensor([[0]]),
            "output_char_ids": torch.tensor([[-100]]),
        }
        output={}
        for key, padding_value in padding_values.items():
            if key=="input_char_ids" or key=="output_char_ids":
                max_len = max(len(example[key][0]) for example in batch)
                padded_sequences = []
                for example in batch:
                    seq = torch.cat([e.unsqueeze(0) for e in example[key]])
                    pad_length = max_len - seq.shape[1]
                    padded_seq = torch.cat([seq, torch.full((seq.shape[0], pad_length), padding_value[0][0], dtype=torch.long)], dim=1)
                    padded_sequences.append(padded_seq)
                
                max_sentence_len=max(seq.shape[0] for seq in padded_sequences)
                repadded_seqs=[]
                for seq in padded_sequences:
                    if seq.shape[0]<max_sentence_len:
                        pad_length=max_sentence_len-seq.shape[0]
                        padded_seq=torch.cat([seq, torch.full((pad_length, seq.shape[1]), padding_value[0][0], dtype=torch.long)], dim=0)
                        repadded_seqs.append(padded_seq)
                    else:
                        repadded_seqs.append(seq)

                output[key]= torch.stack(repadded_seqs, dim=0)
            else:
                output[key] = pad_sequence(
                    [example[key] for example in batch],
                    batch_first=True,
                    padding_value=padding_value,
                )   
        return output

    def encode_chars(self, word_list: List[str], vocab: Dict[str, int]) -> torch.Tensor:
        """Encode a string of characters into a tensor of input IDs."""
        char_ids_list = []
        for text in word_list:
            char_ids = [vocab.get('<SOS>')]+[vocab.get(c, vocab["<UNK>"]) for c in text]+[vocab.get('<EOS>')]
            char_ids_list.append(char_ids)
        max_len = max(len(ids) for ids in char_ids_list)
        for i in range(len(char_ids_list)):
            char_ids_list[i] += [vocab.get('<PAD>')] * (max_len - len(char_ids_list[i]))        
        return torch.tensor(char_ids_list, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.regular_tokens()
        forms = [t.form for t in tokens]
        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        seq_len = input_ids.size(0)
        upos_ids = torch.zeros(seq_len, dtype=torch.long)
        all_words=[t.form for t in tokens]+['']
        all_lemmas=[t.lemma for t in tokens]+['']
        input_encoded = self.encode_chars(all_words, self.input_vocab)
        output_encoded = self.encode_chars(all_lemmas, self.output_vocab)
        input_char_ids = []
        output_char_ids = []
        seen_words = set()
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            tok = tokens[word_id] if word_id < len(tokens) else None
            if tok is not None:
                upos_ids[i] = self.upos_vocab.get(tok.upos, self.upos_vocab["<UNK>"])
            if word_id not in seen_words and tok is not None:
                seen_words.add(word_id)
                input_char_ids.append(input_encoded[word_id])
                output_char_ids.append(output_encoded[word_id])
            else:
                input_char_ids.append(input_encoded[-1])
                output_char_ids.append(output_encoded[-1])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "upos_ids": upos_ids,
            "input_char_ids": input_char_ids,
            "output_char_ids": output_char_ids
        }


class MWTDataset(Dataset):
    """Sentence-level character dataset for multiword-token expansion."""

    def __init__(
        self,
        sentences: List[Sentence],
        hf_tokenizer_name: str,
        max_length: int = 512,
        output_vocab: Optional[Dict[str, int]] = None,
        input_vocab: Optional[Dict[str, int]] = None,
    ):
        self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.max_length = max_length
        self.sentences = sentences
        self.input_vocab = input_vocab or _build_vocab(
            [char for sentence in sentences for token in sentence.surface_tokens() for char in token.form],
            initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3},
        )
        target_chars = [
            char
            for sentence in sentences
            for token in sentence.surface_tokens()
            for char in self._expansion_targets(sentence).get(token.id, token.form)
        ]
        self.output_vocab = output_vocab or _build_vocab(
            target_chars,
            initial_vocab={"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3},
        )

    @staticmethod
    def _expansion_targets(sentence: Sentence) -> Dict[int | str, str]:
        targets = {}
        regular_tokens = sentence.regular_tokens()
        for token in sentence.tokens:
            if not token.is_multiword():
                continue
            try:
                start, end = map(int, str(token.id).split("-", 1))
            except ValueError:
                continue
            components = [
                part.form
                for part in regular_tokens
                if isinstance(part.id, int) and start <= part.id <= end
            ]
            if len(components) >= 2:
                targets[token.id] = " ".join(components)
        return targets

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sentence = self.sentences[index]
        tokens = sentence.surface_tokens()
        forms = [token.form for token in tokens]
        targets = self._expansion_targets(sentence)
        encoding = self.hf_tok(
            forms,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        input_char_ids = []
        output_char_ids = []
        seen_words = set()
        all_words = [token.form for token in tokens] + [""]
        all_targets = [targets.get(token.id, token.form) for token in tokens] + [""]
        input_encoded = self.encode_chars(all_words, self.input_vocab)
        output_encoded = self.encode_chars(all_targets, self.output_vocab)
        for word_id in word_ids:
            if word_id is None:
                continue
            if word_id not in seen_words and word_id < len(tokens):
                seen_words.add(word_id)
                input_char_ids.append(input_encoded[word_id])
                output_char_ids.append(output_encoded[word_id])
            else:
                input_char_ids.append(input_encoded[-1])
                output_char_ids.append(output_encoded[-1])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "input_char_ids": input_char_ids,
            "output_char_ids": output_char_ids,
        }

    def encode_chars(self, word_list: List[str], vocab: Dict[str, int]) -> torch.Tensor:
        char_ids_list = []
        for text in word_list:
            char_ids = [vocab["<SOS>"]]
            char_ids.extend(vocab.get(char, vocab["<UNK>"]) for char in text)
            char_ids.append(vocab["<EOS>"])
            char_ids_list.append(char_ids)
        max_len = max(len(ids) for ids in char_ids_list)
        for ids in char_ids_list:
            ids.extend([vocab["<PAD>"]] * (max_len - len(ids)))
        return torch.tensor(char_ids_list, dtype=torch.long)

    def collate_fn(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        padding_values = {
            "input_ids": self.hf_tok.pad_token_id,
            "attention_mask": 0,
            "input_char_ids": torch.tensor([[0]]),
            "output_char_ids": torch.tensor([[-100]]),
        }
        output = {}
        for key, padding_value in padding_values.items():
            if key in ("input_char_ids", "output_char_ids"):
                max_len = max(len(example[key][0]) for example in batch)
                padded_sequences = []
                for example in batch:
                    seq = torch.cat([item.unsqueeze(0) for item in example[key]])
                    pad_length = max_len - seq.shape[1]
                    pad_value = padding_value[0][0]
                    padded_sequences.append(torch.cat([
                        seq,
                        torch.full((seq.shape[0], pad_length), pad_value, dtype=torch.long),
                    ], dim=1))
                max_sentence_len = max(seq.shape[0] for seq in padded_sequences)
                repadded = []
                for seq in padded_sequences:
                    pad_length = max_sentence_len - seq.shape[0]
                    repadded.append(torch.cat([
                        seq,
                        torch.full((pad_length, seq.shape[1]), padding_value[0][0], dtype=torch.long),
                    ], dim=0))
                output[key] = torch.stack(repadded)
            else:
                output[key] = pad_sequence(
                    [example[key] for example in batch],
                    batch_first=True,
                    padding_value=padding_value,
                )
        return output
