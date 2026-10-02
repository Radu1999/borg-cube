#!/usr/bin/env python3
"""UD evaluation script (CoNLL 2018 shared task evaluation).

Computes:
  Tokens, Sentences, Words, UPOS, XPOS, FEATS, UAS, LAS, MLAS, BLEX, LEMMA

Usage (standalone)::

    python conll18_ud_eval.py gold.conllu system.conllu

Usage (as module)::

    from eval.conll18_ud_eval import evaluate
    metrics = evaluate("gold.conllu", "system.conllu")
    print(metrics["LAS"])
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UDWord:
    """A single (non-MWT, non-empty) token in a UD sentence."""
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str
    # Span in the raw reconstructed text (character offsets)
    span_start: int = 0
    span_end: int = 0


@dataclass
class UDSentence:
    words: List[UDWord]
    text: Optional[str]  # sent_text from # text = comment


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    aligned_count: int
    gold_count: int
    system_count: int

    def __str__(self) -> str:
        return f"P={self.precision:.2%} R={self.recall:.2%} F1={self.f1:.2%}"


# ---------------------------------------------------------------------------
# CoNLL-U reading
# ---------------------------------------------------------------------------

def _is_mwt(id_str: str) -> bool:
    return "-" in id_str


def _is_empty(id_str: str) -> bool:
    return "." in id_str


def _load_conllu(path: str) -> List[UDSentence]:
    sentences: List[UDSentence] = []
    words: List[UDWord] = []
    sent_text: Optional[str] = None

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("# text ="):
                sent_text = line[len("# text ="):].strip()
            elif line.startswith("#"):
                continue
            elif line == "":
                if words:
                    sentences.append(UDSentence(words=words, text=sent_text))
                words = []
                sent_text = None
            else:
                cols = line.split("\t")
                if len(cols) != 10:
                    continue
                id_str = cols[0]
                if _is_mwt(id_str) or _is_empty(id_str):
                    continue
                head = int(cols[6]) if cols[6] != "_" else 0
                words.append(UDWord(
                    form=cols[1],
                    lemma=cols[2],
                    upos=cols[3],
                    xpos=cols[4],
                    feats=cols[5],
                    head=head,
                    deprel=cols[7],
                    deps=cols[8],
                    misc=cols[9],
                ))

    if words:
        sentences.append(UDSentence(words=words, text=sent_text))

    return sentences


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Unicode NFC normalization for character-level alignment."""
    return unicodedata.normalize("NFC", text)


def _build_char_sequence(sentences: List[UDSentence]) -> Tuple[str, List[Tuple[int, int]]]:
    """Build concatenated text and word spans."""
    chars: List[str] = []
    spans: List[Tuple[int, int]] = []

    for sent in sentences:
        for word in sent.words:
            start = len(chars)
            chars.extend(_normalize(word.form))
            end = len(chars)
            spans.append((start, end))

    return "".join(chars), spans


def _align_spans(
    gold_spans: List[Tuple[int, int]],
    sys_spans: List[Tuple[int, int]],
    gold_text: str,
    sys_text: str,
) -> List[Optional[int]]:
    """Align gold word spans to system word spans by character offsets.

    Returns a list of length len(gold_spans) where each entry is either
    an index into sys_spans (exact match) or None.
    """
    sys_span_set: Dict[Tuple[int, int], int] = {sp: i for i, sp in enumerate(sys_spans)}
    alignment: List[Optional[int]] = []
    for span in gold_spans:
        alignment.append(sys_span_set.get(span))
    return alignment


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _feats_intersect(gold_feats: str, sys_feats: str) -> int:
    """Number of feature key=value pairs in common."""
    if gold_feats == "_" and sys_feats == "_":
        return 1  # both empty — counts as agreement
    if gold_feats == "_" or sys_feats == "_":
        return 0
    g = set(gold_feats.split("|"))
    s = set(sys_feats.split("|"))
    return len(g & s)


def _sentence_starts(sentences: List[UDSentence]) -> List[int]:
    """For each flattened word (across all sentences), the global index of
    the first word in that word's own sentence.

    ``UDWord.head`` is a 1-based index that is only meaningful *within its
    own sentence* (0 means the virtual ROOT). To compare a gold head with a
    system head we first need to translate both into global positions in
    the respective flattened word list.
    """
    starts: List[int] = []
    pos = 0
    for sent in sentences:
        starts.extend([pos] * len(sent.words))
        pos += len(sent.words)
    return starts


def _resolve_head_global(head: int, sent_start: int) -> int:
    """Translate a sentence-relative head index into a global word index.

    Returns ``-1`` as a sentinel for the virtual ROOT (head == 0).
    """
    if head == 0:
        return -1
    return sent_start + head - 1


def _mlas_correct(g: UDWord, s: UDWord, heads_match: bool) -> bool:
    """MLAS: LAS + UPOS + FEATS on both dep and head."""
    return (
        heads_match
        and g.deprel.lower() == s.deprel.lower()
        and g.upos == s.upos
        and g.feats == s.feats
    )


def _blex_correct(g: UDWord, s: UDWord, heads_match: bool) -> bool:
    """BLEX: LAS + LEMMA."""
    return (
        heads_match
        and g.deprel.lower() == s.deprel.lower()
        and g.lemma.lower() == s.lemma.lower()
    )


def evaluate(gold_file: str, system_file: str) -> Dict[str, EvalResult]:
    """Evaluate *system_file* against *gold_file*.

    Returns a dict mapping metric name → EvalResult.
    """
    gold_sents = _load_conllu(gold_file)
    sys_sents = _load_conllu(system_file)

    gold_text, gold_spans = _build_char_sequence(gold_sents)
    sys_text, sys_spans = _build_char_sequence(sys_sents)

    alignment = _align_spans(gold_spans, sys_spans, gold_text, sys_text)

    # Flatten word lists
    gold_words = [w for s in gold_sents for w in s.words]
    sys_words = [w for s in sys_sents for w in s.words]

    n_gold = len(gold_words)
    n_sys = len(sys_words)

    # Per-word global sentence-start offsets, used to translate the
    # sentence-relative `head` index into a position in the flattened word
    # list (see `_resolve_head_global`).
    gold_sent_starts = _sentence_starts(gold_sents)
    sys_sent_starts = _sentence_starts(sys_sents)

    # Per-metric counters
    counts: Dict[str, int] = Counter()

    for g_idx, (g_word, sys_idx) in enumerate(zip(gold_words, alignment)):
        if sys_idx is None:
            continue
        s_word = sys_words[sys_idx]

        counts["Tokens_aligned"] += 1

        # UPOS
        if g_word.upos == s_word.upos:
            counts["UPOS_correct"] += 1

        # XPOS
        if g_word.xpos == s_word.xpos:
            counts["XPOS_correct"] += 1

        # FEATS
        if g_word.feats == s_word.feats:
            counts["FEATS_correct"] += 1

        # LEMMA
        if g_word.lemma.lower() == s_word.lemma.lower():
            counts["LEMMA_correct"] += 1

        # Resolve each word's head through the alignment rather than
        # comparing raw sentence-relative indices: `head` is only
        # meaningful within its own sentence, so whenever gold/system
        # token counts differ (e.g. a contraction segmented differently),
        # raw indices drift out of sync even for otherwise-correct arcs.
        # Instead, map the gold word's head to the *word* it points to,
        # find that word's aligned system word, and check whether the
        # system word's own head points to that same aligned word.
        gold_head_global = _resolve_head_global(g_word.head, gold_sent_starts[g_idx])
        gold_head_target_sys = (
            -1 if gold_head_global == -1 else alignment[gold_head_global]
        )
        sys_head_value = _resolve_head_global(s_word.head, sys_sent_starts[sys_idx])
        heads_match = (
            gold_head_target_sys is not None and gold_head_target_sys == sys_head_value
        )

        # UAS (head)
        if heads_match:
            counts["UAS_correct"] += 1

        # LAS (head + deprel, case-insensitive deprel)
        if heads_match and g_word.deprel.lower() == s_word.deprel.lower():
            counts["LAS_correct"] += 1

        # MLAS: LAS + UPOS + FEATS
        if _mlas_correct(g_word, s_word, heads_match):
            counts["MLAS_correct"] += 1

        # BLEX: LAS + LEMMA
        if _blex_correct(g_word, s_word, heads_match):
            counts["BLEX_correct"] += 1

    aligned = counts["Tokens_aligned"]

    def make_result(correct_key: str) -> EvalResult:
        correct = counts.get(correct_key, 0)
        p = correct / max(n_sys, 1)
        r = correct / max(n_gold, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        return EvalResult(
            precision=p, recall=r, f1=f1,
            aligned_count=correct, gold_count=n_gold, system_count=n_sys,
        )

    # Sentence-level
    n_gold_sent = len(gold_sents)
    n_sys_sent = len(sys_sents)
    sent_correct = min(n_gold_sent, n_sys_sent)  # basic approximation

    results: Dict[str, EvalResult] = {
        "Tokens": EvalResult(
            precision=aligned / max(n_sys, 1),
            recall=aligned / max(n_gold, 1),
            f1=2 * aligned / max(n_gold + n_sys, 1),
            aligned_count=aligned,
            gold_count=n_gold,
            system_count=n_sys,
        ),
        "Sentences": EvalResult(
            precision=sent_correct / max(n_sys_sent, 1),
            recall=sent_correct / max(n_gold_sent, 1),
            f1=2 * sent_correct / max(n_gold_sent + n_sys_sent, 1),
            aligned_count=sent_correct,
            gold_count=n_gold_sent,
            system_count=n_sys_sent,
        ),
        "UPOS": make_result("UPOS_correct"),
        "XPOS": make_result("XPOS_correct"),
        "FEATS": make_result("FEATS_correct"),
        "LEMMA": make_result("LEMMA_correct"),
        "UAS": make_result("UAS_correct"),
        "LAS": make_result("LAS_correct"),
        "MLAS": make_result("MLAS_correct"),
        "BLEX": make_result("BLEX_correct"),
    }
    return results


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _print_results(results: Dict[str, EvalResult]) -> None:
    metrics_order = ["Tokens", "Sentences", "UPOS", "XPOS", "FEATS", "LEMMA", "UAS", "LAS", "MLAS", "BLEX"]
    print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>8} {'System':>8}")
    print("-" * 62)
    for name in metrics_order:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<12} {r.precision:>10.2%} {r.recall:>10.2%} {r.f1:>10.2%}"
            f" {r.gold_count:>8} {r.system_count:>8}"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="CoNLL-U UD evaluation (CoNLL 2018 style)"
    )
    parser.add_argument("gold_file", help="Gold CoNLL-U file")
    parser.add_argument("system_file", help="System CoNLL-U file")
    parser.add_argument("--quiet", action="store_true", help="Only print LAS F1")
    args = parser.parse_args(argv)

    results = evaluate(args.gold_file, args.system_file)

    if args.quiet:
        print(f"LAS: {results['LAS'].f1:.4f}")
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
