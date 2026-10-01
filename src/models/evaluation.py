"""Helpers for evaluating component predictions on CoNLL-U validation data."""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Sequence

from eval.conll18_ud_eval import EvalResult, evaluate
from src.data.conllu import Sentence, write_conllu


def evaluate_predictions(
    gold_sentences: Sequence[Sentence],
    predicted_sentences: Sequence[Sentence],
) -> Dict[str, EvalResult]:
    """Evaluate predicted sentences with the CoNLL-U UD evaluator."""
    with tempfile.TemporaryDirectory() as temp_dir:
        gold_path = os.path.join(temp_dir, "gold.conllu")
        predicted_path = os.path.join(temp_dir, "predicted.conllu")
        write_conllu(list(gold_sentences), gold_path)
        write_conllu(list(predicted_sentences), predicted_path)
        return evaluate(gold_path, predicted_path)


def tokenizer_validation_text(sentences: Sequence[Sentence]) -> str:
    """Reconstruct text from gold token forms for tokenizer validation."""
    sentence_texts = []
    for sentence in sentences:
        sentence_texts.append("".join(
            token.form + (" " if token.space_after else "")
            for token in sentence.regular_tokens()
        ))
    return "".join(sentence_texts)


def print_validation_metrics(
    metrics: Dict[str, EvalResult], metric_names: Sequence[str]
) -> None:
    """Print selected CoNLL-U validation metrics in the evaluator's format."""
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>8} {'System':>8}")
    print("  " + "-" * 62)
    for name in metric_names:
        result = metrics[name]
        print(
            f"  {name:<12} {result.precision:>10.2%} {result.recall:>10.2%}"
            f" {result.f1:>10.2%} {result.gold_count:>8} {result.system_count:>8}"
        )


def average_f1(metrics: Dict[str, EvalResult], metric_names: Sequence[str]) -> float:
    """Return the mean F1 score across the requested validation metrics."""
    return sum(metrics[name].f1 for name in metric_names) / len(metric_names)
