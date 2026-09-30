"""Tests for CoNLL-U evaluation used during component training."""
import contextlib
import io
import unittest

from src.data.conllu import Sentence, Token
from src.models.evaluation import (
    evaluate_predictions,
    print_validation_metrics,
    tokenizer_validation_text,
)


class TestTrainingEvaluation(unittest.TestCase):

    def test_evaluates_component_predictions_as_conllu(self):
        gold = [
            Sentence(tokens=[
                Token(id=1, form="New", upos="ADJ", space_after=False),
                Token(id=2, form="York", upos="PROPN"),
            ])
        ]
        predicted = [
            Sentence(tokens=[
                Token(id=1, form="New", upos="NOUN", space_after=False),
                Token(id=2, form="York", upos="PROPN"),
            ])
        ]

        metrics = evaluate_predictions(gold, predicted)

        self.assertEqual(metrics["Tokens"].f1, 1.0)
        self.assertEqual(metrics["UPOS"].f1, 0.5)

    def test_tokenizer_text_preserves_gold_spaces_and_sentences(self):
        sentences = [
            Sentence(tokens=[
                Token(id=1, form="can", space_after=False),
                Token(id=2, form="not"),
            ]),
            Sentence(tokens=[Token(id=1, form="Next")]),
        ]

        self.assertEqual(tokenizer_validation_text(sentences), "cannot Next ")

    def test_prints_only_requested_metrics(self):
        sentence = Sentence(tokens=[Token(id=1, form="word", upos="NOUN")])
        metrics = evaluate_predictions([sentence], [sentence])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_validation_metrics(metrics, ["UPOS"])

        self.assertIn("UPOS", output.getvalue())
        self.assertNotIn("XPOS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
