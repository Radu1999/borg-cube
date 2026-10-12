"""Tests for CoNLL-U evaluation used during component training."""
import contextlib
import io
import unittest

from src.data.conllu import Sentence, Token
from src.models.evaluation import (
    average_f1,
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

    def test_tokenizer_text_uses_multiword_surface_and_spacing(self):
        sentences = [
            Sentence(tokens=[
                Token(id="1-2", form="du", space_after=False),
                Token(id=1, form="de"),
                Token(id=2, form="le"),
                Token(id="2.1", form="empty"),
                Token(id=3, form="."),
            ]),
            Sentence(tokens=[Token(id=1, form="Next")]),
        ]

        self.assertEqual(tokenizer_validation_text(sentences), "du. Next ")

    def test_tokenizer_validation_evaluates_multiword_surface_tokens(self):
        gold = [Sentence(tokens=[
            Token(id="1-2", form="du"),
            Token(id=1, form="de"),
            Token(id=2, form="le"),
            Token(id=3, form="pain"),
        ])]
        text = tokenizer_validation_text(gold)
        predicted = [Sentence(tokens=[
            Token(id=word_id, form=form)
            for word_id, form in enumerate(text.split(), 1)
        ])]

        metrics = evaluate_predictions(gold, predicted)

        self.assertEqual(metrics["Tokens"].f1, 1.0)
        self.assertEqual(metrics["Tokens"].gold_count, 2)
        self.assertEqual(metrics["Tokens"].system_count, 2)
        self.assertEqual(metrics["Sentences"].f1, 1.0)

    def test_prints_only_requested_metrics(self):
        sentence = Sentence(tokens=[Token(id=1, form="word", upos="NOUN")])
        metrics = evaluate_predictions([sentence], [sentence])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_validation_metrics(metrics, ["UPOS"])

        self.assertIn("UPOS", output.getvalue())
        self.assertNotIn("XPOS", output.getvalue())

    def test_average_f1_uses_all_requested_metrics(self):
        sentence = Sentence(tokens=[Token(id=1, form="word", upos="NOUN")])
        metrics = evaluate_predictions([sentence], [sentence])
        metrics["UPOS"].f1 = 0.4
        metrics["XPOS"].f1 = 0.8

        self.assertAlmostEqual(average_f1(metrics, ["UPOS", "XPOS"]), 0.6)


if __name__ == "__main__":
    unittest.main()
