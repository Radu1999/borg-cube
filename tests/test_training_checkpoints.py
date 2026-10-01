"""Tests for retaining latest and best training models."""
import os
import tempfile
import unittest

from src.models.base import save_training_models


class _FakeModel:
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "score"), "w", encoding="utf-8") as output:
            output.write(str(self.score))


class TestTrainingCheckpoints(unittest.TestCase):

    def test_retains_last_and_best_only(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as path:
            model.score = 0.7
            best_score = save_training_models(model, path, model.score, -1.0)
            model.score = 0.6
            best_score = save_training_models(model, path, model.score, best_score)

            self.assertEqual(set(os.listdir(path)), {"last", "best"})
            with open(os.path.join(path, "last", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.6")
            with open(os.path.join(path, "best", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.7")

    def test_updates_best_when_mean_score_improves(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as path:
            best_score = save_training_models(model, path, 0.5, 0.4)

            self.assertEqual(best_score, 0.5)
            with open(os.path.join(path, "best", "score"), encoding="utf-8") as source:
                self.assertEqual(source.read(), "0.5")


if __name__ == "__main__":
    unittest.main()
