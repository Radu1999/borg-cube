"""Regression tests for per-step training progress metrics."""
import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTrainingProgress(unittest.TestCase):

    def test_each_component_reports_per_step_metrics(self):
        expected_metrics = {
            "tokenizer": {"loss", "avg_loss", "lr"},
            "tagger": {"loss", "avg_loss", "lr", "upos_loss", "xpos_loss", "feats_loss"},
            "parser": {"loss", "avg_loss", "lr", "arc_loss", "rel_loss"},
            "lemmatizer": {"loss", "avg_loss", "lr"},
        }

        for component, expected in expected_metrics.items():
            with self.subTest(component=component):
                path = os.path.join(ROOT, "src", "models", f"{component}.py")
                with open(path, encoding="utf-8") as source_file:
                    tree = ast.parse(source_file.read())

                training_loops = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name)
                    and node.iter.id == "progress"
                ]
                self.assertTrue(training_loops, f"{component} has no progress-wrapped training loop")
                postfix_calls = [
                    node
                    for loop in training_loops
                    for node in ast.walk(loop)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_postfix"
                ]
                self.assertTrue(postfix_calls, f"{component} does not report batch metrics")
                actual = {
                    keyword.arg
                    for call in postfix_calls
                    for keyword in call.keywords
                }
                self.assertTrue(
                    expected.issubset(actual),
                    f"{component} is missing metrics: {expected - actual}",
                )


if __name__ == "__main__":
    unittest.main()
