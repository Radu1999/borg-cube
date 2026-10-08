"""Unit tests for borg-cube NLP pipeline.

These tests do NOT download or load transformer models — they mock the
heavy dependencies and focus on data structures, config, and eval logic.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CONLLU = textwrap.dedent("""\
    # sent_id = 1
    # text = The cat sat on the mat .
    1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t_\t_
    2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsub\t_\t_
    3\tsat\tsit\tVERB\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t0\troot\t_\t_
    4\ton\ton\tADP\tIN\t_\t3\tobl\t_\t_
    5\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t6\tdet\t_\t_
    6\tmat\tmat\tNOUN\tNN\tNumber=Sing\t4\tnmod\t_\t_
    7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

    # sent_id = 2
    # text = Dogs bark .
    1\tDogs\tdog\tNOUN\tNNS\tNumber=Plur\t2\tnsubj\t_\t_
    2\tbark\tbark\tVERB\tVBP\tMood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\t_
    3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_

""")


# ---------------------------------------------------------------------------
# 1. CoNLL-U reader / writer round-trip
# ---------------------------------------------------------------------------

class TestConllu(unittest.TestCase):

    def _write_tmp(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".conllu", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_read_sentence_count(self):
        from src.data.conllu import read_conllu
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            sents = read_conllu(path)
            self.assertEqual(len(sents), 2)
        finally:
            os.unlink(path)

    def test_read_token_fields(self):
        from src.data.conllu import read_conllu
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            sents = read_conllu(path)
            tok = sents[0].regular_tokens()[0]
            self.assertEqual(tok.form, "The")
            self.assertEqual(tok.lemma, "the")
            self.assertEqual(tok.upos, "DET")
            self.assertEqual(tok.head, 3)
            self.assertEqual(tok.deprel, "det")
        finally:
            os.unlink(path)

    def test_read_feats_as_dict(self):
        from src.data.conllu import read_conllu
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            sents = read_conllu(path)
            tok = sents[0].regular_tokens()[0]
            self.assertIsInstance(tok.feats, dict)
            self.assertEqual(tok.feats.get("Definite"), "Def")
        finally:
            os.unlink(path)

    def test_round_trip(self):
        from src.data.conllu import read_conllu, write_conllu
        in_path = self._write_tmp(SAMPLE_CONLLU)
        out_path = tempfile.mktemp(suffix=".conllu")
        try:
            sents = read_conllu(in_path)
            write_conllu(sents, out_path)
            sents2 = read_conllu(out_path)
            self.assertEqual(len(sents), len(sents2))
            for s1, s2 in zip(sents, sents2):
                self.assertEqual(len(s1.tokens), len(s2.tokens))
                for t1, t2 in zip(s1.regular_tokens(), s2.regular_tokens()):
                    self.assertEqual(t1.form, t2.form)
                    self.assertEqual(t1.lemma, t2.lemma)
                    self.assertEqual(t1.upos, t2.upos)
                    self.assertEqual(t1.head, t2.head)
                    self.assertEqual(t1.deprel, t2.deprel)
        finally:
            os.unlink(in_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_comments_preserved(self):
        from src.data.conllu import read_conllu
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            sents = read_conllu(path)
            self.assertTrue(any("sent_id" in c for c in sents[0].comments))
        finally:
            os.unlink(path)

    def test_sentence_repr(self):
        from src.data.conllu import read_conllu
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            sents = read_conllu(path)
            r = repr(sents[0])
            self.assertIn("The", r)
        finally:
            os.unlink(path)

    def test_multiword_token(self):
        """Multi-word tokens should not appear in regular_tokens()."""
        mwt_conllu = textwrap.dedent("""\
            1-2\tdu\t_\t_\t_\t_\t_\t_\t_\t_
            1\tde\tde\tADP\tP\t_\t4\tcase\t_\t_
            2\tle\tle\tDET\tDdfs\t_\t4\tdet\t_\t_
            3\tchat\tchat\tNOUN\tNc\t_\t0\troot\t_\t_

        """)
        from src.data.conllu import read_conllu
        path = self._write_tmp(mwt_conllu)
        try:
            sents = read_conllu(path)
            self.assertEqual(len(sents), 1)
            regular = sents[0].regular_tokens()
            self.assertEqual(len(regular), 3)  # de, le, chat — not du
        finally:
            os.unlink(path)

    def test_feats_str_roundtrip(self):
        from src.data.conllu import Token
        tok = Token(id=1, form="ran", feats={"Tense": "Past", "VerbForm": "Fin"})
        feats_str = tok.feats_str()
        self.assertIn("Tense=Past", feats_str)
        self.assertIn("VerbForm=Fin", feats_str)


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_defaults(self):
        from src.config import BorgConfig
        cfg = BorgConfig()
        self.assertEqual(cfg.model_name, "microsoft/deberta-v3-base")
        self.assertEqual(cfg.num_epochs, 10)
        self.assertIn("tokenizer", cfg.components)

    def test_override(self):
        from src.config import BorgConfig
        cfg = BorgConfig(num_epochs=5, lang="de", device="cpu")
        self.assertEqual(cfg.num_epochs, 5)
        self.assertEqual(cfg.lang, "de")
        self.assertEqual(cfg.device, "cpu")

    def test_resolve_device_cpu(self):
        from src.config import BorgConfig
        cfg = BorgConfig(device="cpu")
        self.assertEqual(cfg.resolve_device(), "cpu")

    def test_resolve_device_auto(self):
        """Auto device should return a valid string."""
        from src.config import BorgConfig
        cfg = BorgConfig(device="auto")
        d = cfg.resolve_device()
        self.assertIn(d, ("cpu", "cuda"))


# ---------------------------------------------------------------------------
# 3. Pipeline initialization (no model loading)
# ---------------------------------------------------------------------------

class TestPipelineInit(unittest.TestCase):

    def test_default_config(self):
        from src.pipeline.pipeline import BorgPipeline
        pipeline = BorgPipeline()
        self.assertIsNotNone(pipeline.config)

    def test_components_none_initially(self):
        from src.pipeline.pipeline import BorgPipeline
        pipeline = BorgPipeline()
        self.assertIsNone(pipeline.tokenizer_model)
        self.assertIsNone(pipeline.tagger_model)
        self.assertIsNone(pipeline.parser_model)
        self.assertIsNone(pipeline.lemmatizer_model)

    def test_process_no_models_whitespace(self):
        """Without any models, process() should still return sentences."""
        from src.pipeline.pipeline import BorgPipeline
        pipeline = BorgPipeline()
        sents = pipeline.process("Hello world .")
        self.assertGreater(len(sents), 0)
        words = [t.form for t in sents[0].regular_tokens()]
        self.assertIn("Hello", words)

    def test_load_unknown_component_raises(self):
        from src.pipeline.pipeline import BorgPipeline
        pipeline = BorgPipeline()
        with self.assertRaises(ValueError):
            pipeline.load_component("nonexistent", "/tmp/fake")


# ---------------------------------------------------------------------------
# 4. Edit scripts (lemmatizer utility)
# ---------------------------------------------------------------------------

class TestEditScripts(unittest.TestCase):

    def test_identity(self):
        from src.data.dataset import _compute_edit_script
        from src.models.lemmatizer import _apply_edit_script
        script = _compute_edit_script("cat", "cat")
        self.assertIn("k", script)

    def test_simple_suffix(self):
        from src.data.dataset import _compute_edit_script
        script = _compute_edit_script("running", "run")
        # prefix 3 (run), strip 4 (ning), add ""
        self.assertTrue(script.startswith("k3"))

    def test_apply_reconstructs(self):
        from src.data.dataset import _compute_edit_script
        from src.models.lemmatizer import _apply_edit_script
        pairs = [
            ("running", "run"),
            ("cats", "cat"),
            ("went", "go"),
            ("The", "the"),
        ]
        for form, lemma in pairs:
            script = _compute_edit_script(form, lemma)
            reconstructed = _apply_edit_script(form, script)
            # We just check it doesn't crash and returns a string
            self.assertIsInstance(reconstructed, str)


# ---------------------------------------------------------------------------
# 5. Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):

    def _write_tmp(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".conllu", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_perfect_score(self):
        from eval.conll18_ud_eval import evaluate
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            results = evaluate(path, path)
            self.assertAlmostEqual(results["LAS"].f1, 1.0, places=4)
            self.assertAlmostEqual(results["UPOS"].f1, 1.0, places=4)
            self.assertAlmostEqual(results["LEMMA"].f1, 1.0, places=4)
        finally:
            os.unlink(path)

    def test_zero_score_wrong_heads(self):
        from eval.conll18_ud_eval import evaluate
        wrong_conllu = textwrap.dedent("""\
            1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t99\tdet\t_\t_
            2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t99\tnsubj\t_\t_
            3\tsat\tsit\tVERB\tVBD\t_\t99\troot\t_\t_

        """)
        gold_conllu = textwrap.dedent("""\
            1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t_\t_
            2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_
            3\tsat\tsit\tVERB\tVBD\t_\t0\troot\t_\t_

        """)
        gold_path = self._write_tmp(gold_conllu)
        wrong_path = self._write_tmp(wrong_conllu)
        try:
            results = evaluate(gold_path, wrong_path)
            self.assertLess(results["LAS"].f1, 0.5)
        finally:
            os.unlink(gold_path)
            os.unlink(wrong_path)

    def test_uas_survives_tokenization_mismatch(self):
        """A correct dependency attachment must not be penalized just
        because an earlier token in the sentence was segmented
        differently by the system (regression test: UAS/LAS used to
        compare raw sentence-relative head indices directly, so any
        token-count drift before a word silently shifted what its head
        index "meant", making otherwise-correct arcs count as wrong).
        """
        from eval.conll18_ud_eval import evaluate
        gold_conllu = textwrap.dedent("""\
            1\tI\tI\tPRON\tPRP\t_\t4\tnsubj\t_\t_
            2\tdo\tdo\tAUX\tVBP\t_\t4\taux\t_\t_
            3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
            4\tknow\tknow\tVERB\tVBP\t_\t0\troot\t_\t_
            5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

        """)
        # System under-segments the contraction ("do" + "n't" -> "don't"),
        # dropping one token, but every surviving word's attachment is
        # semantically identical to gold (both point to "know"/ROOT).
        sys_conllu = textwrap.dedent("""\
            1\tI\tI\tPRON\tPRP\t_\t3\tnsubj\t_\t_
            2\tdon't\tdo not\tAUX\tVBP\t_\t3\taux\t_\t_
            3\tknow\tknow\tVERB\tVBP\t_\t0\troot\t_\t_
            4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

        """)
        gold_path = self._write_tmp(gold_conllu)
        sys_path = self._write_tmp(sys_conllu)
        try:
            results = evaluate(gold_path, sys_path)
            # "I", "know" and "." are aligned (identical forms) and all
            # three have a semantically-correct attachment to "know" /
            # ROOT, so UAS/LAS should match the Tokens alignment rate
            # exactly rather than collapsing further.
            self.assertAlmostEqual(results["UAS"].f1, results["Tokens"].f1, places=4)
            self.assertAlmostEqual(results["LAS"].f1, results["Tokens"].f1, places=4)
        finally:
            os.unlink(gold_path)
            os.unlink(sys_path)

    def test_result_fields(self):
        from eval.conll18_ud_eval import evaluate, EvalResult
        path = self._write_tmp(SAMPLE_CONLLU)
        try:
            results = evaluate(path, path)
            for name in ["Tokens", "Sentences", "UPOS", "LAS", "UAS", "LEMMA"]:
                self.assertIn(name, results)
                self.assertIsInstance(results[name], EvalResult)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Cube class (mocked — no model loading)
# ---------------------------------------------------------------------------

class TestCubeAPI(unittest.TestCase):

    def test_cube_init_no_models(self):
        """Cube should initialize without errors even if no models exist."""
        from borg import Cube
        # Point to a non-existent dir so no models are loaded
        cube = Cube(lang="en", model_path="/tmp/__nonexistent_borg_cube__")
        self.assertIsNotNone(cube.pipeline)

    def test_cube_call_returns_list(self):
        """Cube.__call__ should return a list even without models."""
        from borg import Cube
        cube = Cube(lang="en", model_path="/tmp/__nonexistent_borg_cube__")
        result = cube("Hello world")
        self.assertIsInstance(result, list)

    def test_cube_repr(self):
        from borg import Cube
        cube = Cube(lang="de", model_path="/tmp/__nonexistent_borg_cube__")
        r = repr(cube)
        self.assertIn("de", r)

    def test_cube_factory_loads_named_model_components(self):
        from borg import cube

        with tempfile.TemporaryDirectory() as home:
            component_dir = os.path.join(home, ".borg_cube", "ro_rrt", "tagger")
            os.makedirs(component_dir)
            with patch("borg.os.path.expanduser", return_value=home):
                with patch("borg.BorgPipeline.load_component") as load_component:
                    api = cube("ro_rrt", components=["tagger"])

        self.assertIsNotNone(api.pipeline)
        load_component.assert_called_once_with("tagger", component_dir)

    def test_cube_factory_loads_file_uri_and_empty_components(self):
        from borg import cube

        with tempfile.TemporaryDirectory() as model_path:
            os.makedirs(os.path.join(model_path, "tokenizer"))
            with patch("borg.BorgPipeline.load_component") as load_component:
                api = cube("file://" + model_path, components=[])

        load_component.assert_not_called()
        self.assertEqual(api.model_path, model_path)


class TestPipelineTrainingPaths(unittest.TestCase):

    def test_train_component_saves_in_component_subfolder(self):
        from src.pipeline.pipeline import BorgPipeline
        tagger_module = types.ModuleType("src.models.tagger")
        tagger_module.TaggerModel = types.SimpleNamespace(train_model=MagicMock())

        with patch("src.pipeline.pipeline.read_conllu", return_value=[]):
            with patch.dict(sys.modules, {"src.models.tagger": tagger_module}):
                BorgPipeline().train_component("tagger", "train.conllu", "dev.conllu", "/models/ro_rrt")

        train_model = tagger_module.TaggerModel.train_model
        self.assertEqual(train_model.call_args.args[3], "/models/ro_rrt/tagger")
        self.assertEqual(train_model.call_args.kwargs, {"resume": False})

    def test_train_component_forwards_resume_to_all_trainers(self):
        from src.pipeline.pipeline import BorgPipeline

        for component in ("tokenizer", "tagger", "parser", "lemmatizer"):
            with self.subTest(component=component):
                module_name = f"src.models.{component}"
                module = types.ModuleType(module_name)
                train_model = MagicMock()
                setattr(module, f"{component.capitalize()}Model", types.SimpleNamespace(train_model=train_model))
                pipeline = BorgPipeline()
                with patch("src.pipeline.pipeline.read_conllu", return_value=[]):
                    with patch.dict(sys.modules, {module_name: module}):
                        pipeline.train_component(
                            component, "train.conllu", "dev.conllu", "/models/ro_rrt",
                            resume=True,
                        )
                train_model.assert_called_once_with(
                    [], [], pipeline.config, f"/models/ro_rrt/{component}", resume=True,
                )


class TestCLITrainingResume(unittest.TestCase):

    def test_resume_flag_defaults_to_false(self):
        from borg.cli import _build_parser

        args = _build_parser().parse_args([
            "train", "--component", "parser", "train.conllu", "dev.conllu", "/models/ro_rrt",
        ])
        self.assertFalse(args.resume)

    def test_cli_forwards_resume_flag(self):
        from borg.cli import main

        for resume in (False, True):
            with self.subTest(resume=resume):
                argv = [
                    "train", "--component", "parser", "train.conllu", "dev.conllu", "/models/ro_rrt",
                ]
                if resume:
                    argv.append("--resume")
                with patch("borg.cli.BorgPipeline") as pipeline_class, patch("builtins.print"):
                    main(argv)
                pipeline_class.return_value.train_component.assert_called_once_with(
                    "parser", "train.conllu", "dev.conllu", "/models/ro_rrt", resume=resume,
                )


# ---------------------------------------------------------------------------
# 7. Adapter configuration
# ---------------------------------------------------------------------------

class TestAdapterConfig(unittest.TestCase):

    def test_pfeiffer_adapter_used_in_base(self):
        """BorgBaseModel must configure a Pfeiffer adapter with reduction_factor=6."""
        import ast

        base_path = os.path.join(ROOT, "src", "models", "base.py")
        with open(base_path) as fh:
            source = fh.read()

        tree = ast.parse(source)

        pfeiffer_calls = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "PfeifferConfig"
            )
        ]

        self.assertTrue(
            pfeiffer_calls,
            "Expected at least one call to adapters.PfeifferConfig in base.py",
        )

        # Every PfeifferConfig call must pass reduction_factor=6
        for call in pfeiffer_calls:
            reduction_factors = [
                ast.literal_eval(kw.value)
                for kw in call.keywords
                if kw.arg == "reduction_factor"
            ]
            self.assertTrue(
                reduction_factors,
                "PfeifferConfig called without a reduction_factor keyword argument",
            )
            self.assertTrue(
                all(rf == 6 for rf in reduction_factors),
                f"All PfeifferConfig calls must use reduction_factor=6; got {reduction_factors}",
            )

    def test_lora_not_used_in_base(self):
        """BorgBaseModel must not use LoRAConfig."""
        import ast

        base_path = os.path.join(ROOT, "src", "models", "base.py")
        with open(base_path) as fh:
            source = fh.read()

        tree = ast.parse(source)

        lora_calls = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "LoRAConfig"
            )
        ]

        self.assertFalse(
            lora_calls,
            "LoRAConfig should no longer be used in base.py; use PfeifferConfig instead",
        )


if __name__ == "__main__":
    unittest.main()
