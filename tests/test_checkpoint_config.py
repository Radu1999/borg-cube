"""Regression tests for JSON checkpoint configuration and dtype restoration."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.config import BorgConfig
from src.models.base import BorgBaseModel
from src.models.checkpoints import save_training_models


def _make_model(config, component="parser"):
    model = BorgBaseModel.__new__(BorgBaseModel)
    nn.Module.__init__(model)
    model.config = config
    model.component = component
    model.encoder = MagicMock()
    return model


class TestCheckpointConfig(unittest.TestCase):
    def test_json_round_trip_preserves_config_and_dtype(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                config = BorgConfig(dtype=dtype, lang="de", num_epochs=5)
                serialized = json.loads(json.dumps(config.to_dict()))

                self.assertEqual(serialized["dtype"], str(dtype))
                self.assertEqual(BorgConfig.from_dict(serialized), config)
                self.assertIs(config.dtype, dtype)

    def test_missing_dtype_uses_default_and_unknown_metadata_is_ignored(self):
        config = BorgConfig.from_dict({"lang": "de", "component": "parser"})

        self.assertIs(config.dtype, torch.bfloat16)
        self.assertEqual(config.lang, "de")

    def test_accepts_unprefixed_dtype_and_native_dtype(self):
        for dtype in ("float32", torch.float32):
            with self.subTest(dtype=dtype):
                self.assertIs(BorgConfig.from_dict({"dtype": dtype}).dtype, torch.float32)

    def test_invalid_dtype_is_rejected(self):
        for dtype in ("torch.not_a_dtype", "torch.nn", "", None, 32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "Invalid checkpoint dtype"):
                    BorgConfig.from_dict({"dtype": dtype})

    def test_saves_last_and_best_for_every_component(self):
        for component in ("tokenizer", "tagger", "parser", "lemmatizer"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as path:
                model = _make_model(BorgConfig(), component)
                best_score = save_training_models(model, path, 0.7, -1.0)

                self.assertEqual(best_score, 0.7)
                for checkpoint in ("last", "best"):
                    with open(os.path.join(path, checkpoint, "borg_config.json")) as source:
                        values = json.load(source)
                    self.assertEqual(values["component"], component)
                    self.assertEqual(values["dtype"], "torch.bfloat16")
                    self.assertEqual(BorgConfig.from_dict(values), model.config)
                self.assertEqual(model.encoder.save_adapter.call_count, 2)

    def test_load_restores_dtype_before_model_initialization(self):
        config = BorgConfig(dtype=torch.float16, lang="de")
        model = _make_model(config)

        def initialize(obj, loaded_config, component):
            nn.Module.__init__(obj)
            obj.config = loaded_config
            obj.component = component
            obj.encoder = MagicMock()

        with tempfile.TemporaryDirectory() as path:
            model.save(path)
            with patch.object(BorgBaseModel, "__init__", side_effect=initialize, autospec=True):
                loaded = BorgBaseModel.load(path)

            self.assertEqual(loaded.config, config)
            self.assertIs(loaded.config.dtype, torch.float16)
            loaded.encoder.load_adapter.assert_called_once_with(os.path.join(path, "adapter"))
            loaded.encoder.set_active_adapters.assert_called_once_with("parser")

    def test_load_preserves_explicit_config_override(self):
        model = _make_model(BorgConfig())
        override = BorgConfig(dtype=torch.float32, device="cpu")

        def initialize(obj, loaded_config, component):
            nn.Module.__init__(obj)
            obj.config = loaded_config
            obj.component = component
            obj.encoder = MagicMock()

        with tempfile.TemporaryDirectory() as path:
            model.save(path)
            with patch.object(BorgBaseModel, "__init__", side_effect=initialize, autospec=True):
                loaded = BorgBaseModel.load(path, config=override)

            self.assertIs(loaded.config, override)


if __name__ == "__main__":
    unittest.main()
