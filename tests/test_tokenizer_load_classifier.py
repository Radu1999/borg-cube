"""Regression test for TokenizerModel.load() rebuilding its classifier head.

`BorgBaseModel.load` reconstructs instances via `BorgBaseModel.__init__`
rather than the subclass's own `__init__`, so `TokenizerModel._set_extras`
cannot assume `self.classifier` already exists — it must rebuild it before
loading the saved state dict (see issue #24).
"""
from __future__ import annotations

import torch.nn as nn

from src.config import BorgConfig
from src.models.tokenizer import TokenizerModel


def _make_bare_instance(hidden_size: int = 768) -> TokenizerModel:
    """Build a TokenizerModel instance the way BorgBaseModel.load() does:
    only `nn.Module.__init__` has run, so subclass-specific attributes
    (like `classifier`) are not yet set.
    """
    obj = TokenizerModel.__new__(TokenizerModel)
    nn.Module.__init__(obj)
    obj.config = BorgConfig()
    obj.hidden_size = hidden_size
    return obj


def test_set_extras_rebuilds_missing_classifier():
    obj = _make_bare_instance()
    assert not hasattr(obj, "classifier")

    extras = {"classifier": nn.Linear(obj.hidden_size, TokenizerModel.NUM_LABELS).state_dict()}

    # This used to raise AttributeError: 'TokenizerModel' object has no
    # attribute 'classifier' because _set_extras assumed __init__ had run.
    obj._set_extras(extras)

    assert isinstance(obj.classifier, nn.Linear)
    assert obj.classifier.in_features == obj.hidden_size
    assert obj.classifier.out_features == TokenizerModel.NUM_LABELS


def test_set_extras_loads_saved_weights():
    obj = _make_bare_instance()
    source = nn.Linear(obj.hidden_size, TokenizerModel.NUM_LABELS)

    obj._set_extras({"classifier": source.state_dict()})

    for p1, p2 in zip(obj.classifier.parameters(), source.parameters()):
        assert (p1 == p2).all()
