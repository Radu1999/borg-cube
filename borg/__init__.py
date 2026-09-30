"""Public Python API for borg-cube.

Usage::

    from borg import cube
    nlp = cube('en')
    sentences = nlp("The cat sat on the mat.")
    print(sentences)
"""
from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import unquote

from src.config import BorgConfig
from src.data.conllu import Sentence
from src.pipeline.pipeline import BorgPipeline


class Cube:
    """High-level interface to the borg-cube NLP pipeline.

    Parameters
    ----------
    lang:
        Language code (used for locating default model files).
    model_path:
        Base directory containing sub-directories for each component
        (``tokenizer/``, ``tagger/``, ``parser/``, ``lemmatizer/``).
        If *None*, defaults to ``~/.borg_cube/<lang>/``.
    components:
        List of components to load.  Defaults to all four.
    config:
        Optional :class:`~src.config.BorgConfig` instance.
    """

    DEFAULT_COMPONENTS = ["tokenizer", "tagger", "parser", "lemmatizer"]

    def __init__(
        self,
        lang: str = "en",
        model_path: Optional[str] = None,
        components: Optional[List[str]] = None,
        config: Optional[BorgConfig] = None,
    ):
        self.lang = lang
        if model_path is None:
            model_path = os.path.join(os.path.expanduser("~"), ".borg_cube", lang)
        self.model_path = model_path

        if config is None:
            config = BorgConfig(lang=lang)
        self.config = config

        components = self.DEFAULT_COMPONENTS if components is None else components

        self.pipeline = BorgPipeline(config)
        for component in components:
            component_dir = os.path.join(model_path, component)
            if os.path.isdir(component_dir):
                self.pipeline.load_component(component, component_dir)

    # ------------------------------------------------------------------
    def __call__(self, text: str) -> List[Sentence]:
        """Process *text* and return a list of annotated :class:`Sentence` objects."""
        return self.pipeline.process(text)

    def __repr__(self) -> str:
        loaded = [
            c for c in self.DEFAULT_COMPONENTS
            if getattr(self.pipeline, f"{c}_model") is not None
        ]
        return f"Cube(lang={self.lang!r}, loaded={loaded})"


def cube(
    model: str,
    components: Optional[List[str]] = None,
    config: Optional[BorgConfig] = None,
) -> Cube:
    """Load a model by name or from a local ``file://`` model folder.

    Named models are loaded from ``~/.borg_cube/<model>/``. Local folders
    should contain one subdirectory per component.
    """
    if model.startswith("file://"):
        model_path = unquote(model[len("file://"):])
        if not model_path:
            raise ValueError("A file:// model URI must include a folder path")
        lang = os.path.basename(os.path.normpath(model_path)) or "en"
        return Cube(lang=lang, model_path=model_path, components=components, config=config)
    return Cube(lang=model, components=components, config=config)
