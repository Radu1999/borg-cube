"""Pipeline orchestration: loads components, runs end-to-end inference."""
from __future__ import annotations

import os
from typing import List, Optional

from src.config import BorgConfig
from src.data.conllu import Sentence, read_conllu, write_conllu


class BorgPipeline:
    """Orchestrates tokenizer → tagger → parser → lemmatizer."""

    def __init__(self, config: Optional[BorgConfig] = None):
        self.config = config or BorgConfig()
        self.tokenizer_model = None
        self.tagger_model = None
        self.parser_model = None
        self.lemmatizer_model = None

    # ------------------------------------------------------------------
    def load_component(self, component: str, model_path: str) -> None:
        """Load a trained component from *model_path*."""
        if component == "tokenizer":
            from src.models.tokenizer import TokenizerModel
            self.tokenizer_model = TokenizerModel.load(model_path, self.config)
        elif component == "tagger":
            from src.models.tagger import TaggerModel
            self.tagger_model = TaggerModel.load(model_path, self.config)
        elif component == "parser":
            from src.models.parser import ParserModel
            self.parser_model = ParserModel.load(model_path, self.config)
        elif component == "lemmatizer":
            from src.models.lemmatizer import LemmatizerModel
            self.lemmatizer_model = LemmatizerModel.load(model_path, self.config)
        else:
            raise ValueError(f"Unknown component: {component}")

    # ------------------------------------------------------------------
    def train_component(
        self,
        component: str,
        train_file: str,
        dev_file: str,
        model_path: str,
        *,
        resume: bool = False,
    ) -> None:
        """Train a component, optionally resuming its last checkpoint."""
        train_sentences = read_conllu(train_file)
        dev_sentences = read_conllu(dev_file)
        component_path = os.path.join(model_path, component)

        if component == "tokenizer":
            from src.models.tokenizer import TokenizerModel
            TokenizerModel.train_model(train_sentences, dev_sentences, self.config, component_path, resume=resume)

        elif component == "tagger":
            from src.models.tagger import TaggerModel
            TaggerModel.train_model(train_sentences, dev_sentences, self.config, component_path, resume=resume)

        elif component == "parser":
            from src.models.parser import ParserModel
            ParserModel.train_model(train_sentences, dev_sentences, self.config, component_path, resume=resume)

        elif component == "lemmatizer":
            from src.models.lemmatizer import LemmatizerModel
            LemmatizerModel.train_model(train_sentences, dev_sentences, self.config, component_path, resume=resume)

        else:
            raise ValueError(f"Unknown component: {component}")

    # ------------------------------------------------------------------
    def process(self, text: str) -> List[Sentence]:
        """Run text through the full pipeline.

        Steps applied in order depending on which components are loaded:
        1. Tokenization (if tokenizer_model present)
        2. POS tagging
        3. Dependency parsing
        4. Lemmatization
        """
        # Step 1: tokenize
        if self.tokenizer_model is not None:
            sentences = self.tokenizer_model.predict(text)
        else:
            # Naive fallback: one sentence, whitespace tokens
            from src.data.conllu import Token
            words = text.split()
            sentences = [
                Sentence(tokens=[Token(id=i + 1, form=w) for i, w in enumerate(words)])
            ]

        # Step 2: tag
        if self.tagger_model is not None:
            sentences = self.tagger_model.predict(sentences)

        # Step 3: parse
        if self.parser_model is not None:
            sentences = self.parser_model.predict(sentences)

        # Step 4: lemmatize
        if self.lemmatizer_model is not None:
            sentences = self.lemmatizer_model.predict(sentences)

        return sentences

    # ------------------------------------------------------------------
    def test(
        self,
        model_path: str,
        input_file: str,
        output_file: str,
    ) -> None:
        """Process an input file and write CoNLL-U output.

        If *input_file* ends with .txt (or contains no CoNLL-U tokens),
        it is treated as raw text and passed through the full pipeline.
        Otherwise it is read as CoNLL-U and only the annotation steps run.
        """
        _, ext = os.path.splitext(input_file)

        if ext.lower() in (".txt",):
            # Raw text input
            with open(input_file, encoding="utf-8") as fh:
                text = fh.read()
            sentences = self.process(text)
        else:
            # CoNLL-U input — skip tokenization
            sentences = read_conllu(input_file)
            if self.tagger_model is not None:
                sentences = self.tagger_model.predict(sentences)
            if self.parser_model is not None:
                sentences = self.parser_model.predict(sentences)
            if self.lemmatizer_model is not None:
                sentences = self.lemmatizer_model.predict(sentences)

        write_conllu(sentences, output_file)
        print(f"Written {len(sentences)} sentences to {output_file}")
