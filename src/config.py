"""Configuration dataclass for borg-cube."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BorgConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    adapter_reduction_factor: int = 16
    seed: int = 42
    lang: str = "en"
    device: str = "auto"  # "auto", "cpu", "cuda"
    components: List[str] = field(
        default_factory=lambda: ["tokenizer", "tagger", "parser", "lemmatizer"]
    )
    eval_batch_size: int = 32

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
