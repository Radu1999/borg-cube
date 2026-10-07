"""Configuration dataclass for borg-cube."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import torch


@dataclass
class BorgConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_ratio: float = 0
    adapter_reduction_factor: int = 16
    seed: int = 42
    lang: str = "en"
    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"  # "auto", "cpu", "cuda"
    components: List[str] = field(
        default_factory=lambda: ["tokenizer", "tagger", "parser", "lemmatizer"]
    )
    eval_batch_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        """Return checkpoint metadata with a JSON-compatible dtype."""
        values = dict(self.__dict__)
        values["dtype"] = str(self.dtype)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BorgConfig":
        """Restore checkpoint metadata, including its PyTorch dtype."""
        fields = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        if "dtype" in fields:
            dtype = fields["dtype"]
            if isinstance(dtype, str):
                name = dtype[len("torch."):] if dtype.startswith("torch.") else dtype
                dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid checkpoint dtype: {fields['dtype']!r}")
            fields["dtype"] = dtype
        return cls(**fields)

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device