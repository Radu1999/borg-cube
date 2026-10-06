"""Base model: DeBERTa-v3 + adapter framework."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.config import BorgConfig


class BorgBaseModel(nn.Module):
    """Wraps a DeBERTa-v3 encoder with a task-specific Pfeiffer adapter."""

    def __init__(self, config: BorgConfig, component: str):
        super().__init__()
        self.config = config
        self.component = component

        from transformers import AutoModel, AutoTokenizer
        import adapters

        self.hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
        )
        
        adapters.init(self.encoder)
        pfeiffer_cfg = adapters.SeqBnConfig(reduction_factor=6)
        self.encoder.add_adapter(component, config=pfeiffer_cfg)
        self.encoder.set_active_adapters(component)
        self.encoder.train_adapter(component)

        self.hidden_size: int = self.encoder.config.hidden_size

    # ------------------------------------------------------------------
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return last-hidden-state from the encoder."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs.last_hidden_state  # (batch, seq, hidden)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights and extra metadata to *path* (directory)."""
        os.makedirs(path, exist_ok=True)
        # Save adapter weights
        self.encoder.save_adapter(os.path.join(path, "adapter"), self.component)
        # Save extra heads / vocab stored by subclasses
        extras = self._get_extras()
        if extras:
            torch.save(extras, os.path.join(path, "extras.pt"))
        # Save config
        cfg_dict = self.config.to_dict()
        cfg_dict["component"] = self.component
        with open(os.path.join(path, "borg_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def load(cls, path: str, config: Optional[BorgConfig] = None) -> "BorgBaseModel":
        """Load a saved model from *path*."""
        with open(os.path.join(path, "borg_config.json")) as f:
            cfg_dict = json.load(f)
        component = cfg_dict.pop("component")
        if config is None:
            config = BorgConfig.from_dict(cfg_dict)

        obj = cls.__new__(cls)
        BorgBaseModel.__init__(obj, config, component)

        from transformers import AutoModel
        import adapters

        obj.encoder.load_adapter(os.path.join(path, "adapter"))
        obj.encoder.set_active_adapters(component)

        extras_path = os.path.join(path, "extras.pt")
        if os.path.exists(extras_path):
            extras = torch.load(extras_path, map_location="cpu")
            obj._set_extras(extras)

        return obj

    # ------------------------------------------------------------------
    # Subclasses override these to persist additional state (heads, vocabs).
    def _get_extras(self) -> Dict[str, Any]:
        return {}

    def _set_extras(self, extras: Dict[str, Any]) -> None:
        pass
