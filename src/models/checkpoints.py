"""Utilities for retaining training model checkpoints."""
import os
from typing import Any


def save_training_models(
    model: Any, path: str, score: float, best_score: float
) -> float:
    """Save the latest model and update the best model when its score improves."""
    model.save(os.path.join(path, "last"))
    if score > best_score:
        model.save(os.path.join(path, "best"))
        return score
    return best_score
