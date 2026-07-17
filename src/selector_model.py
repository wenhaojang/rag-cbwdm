"""Selector models and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class FeatureMLPSelector(nn.Module):
    """Small MLP that maps numeric selector features to scalar logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar logits with shape [batch]."""
        return self.network(features).squeeze(-1)


def save_feature_selector_checkpoint(
    path: str | Path,
    model: FeatureMLPSelector,
    feature_dim: int,
    config_dict: dict[str, Any],
    feature_names: list[str] | None = None,
) -> None:
    """Save a feature selector checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "feature_mlp",
            "feature_dim": feature_dim,
            "state_dict": model.state_dict(),
            "config": config_dict,
            "feature_names": feature_names,
        },
        checkpoint_path,
    )


def load_feature_selector_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a feature selector checkpoint dictionary."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    if checkpoint.get("model_type") != "feature_mlp":
        raise ValueError(f"Unsupported selector checkpoint type: {checkpoint.get('model_type')}")
    config = checkpoint.get("config", {})
    model = FeatureMLPSelector(
        input_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["model"] = model
    return checkpoint


class CrossEncoderSelector(nn.Module):
    """Placeholder for future state-aware text cross-encoder selector."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        raise NotImplementedError("CrossEncoderSelector will be implemented in a later stage.")
