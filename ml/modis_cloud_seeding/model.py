"""Simple PyTorch model definitions for cloud seeding likelihood prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a fully connected neural network."""

    input_dim: int
    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.1
    output_dim: int = 1


class CloudSeedingNet(nn.Module):
    """Feed-forward network for regression/classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dims = [config.input_dim, *config.hidden_dims, config.output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != config.output_dim:
                layers.append(nn.ReLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(config: ModelConfig, *, device: torch.device | None = None) -> nn.Module:
    """Instantiate the neural network and move it to the target device."""

    model = CloudSeedingNet(config)
    if device is not None:
        model.to(device)
    return model


__all__ = [
    "CloudSeedingNet",
    "ModelConfig",
    "build_model",
]
