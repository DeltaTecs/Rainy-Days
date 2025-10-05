"""Model definitions for MODIS cloud seedability prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResMLPConfig:
    """Configuration for the ResMLP classifier."""

    input_dim: int
    hidden_dim: int
    num_blocks: int
    num_classes: int = 3
    expansion: int = 4
    dropout: float = 0.0


class _ResidualFFNBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner = width * expansion
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, inner),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ResMLP(nn.Module):
    """ResMLP with a LayerNorm/Linear/GELU stem followed by residual FFN blocks."""

    def __init__(self, config: ResMLPConfig) -> None:
        super().__init__()
        stem_layers = [
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
        ]
        if config.dropout > 0:
            stem_layers.append(nn.Dropout(config.dropout))
        self.stem = nn.Sequential(*stem_layers)
        self.blocks = nn.ModuleList(
            [
                _ResidualFFNBlock(config.hidden_dim, config.expansion, config.dropout)
                for _ in range(config.num_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x)
        return self.head(x)


def build_resmlp(config: ResMLPConfig, *, device: torch.device | None = None) -> ResMLP:
    """Factory helper that moves the model to ``device`` when provided."""

    model = ResMLP(config)
    if device is not None:
        model = model.to(device)
    return model


__all__ = ["ResMLP", "ResMLPConfig", "build_resmlp"]
