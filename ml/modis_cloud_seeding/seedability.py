"""Seedability scoring utilities derived from cloud property forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeedabilityRule:
    """Describe how to score a cloud property into [0, 1]."""

    kind: str
    lo: float
    hi: float
    weight: float

    def score(self, values: pd.Series) -> pd.Series:
        series = values.astype(float)
        span = max(self.hi - self.lo, 1e-6)
        if self.kind == "higher_better":
            raw = (series - self.lo) / span
        elif self.kind == "lower_better":
            raw = (self.hi - series) / span
        elif self.kind == "target_range":
            centre = 0.5 * (self.lo + self.hi)
            half_span = span / 2.0
            raw = 1.0 - (series - centre).abs() / half_span
        else:  # pragma: no cover - configuration guard
            raise ValueError(f"Unsupported rule kind '{self.kind}'.")
        raw = raw.clip(lower=0.0, upper=1.0)
        raw = raw.fillna(0.0)
        return raw


DEFAULT_RULES: Mapping[str, SeedabilityRule] = {
    "cloud_optical_thickness_mean": SeedabilityRule("higher_better", lo=5.0, hi=45.0, weight=0.25),
    "cloud_water_path_mean": SeedabilityRule("higher_better", lo=50.0, hi=400.0, weight=0.2),
    "cloud_top_temperature_mean": SeedabilityRule("lower_better", lo=220.0, hi=270.0, weight=0.2),
    "cloud_top_pressure_mean": SeedabilityRule("lower_better", lo=100.0, hi=550.0, weight=0.2),
    "cloud_effective_radius_mean": SeedabilityRule("target_range", lo=12.0, hi=28.0, weight=0.1),
    "cloud_optical_thickness_std": SeedabilityRule("lower_better", lo=0.0, hi=15.0, weight=0.05),
}


def compute_seedability(
    frame: pd.DataFrame,
    rules: Mapping[str, SeedabilityRule] = DEFAULT_RULES,
) -> pd.Series:
    """Return a seedability score in [0, 1] for each row of ``frame``."""

    missing = [column for column in rules if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for seedability scoring: {missing}")

    total_weight = sum(rule.weight for rule in rules.values())
    weighted_scores = []
    for column, rule in rules.items():
        contribution = rule.score(frame[column]) * rule.weight
        weighted_scores.append(contribution)
    combined = sum(weighted_scores) / total_weight if total_weight else sum(weighted_scores)
    combined = combined.clip(lower=0.0, upper=1.0)
    return combined.fillna(0.0)


def append_seedability(
    frame: pd.DataFrame,
    *,
    rules: Mapping[str, SeedabilityRule] = DEFAULT_RULES,
    score_column: str = "seedability_score",
    percent_column: str = "seedability_percent",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with seedability score and percent columns appended."""

    scores = compute_seedability(frame, rules=rules)
    result = frame.copy()
    result[score_column] = scores
    result[percent_column] = (scores * 100.0).round(1)
    return result


__all__ = [
    "SeedabilityRule",
    "DEFAULT_RULES",
    "append_seedability",
    "compute_seedability",
]
