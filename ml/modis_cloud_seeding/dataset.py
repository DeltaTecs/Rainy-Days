"""PyTorch dataset utilities for the MODIS cloud seeding project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


@dataclass(frozen=True)
class SplitConfig:
    """Define proportions for dataset splits."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):  # pragma: no cover - config guard
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
        if min(self.train, self.val, self.test) < 0:  # pragma: no cover - config guard
            raise ValueError("Split ratios must be non-negative.")


class CloudSeedingDataset(Dataset):
    """Wrap a feature table (CSV/parquet) for PyTorch training."""

    def __init__(
        self,
        table_path: Path,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        label_column: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.path = table_path
        self.frame = self._load_table(table_path)
        self.label_column = label_column
        self.feature_columns = list(feature_columns) if feature_columns else self._infer_feature_columns()
        self.transform = transform

    @staticmethod
    def _load_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _infer_feature_columns(self) -> List[str]:
        candidates = [
            col
            for col in self.frame.columns
            if col not in {self.label_column, "granule", "region_id"}
            and not col.startswith("meta_")
        ]
        return candidates

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        row = self.frame.iloc[index]
        features = torch.tensor(row[self.feature_columns].to_numpy(dtype=float), dtype=torch.float32)
        label_tensor: Optional[torch.Tensor] = None
        if self.label_column:
            label_value = row[self.label_column]
            if pd.isna(label_value):
                label_tensor = torch.tensor(float("nan"), dtype=torch.float32)
            else:
                label_tensor = torch.tensor(float(label_value), dtype=torch.float32)
        if self.transform:
            features = self.transform(features)
        return features, label_tensor


def split_dataset(dataset: CloudSeedingDataset, config: SplitConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test with rounded lengths that sum to dataset size."""

    config.validate()
    total_len = len(dataset)
    if total_len < 3:
        raise ValueError("Need at least 3 samples to perform a train/val/test split.")
    train_len = int(round(total_len * config.train))
    val_len = int(round(total_len * config.val))
    test_len = total_len - train_len - val_len
    # Fix rounding artifacts by borrowing examples.
    while train_len <= 0:
        train_len += 1
        test_len -= 1
    while val_len <= 0:
        val_len += 1
        train_len -= 1
    while test_len <= 0:
        test_len += 1
        train_len -= 1
    if min(train_len, val_len, test_len) <= 0:
        raise ValueError("Invalid split configuration for given dataset size.")
    return tuple(random_split(dataset, [train_len, val_len, test_len]))


__all__ = [
    "CloudSeedingDataset",
    "SplitConfig",
    "split_dataset",
]
