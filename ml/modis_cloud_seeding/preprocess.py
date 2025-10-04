"""Transform raw MODIS cloud property granules into model-ready tabular features."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import xarray as xr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xr = None


@dataclass
class FeatureSpec:
    """Describe how to extract a single feature from a MODIS variable."""

    variable: str
    reducer: str = "mean"
    scale: Optional[float] = None

    def reduce(self, data: np.ndarray) -> float:
        array = data[np.isfinite(data)]
        if array.size == 0:
            return float("nan")
        if self.reducer == "mean":
            value = float(array.mean())
        elif self.reducer == "median":
            value = float(np.median(array))
        elif self.reducer == "std":
            value = float(array.std())
        elif self.reducer == "max":
            value = float(array.max())
        elif self.reducer == "min":
            value = float(array.min())
        else:  # pragma: no cover - configuration guard
            raise ValueError(f"Unsupported reducer '{self.reducer}'.")
        if self.scale is not None:
            value *= self.scale
        return value


DEFAULT_FEATURES: Dict[str, FeatureSpec] = {
    "cloud_optical_thickness_mean": FeatureSpec(
        variable="Cloud_Optical_Thickness", reducer="mean"
    ),
    "cloud_optical_thickness_std": FeatureSpec(
        variable="Cloud_Optical_Thickness", reducer="std"
    ),
    "cloud_effective_radius_mean": FeatureSpec(
        variable="Cloud_Effective_Radius", reducer="mean", scale=1.0
    ),
    "cloud_water_path_mean": FeatureSpec(
        variable="Cloud_Water_Path", reducer="mean"
    ),
    "cloud_top_temperature_mean": FeatureSpec(
        variable="Cloud_Top_Temperature", reducer="mean"
    ),
    "cloud_top_pressure_mean": FeatureSpec(
        variable="Cloud_Top_Pressure", reducer="mean"
    ),
}


class XarrayUnavailable(RuntimeError):
    """Raised when xarray is not installed."""


def load_granule(path: Path) -> "xr.Dataset":
    """Load a MODIS granule (HDF or netCDF) using xarray."""

    if xr is None:  # pragma: no cover - import guard
        raise XarrayUnavailable("xarray is required for preprocessing. Install xarray and netCDF4.")
    # ``engine=None`` lets xarray pick h5netcdf/netCDF4 based on file extension.
    return xr.open_dataset(path, engine=None, mask_and_scale=True)


def extract_features(
    path: Path,
    features: Dict[str, FeatureSpec] = DEFAULT_FEATURES,
    *,
    region_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Extract configured features from a single granule as a one-row DataFrame."""

    dataset = load_granule(path)
    rows: Dict[str, float] = {}
    for name, spec in features.items():
        if spec.variable not in dataset:
            rows[name] = float("nan")
            continue
        data_array = dataset[spec.variable]
        rows[name] = spec.reduce(data_array.values)
    dataset.close()

    result = {
        "granule": path.name,
        **rows,
    }
    if region_id:
        result["region_id"] = region_id
    if metadata:
        for key, value in metadata.items():
            result[f"meta_{key}"] = value
    return pd.DataFrame([result])


def export_features(
    granules: Iterable[Path],
    output_csv: Path,
    *,
    features: Dict[str, FeatureSpec] = DEFAULT_FEATURES,
    region_id: Optional[str] = None,
) -> None:
    """Process multiple granules and persist their features to CSV."""

    frames = [
        extract_features(path, features=features, region_id=region_id) for path in granules
    ]
    table = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)


def save_feature_config(features: Dict[str, FeatureSpec], path: Path) -> None:
    """Persist the feature configuration for reproducibility."""

    serialisable = {
        name: {
            "variable": spec.variable,
            "reducer": spec.reducer,
            "scale": spec.scale,
        }
        for name, spec in features.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialisable, indent=2))


__all__ = [
    "DEFAULT_FEATURES",
    "FeatureSpec",
    "XarrayUnavailable",
    "extract_features",
    "export_features",
    "load_granule",
    "save_feature_config",
]
