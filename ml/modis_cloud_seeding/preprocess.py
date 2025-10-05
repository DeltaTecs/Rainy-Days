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

try:  # pragma: no cover - optional dependency
    from pyhdf.SD import SD, SDC  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SD = None
    SDC = None


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


class PyHDFUnavailable(RuntimeError):
    """Raised when pyhdf is required but missing."""


@dataclass
class _HDF4Variable:
    """Mimic a minimal xarray ``DataArray`` interface for HDF4 usage."""

    values: np.ndarray


class _HDF4Dataset:
    """Provide xarray-like accessors backed by ``pyhdf`` for HDF4 granules."""

    def __init__(self, path: Path):
        if SD is None or SDC is None:  # pragma: no cover - import guard
            raise PyHDFUnavailable(
                "pyhdf is required to read HDF4 granules. Install it via conda-forge, e.g."
                " `conda install -c conda-forge pyhdf`."
            )
        self._sd = SD(str(path), SDC.READ)
        self._datasets = self._sd.datasets()
        self._cache: Dict[str, _HDF4Variable] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __getitem__(self, name: str) -> _HDF4Variable:
        if name not in self._datasets:
            raise KeyError(name)
        if name not in self._cache:
            dataset = self._sd.select(name)
            try:
                array = _read_hdf4_dataset(dataset)
            finally:
                dataset.endaccess()
            self._cache[name] = _HDF4Variable(array)
        return self._cache[name]

    def close(self) -> None:
        if getattr(self, "_sd", None) is not None:
            self._sd.end()
            self._sd = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _apply_scale_offset(array: np.ndarray, attrs: Dict[str, object]) -> np.ndarray:
    """Apply ``add_offset`` and ``scale_factor`` metadata to the raw array."""

    result = array.astype(np.float64, copy=False)
    offset = attrs.get("add_offset")
    scale = attrs.get("scale_factor")
    if offset is not None:
        result = result - float(offset)
    if scale is not None:
        result = result * float(scale)
    return result.astype(np.float32, copy=False)


def _read_hdf4_dataset(dataset) -> np.ndarray:
    """Return a masked/scaled numpy array from a ``pyhdf`` dataset."""

    array = np.array(dataset.get(), dtype=np.float32)
    attrs = dataset.attributes()

    fill_values = {
        attrs.get("_FillValue"),
        attrs.get("missing_value"),
        attrs.get("MissingValue"),
    }
    fill_values = {value for value in fill_values if value is not None}
    for fill in fill_values:
        array[array == float(fill)] = np.nan

    valid_range = attrs.get("valid_range")
    if valid_range:
        lo, hi = valid_range
        mask = (array < float(lo)) | (array > float(hi))
        array = array.astype(np.float32)
        array[mask] = np.nan

    return _apply_scale_offset(array, attrs)


def load_granule(path: Path):
    """Load a MODIS granule (HDF, HDF4, or netCDF) using available backends."""

    errors: List[str] = []
    if xr is not None:
        engines = ("netcdf4", "h5netcdf", None)
        for engine in engines:
            try:
                return xr.open_dataset(path, engine=engine, mask_and_scale=True)
            except Exception as exc:  # pragma: no cover - passthrough for clearer error
                errors.append(f"engine '{engine}': {exc}")
    if SD is not None and SDC is not None:
        try:
            return _HDF4Dataset(path)
        except Exception as exc:  # pragma: no cover - fallback diagnostics
            errors.append(f"pyhdf: {exc}")

    if xr is None and (SD is None or SDC is None):  # pragma: no cover - no backends available
        raise XarrayUnavailable(
            "No IO backend available. Install xarray with netCDF4 support or pyhdf for HDF4 granules."
        )

    detail = ", ".join(errors) or "no backend attempts were made"
    hint = (
        "Install the appropriate IO backend. For HDF4 granules install pyhdf (e.g."
        " `conda install -c conda-forge pyhdf`)."
    )
    raise ValueError(f"Failed to open granule '{path}'. Tried backends: {detail}. {hint}")


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
    try:
        for name, spec in features.items():
            if spec.variable not in dataset:
                rows[name] = float("nan")
                continue
            data_array = dataset[spec.variable]
            rows[name] = spec.reduce(data_array.values)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

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
    "PyHDFUnavailable",
    "XarrayUnavailable",
    "extract_features",
    "export_features",
    "load_granule",
    "save_feature_config",
]
