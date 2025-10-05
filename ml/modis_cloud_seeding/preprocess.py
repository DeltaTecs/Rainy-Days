"""Transform raw MODIS cloud property granules into model-ready tabular features."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


LatLonBox = Tuple[float, float, float, float]
GridSize = Tuple[float, float]


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


def _dataset_contains(dataset, name: str) -> bool:
    try:
        return name in dataset
    except Exception:  # pragma: no cover - defensive guard
        return False


def _dataset_to_numpy(dataset, name: str) -> np.ndarray:
    data = dataset[name]
    values = getattr(data, "values", data)
    return np.asarray(values, dtype=np.float32)


def _extract_lat_lon(dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    lat_keys = ("Latitude", "latitude", "Latitude_deg")
    lon_keys = ("Longitude", "longitude", "Longitude_deg")
    lat = lon = None
    for key in lat_keys:
        if _dataset_contains(dataset, key):
            lat = _dataset_to_numpy(dataset, key)
            break
    for key in lon_keys:
        if _dataset_contains(dataset, key):
            lon = _dataset_to_numpy(dataset, key)
            break
    if lat is None or lon is None:
        return None, None
    if lat.shape != lon.shape:
        return None, None
    return lat, lon


def _bbox_mask(lat: np.ndarray, lon: np.ndarray, bbox: LatLonBox) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


def _apply_mask(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return values
    if values.shape != mask.shape:
        return values
    return np.where(mask, values, np.nan)


def _extract_granule_metadata(granule_name: str) -> Dict[str, str]:
    pattern = re.compile(r"A(\d{4})(\d{3})\.(\d{2})(\d{2})")
    match = pattern.search(granule_name)
    metadata: Dict[str, str] = {}
    if match:
        year, doy, hour, minute = match.groups()
        timestamp = datetime.strptime(f"{year}{doy}{hour}{minute}", "%Y%j%H%M")
        metadata["timestamp_utc"] = timestamp.replace(tzinfo=None).isoformat()
        metadata["date"] = timestamp.date().isoformat()
    if granule_name.startswith("MOD"):
        metadata["platform"] = "Terra"
    elif granule_name.startswith("MYD"):
        metadata["platform"] = "Aqua"
    else:
        metadata["platform"] = "Unknown"
    return metadata


def _normalise_grid_size(grid_size: Optional[Sequence[float]]) -> Optional[GridSize]:
    if grid_size is None:
        return None
    if isinstance(grid_size, (float, int)):
        value = float(grid_size)
        if value <= 0:
            raise ValueError("Grid size must be positive.")
        return (value, value)
    if len(grid_size) != 2:
        raise ValueError("Grid size must contain exactly two floats (lat_step, lon_step).")
    lat_step = float(grid_size[0])
    lon_step = float(grid_size[1])
    if lat_step <= 0 or lon_step <= 0:
        raise ValueError("Grid size components must be positive.")
    return (lat_step, lon_step)


def _grid_cells(
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    bbox: LatLonBox,
    grid_size: GridSize,
) -> List[Dict[str, object]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_step, lon_step = grid_size

    mask = mask & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(mask):
        return []

    lat_flat = lat.reshape(-1)
    lon_flat = lon.reshape(-1)
    mask_flat = mask.reshape(-1)
    valid_idx = np.flatnonzero(mask_flat)
    if valid_idx.size == 0:
        return []

    lat_sel = lat_flat[valid_idx]
    lon_sel = lon_flat[valid_idx]

    n_lat = max(1, int(math.ceil((lat_max - lat_min) / lat_step)))
    n_lon = max(1, int(math.ceil((lon_max - lon_min) / lon_step)))

    row_idx = np.floor((lat_sel - lat_min) / lat_step).astype(int)
    col_idx = np.floor((lon_sel - lon_min) / lon_step).astype(int)

    valid_bounds = (
        (row_idx >= 0)
        & (row_idx < n_lat)
        & (col_idx >= 0)
        & (col_idx < n_lon)
    )
    if not np.any(valid_bounds):
        return []

    valid_idx = valid_idx[valid_bounds]
    row_idx = row_idx[valid_bounds]
    col_idx = col_idx[valid_bounds]

    cell_ids = row_idx * n_lon + col_idx
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)

    cells: List[Dict[str, object]] = []
    for local_index, cell_id in enumerate(unique_cells):
        member_mask = inverse == local_index
        member_indices = valid_idx[member_mask]
        row = int(cell_id // n_lon)
        col = int(cell_id % n_lon)
        lat_center = lat_min + (row + 0.5) * lat_step
        lon_center = lon_min + (col + 0.5) * lon_step
        cells.append(
            {
                "cell_id": f"cell_r{row}_c{col}",
                "row": row,
                "col": col,
                "lat_center": float(lat_center),
                "lon_center": float(lon_center),
                "indices": member_indices,
            }
        )
    return cells


def extract_features(
    path: Path,
    features: Dict[str, FeatureSpec] = DEFAULT_FEATURES,
    *,
    region_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    bounding_box: Optional[LatLonBox] = None,
    grid_size: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Extract configured features from a single granule as a one-row DataFrame."""

    dataset = load_granule(path)
    rows: Dict[str, float] = {}
    bbox_mask: Optional[np.ndarray] = None
    normalised_grid = _normalise_grid_size(grid_size)
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None

    try:
        if bounding_box is not None:
            lat, lon = _extract_lat_lon(dataset)
            if lat is not None and lon is not None:
                bbox_mask = _bbox_mask(lat, lon, bounding_box)
        if normalised_grid is None:
            for name, spec in features.items():
                if spec.variable not in dataset:
                    rows[name] = float("nan")
                    continue
                data_array = dataset[spec.variable]
                values = getattr(data_array, "values", data_array)
                values = np.asarray(values, dtype=np.float32)
                masked = _apply_mask(values, bbox_mask)
                rows[name] = spec.reduce(masked)

        merged_metadata = _extract_granule_metadata(path.name)
        base_metadata: Dict[str, str] = {}
        if metadata:
            for key, value in metadata.items():
                base_metadata.setdefault(f"meta_{key}", value)
        for key, value in merged_metadata.items():
            base_metadata.setdefault(f"meta_{key}", value)

        if normalised_grid is None:
            result = {
                "granule": path.name,
                **rows,
                **base_metadata,
            }
            if region_id:
                result["region_id"] = region_id
            return pd.DataFrame([result])

        if bounding_box is None:
            raise ValueError("Grid-based extraction requires a bounding_box to be specified.")

        if lat is None or lon is None:
            lat, lon = _extract_lat_lon(dataset)
        if lat is None or lon is None:
            raise ValueError("Latitude/Longitude variables are required for grid extraction.")

        bbox_mask = bbox_mask if bbox_mask is not None else _bbox_mask(lat, lon, bounding_box)
        cells = _grid_cells(lat, lon, bbox_mask, bounding_box, normalised_grid)
        if not cells:
            return pd.DataFrame()

        arrays_cache: Dict[str, Optional[np.ndarray]] = {}
        for spec in features.values():
            if spec.variable not in dataset:
                arrays_cache[spec.variable] = None
                continue
            data_array = dataset[spec.variable]
            values = getattr(data_array, "values", data_array)
            arrays_cache[spec.variable] = np.asarray(values, dtype=np.float32).reshape(-1)

        records: List[Dict[str, object]] = []
        lat_step, lon_step = normalised_grid
        parent_region = region_id if region_id else None
        for cell in cells:
            cell_indices = cell["indices"]
            record: Dict[str, object] = {
                "granule": path.name,
                "region_id": f"{parent_region}_{cell['cell_id']}" if parent_region else cell["cell_id"],
                "grid_row": cell["row"],
                "grid_col": cell["col"],
                "cell_lat_center": cell["lat_center"],
                "cell_lon_center": cell["lon_center"],
                "grid_lat_step": lat_step,
                "grid_lon_step": lon_step,
                **base_metadata,
            }
            for name, spec in features.items():
                values_flat = arrays_cache.get(spec.variable)
                if values_flat is None:
                    record[name] = float("nan")
                    continue
                cell_values = values_flat[cell_indices]
                record[name] = spec.reduce(cell_values)
            records.append(record)

        return pd.DataFrame(records)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def export_features(
    granules: Iterable[Path],
    output_csv: Path,
    *,
    features: Dict[str, FeatureSpec] = DEFAULT_FEATURES,
    region_id: Optional[str] = None,
    bounding_box: Optional[LatLonBox] = None,
    grid_size: Optional[Sequence[float]] = None,
) -> None:
    """Process multiple granules and persist their features to CSV."""

    frames = [
        extract_features(
            path,
            features=features,
            region_id=region_id,
            metadata=None,
            bounding_box=bounding_box,
            grid_size=grid_size,
        )
        for path in granules
    ]
    if not frames:
        raise ValueError("No granules supplied for feature extraction.")
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
