"""Daily aggregation helpers for MODIS cloud property features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXCLUDE_COLUMNS = {"granule"}
TIMESTAMP_CANDIDATES = ("meta_timestamp_utc", "meta_timestamp", "timestamp")
DATE_CANDIDATES = ("meta_date", "date")


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    timestamp_col = next((col for col in TIMESTAMP_CANDIDATES if col in frame.columns), None)
    if timestamp_col is not None:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        frame["date"] = frame[timestamp_col].dt.floor("D")
    else:
        date_col = next((col for col in DATE_CANDIDATES if col in frame.columns), None)
        if date_col is None:
            raise KeyError(
                "Feature table must contain a timestamp column (e.g., 'meta_timestamp_utc') "
                "or a date column."
            )
        frame["date"] = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["date"] = frame["date"].dt.tz_localize(None)
    if "meta_platform" not in frame.columns:
        frame["meta_platform"] = "Unknown"
    return frame


def _numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> Iterable[str]:
    excluded = set(EXCLUDE_COLUMNS).union(exclude)
    for column, dtype in df.dtypes.items():
        if column in excluded or column.startswith("meta_"):
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            yield column


def aggregate_daily(df: pd.DataFrame, *, combine_platforms: bool = True) -> pd.DataFrame:
    """Aggregate per-granule features into daily averages."""

    frame = _ensure_datetime_column(df)
    region_keys = [
        column
        for column in (
            "region_id",
            "grid_row",
            "grid_col",
            "cell_lat_center",
            "cell_lon_center",
            "grid_lat_step",
            "grid_lon_step",
        )
        if column in frame.columns
    ]

    group_keys = ["date", *region_keys]
    numeric_cols = list(_numeric_feature_columns(frame, exclude=group_keys))
    if not numeric_cols:
        raise ValueError("No numeric feature columns found to aggregate.")

    frame["date"] = frame["date"].dt.date
    grouped: pd.DataFrame
    if combine_platforms:
        grouped = (
            frame.groupby(group_keys, dropna=False)[numeric_cols]
            .mean()
            .reset_index()
        )
    else:
        pivot = frame.pivot_table(
            index=group_keys,
            columns="meta_platform",
            values=numeric_cols,
            aggfunc="mean",
        )
        pivot = pivot.sort_index(axis=1)
        pivot.columns = [
            f"{feature}_{platform.lower()}"
            for feature, platform in pivot.columns
        ]
        grouped = pivot.reset_index()

    grouped["date"] = pd.to_datetime(grouped["date"], format="%Y-%m-%d")
    grouped["day_of_year"] = grouped["date"].dt.dayofyear.astype(np.int16)
    grouped["doy_sin"] = np.sin(2 * np.pi * grouped["day_of_year"] / 365.25)
    grouped["doy_cos"] = np.cos(2 * np.pi * grouped["day_of_year"] / 365.25)
    grouped["date"] = grouped["date"].dt.date
    sort_columns = ["date"]
    if "region_id" in grouped.columns:
        sort_columns.append("region_id")
    return grouped.sort_values(sort_columns).reset_index(drop=True)


def export_daily_features(
    input_csv: Path,
    output_csv: Path,
    *,
    combine_platforms: bool = True,
) -> pd.DataFrame:
    """Read per-granule features, aggregate them daily, and write to CSV."""

    frame = pd.read_csv(input_csv)
    daily = aggregate_daily(frame, combine_platforms=combine_platforms)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)
    return daily


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to per-granule feature CSV")
    parser.add_argument("--output", type=Path, required=True, help="Destination for daily feature CSV")
    parser.add_argument(
        "--split-platforms",
        action="store_true",
        help="Keep Terra/Aqua statistics separate instead of averaging them together.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_daily_features(
        args.input,
        args.output,
        combine_platforms=not args.split_platforms,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
