"""Forecast daily cloud properties and derive seedability scores."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from . import preprocess
from .seedability import append_seedability


@dataclass(frozen=True)
class DatasetSplit:
    features: pd.DataFrame
    targets: pd.DataFrame
    meta: pd.DataFrame


@dataclass(frozen=True)
class ForecastSettings:
    target_columns: Sequence[str]
    lags: Sequence[int]
    horizons: Sequence[int]
    train_fraction: float
    val_fraction: float
    artifacts_dir: Path

    @property
    def test_fraction(self) -> float:
        return max(0.0, 1.0 - self.train_fraction - self.val_fraction)


def _lagged_feature_frame(df: pd.DataFrame, target_columns: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    frame = df.copy()
    if "date" not in frame.columns:
        raise KeyError("Daily feature table must contain a 'date' column.")
    frame["date"] = pd.to_datetime(frame["date"], utc=False, errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["date"] = frame["date"].dt.tz_localize(None)

    group_keys = [col for col in ("region_id",) if col in frame.columns]
    sort_keys = group_keys + ["date"]
    frame = frame.sort_values(sort_keys).reset_index(drop=True)

    for lag in sorted(set(lags)):
        if group_keys:
            shifted = frame.groupby(group_keys, sort=False)[list(target_columns)].shift(lag)
        else:
            shifted = frame[list(target_columns)].shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in target_columns]
        frame = pd.concat([frame, shifted], axis=1)
    return frame


def prepare_supervised_dataset(
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    lags: Sequence[int],
    horizon: int,
) -> DatasetSplit:
    frame = _lagged_feature_frame(df, target_columns, lags)
    group_keys = [col for col in ("region_id",) if col in frame.columns]

    if group_keys:
        future = frame.groupby(group_keys, sort=False)[list(target_columns)].shift(-horizon)
    else:
        future = frame[list(target_columns)].shift(-horizon)
    future.columns = [f"__target_{col}" for col in target_columns]
    frame = pd.concat([frame, future], axis=1)

    meta_columns = [col for col in ("date", "region_id", "grid_row", "grid_col", "cell_lat_center", "cell_lon_center", "grid_lat_step", "grid_lon_step") if col in frame.columns]
    target_internal = [f"__target_{col}" for col in target_columns]
    feature_columns = [
        col
        for col in frame.columns
        if col not in target_internal and col not in {"region_id", "date"}
    ]
    feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(frame[col])]

    required_columns = feature_columns + target_internal
    dataset = frame.dropna(subset=required_columns).copy()
    features = dataset[feature_columns].reset_index(drop=True)
    labels = dataset[target_internal].rename(columns=lambda name: name.replace("__target_", "")).reset_index(drop=True)
    meta = dataset[[col for col in meta_columns if col in dataset.columns]].reset_index(drop=True)
    return DatasetSplit(features=features, targets=labels, meta=meta)


def split_time_series(split: DatasetSplit, *, train_fraction: float, val_fraction: float) -> dict[str, DatasetSplit]:
    if "date" not in split.meta.columns:
        raise KeyError("Metadata must include a 'date' column for time-based splitting.")

    unique_dates = split.meta["date"].drop_duplicates().sort_values()
    n_dates = unique_dates.size
    if n_dates < 3:
        raise ValueError("Need at least 3 unique dates to split into train/val/test.")

    train_count = max(int(round(n_dates * train_fraction)), 1)
    val_count = max(int(round(n_dates * val_fraction)), 1)
    if train_count + val_count >= n_dates:
        val_count = max(1, n_dates - train_count - 1)
    if train_count + val_count >= n_dates:
        raise ValueError("Train/val fractions leave no room for test split.")

    train_end_date = unique_dates.iloc[train_count - 1]
    val_end_date = unique_dates.iloc[train_count + val_count - 1]

    dates = split.meta["date"].reset_index(drop=True)
    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date) & (dates <= val_end_date)
    test_mask = dates > val_end_date

    def _subset(mask: pd.Series) -> DatasetSplit:
        return DatasetSplit(
            features=split.features.loc[mask].reset_index(drop=True),
            targets=split.targets.loc[mask].reset_index(drop=True),
            meta=split.meta.loc[mask].reset_index(drop=True),
        )

    return {
        "train": _subset(train_mask),
        "val": _subset(val_mask),
        "test": _subset(test_mask),
    }


def _fit_multioutput_lightgbm(train: DatasetSplit) -> MultiOutputRegressor:
    base = LGBMRegressor(
        random_state=2024,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
    )
    model = MultiOutputRegressor(base)
    model.fit(
        train.features.to_numpy(dtype=np.float32),
        train.targets.to_numpy(dtype=np.float32),
    )
    return model


def _evaluate(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    if y_true.empty:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for idx, column in enumerate(y_true.columns):
        true_vals = y_true.iloc[:, idx].to_numpy(dtype=np.float32)
        pred_vals = y_pred[:, idx].astype(np.float32)
        metrics[column] = {
            "mae": float(mean_absolute_error(true_vals, pred_vals)),
            "rmse": float(mean_squared_error(true_vals, pred_vals, squared=False)),
        }
    metrics["overall_mae"] = float(mean_absolute_error(y_true.values.flatten(), y_pred.flatten()))
    metrics["overall_rmse"] = float(
        mean_squared_error(y_true.values.flatten(), y_pred.flatten(), squared=False)
    )
    return metrics


def _prediction_frame(
    model: MultiOutputRegressor,
    feature_frame: pd.DataFrame,
    *,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    predictions = model.predict(feature_frame.to_numpy(dtype=np.float32))
    return pd.DataFrame(predictions, columns=target_columns, index=feature_frame.index)


def _historical_predictions(
    model: MultiOutputRegressor,
    split: DatasetSplit,
) -> pd.DataFrame:
    if split.features.empty:
        return pd.DataFrame()
    preds = model.predict(split.features.to_numpy(dtype=np.float32))
    history = split.meta.copy()
    for idx, column in enumerate(split.targets.columns):
        history[f"actual_{column}"] = split.targets.iloc[:, idx].to_numpy(dtype=np.float32)
        history[f"pred_{column}"] = preds[:, idx].astype(np.float32)
    return history


def _latest_feature_rows(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    lags: Sequence[int],
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _lagged_feature_frame(df, target_columns, lags)
    meta_cols = [col for col in ("date", "region_id", "grid_row", "grid_col", "cell_lat_center", "cell_lon_center", "grid_lat_step", "grid_lon_step") if col in frame.columns]
    missing_features = [col for col in feature_columns if col not in frame.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns for inference: {missing_features}")

    subset = frame[meta_cols + list(feature_columns)].dropna(subset=feature_columns)
    if subset.empty:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame(columns=meta_cols)

    if "region_id" in subset.columns:
        subset = subset.sort_values(["region_id", "date"])
        latest = subset.groupby("region_id", sort=False).tail(1)
    else:
        latest = subset.sort_values("date").tail(1)

    latest_features = latest[feature_columns].copy().reset_index(drop=True)
    latest_meta = latest[meta_cols].copy().reset_index(drop=True)
    return latest_features, latest_meta


def train_and_forecast(
    daily_features: pd.DataFrame,
    settings: ForecastSettings,
    *,
    output_path: Path,
) -> pd.DataFrame:
    records: list[dict] = []
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for horizon in settings.horizons:
        supervised = prepare_supervised_dataset(
            daily_features,
            target_columns=settings.target_columns,
            lags=settings.lags,
            horizon=horizon,
        )
        splits = split_time_series(
            supervised,
            train_fraction=settings.train_fraction,
            val_fraction=settings.val_fraction,
        )
        model = _fit_multioutput_lightgbm(splits["train"])

        horizon_dir = settings.artifacts_dir / f"forecast_h{horizon}d"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, horizon_dir / "model.joblib")

        metrics = {}
        feature_columns = splits["train"].features.columns.tolist()

        for split_name, split_data in splits.items():
            history = _historical_predictions(model, split_data)
            if not history.empty:
                metrics[split_name] = _evaluate(
                    split_data.targets,
                    history[[f"pred_{col}" for col in split_data.targets.columns]].to_numpy(dtype=np.float32),
                )
                history = history.copy()
                history["target_date"] = pd.to_datetime(history["date"]) + pd.to_timedelta(horizon, unit="D")
                history["horizon_days"] = horizon
                history_path = horizon_dir / f"{split_name}_predictions.csv"
                history.to_csv(history_path, index=False)
            else:
                metrics[split_name] = {}

        (horizon_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        latest_features, latest_meta = _latest_feature_rows(
            daily_features,
            settings.target_columns,
            settings.lags,
            feature_columns,
        )
        if latest_features.empty:
            continue
        latest_predictions = model.predict(latest_features.to_numpy(dtype=np.float32))
        for row_idx in range(len(latest_features)):
            meta_row = latest_meta.iloc[row_idx]
            base_timestamp = pd.to_datetime(meta_row.get("date"))
            if pd.isna(base_timestamp):
                continue
            forecast_timestamp = base_timestamp + pd.to_timedelta(horizon, unit="D")
            record = {
                "base_date": base_timestamp.date().isoformat(),
                "forecast_date": forecast_timestamp.date().isoformat(),
                "horizon_days": horizon,
            }
            for meta_column in latest_meta.columns:
                if meta_column == "date":
                    continue
                value = meta_row[meta_column]
                if isinstance(value, np.generic):
                    value = value.item()
                record[meta_column] = value
            for idx, column in enumerate(settings.target_columns):
                record[column] = float(latest_predictions[row_idx][idx])
            records.append(record)

    forecast_table = pd.DataFrame(records).sort_values("horizon_days").reset_index(drop=True)
    forecast_with_seedability = append_seedability(forecast_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_with_seedability.to_csv(output_path, index=False)
    return forecast_with_seedability


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-features",
        type=Path,
        required=True,
        help="CSV produced by daily aggregation step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/daily_predictions.csv"),
        help="Destination CSV for forecasted properties and seedability score.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("ml_artifacts/forecasts"),
        help="Directory to store trained models and metrics.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 7, 14],
        help="Lag windows (in days) to include as features.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 7],
        help="Forecast horizons (in days) to train individual models for.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of samples allocated to the training split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of samples allocated to the validation split.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=list(preprocess.DEFAULT_FEATURES.keys()),
        help="Target columns to forecast. Defaults to processed cloud property statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    daily_features = pd.read_csv(args.daily_features)
    settings = ForecastSettings(
        target_columns=args.targets,
        lags=args.lags,
        horizons=args.horizons,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        artifacts_dir=args.artifacts_dir,
    )
    result = train_and_forecast(daily_features, settings, output_path=args.output)
    print(result.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
