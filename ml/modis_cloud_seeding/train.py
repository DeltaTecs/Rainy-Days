"""Training pipeline for the MODIS cloud seedability ensemble."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ResMLPConfig, build_resmlp


def _to_serialisable(value):
    """Recursively convert numpy types into JSON serialisable primitives."""

    if isinstance(value, dict):
        return {key: _to_serialisable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_serialisable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


@dataclass(frozen=True)
class FeatureStandardizer:
    """Hold feature normalization statistics and apply the transform."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_array(cls, array: np.ndarray) -> "FeatureStandardizer":
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def load_config(path: Path) -> Dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_feature_columns(frame: pd.DataFrame, label_column: str, split_column: str | None, extras: Iterable[str]) -> List[str]:
    exclude = {label_column, *(extras or [])}
    if split_column:
        exclude.add(split_column)
    return [col for col in frame.columns if col not in exclude and not col.startswith("meta_")]


def split_frame(
    frame: pd.DataFrame,
    *,
    split_column: str,
    split_values: Dict[str, Iterable[str]],
    granule_column: str | None = None,
) -> Dict[str, pd.DataFrame]:
    if split_column not in frame.columns:
        raise KeyError(f"Split column '{split_column}' not found in data.")
    result: Dict[str, pd.DataFrame] = {}
    for name, values in split_values.items():
        mask = frame[split_column].isin(list(values))
        result[name] = frame.loc[mask].copy()
    if granule_column and granule_column in frame.columns:
        split_map = {idx: split for split, data in result.items() for idx in data.index}
        for granule, rows in frame.groupby(granule_column):
            splits = {split_map.get(idx) for idx in rows.index if idx in split_map}
            splits.discard(None)
            if len(splits) > 1:
                raise ValueError(
                    f"Granule '{granule}' spans multiple splits: {sorted(s for s in splits if s)}."  # pragma: no cover - safety check
                )
    return result


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return weights.astype(np.float32)


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    config: Dict,
) -> LGBMClassifier:
    cfg = dict(config)
    early_rounds = cfg.pop("early_stopping_rounds", 50)
    random_state = cfg.pop("random_state", 2024)
    params = {
        "objective": "multiclass",
        "num_class": int(class_weights.size),
        "class_weight": {idx: float(weight) for idx, weight in enumerate(class_weights)},
        "random_state": random_state,
    }
    params.update(cfg)
    model = LGBMClassifier(**params)
    callbacks = []
    if early_rounds:
        callbacks.append(lgb.early_stopping(early_rounds, verbose=False))
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks or None,
    )
    return model


def _tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x_tensor = torch.from_numpy(features.astype(np.float32))
    y_tensor = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x_tensor, y_tensor)


def evaluate_resmlp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features, labels = (tensor.to(device) for tensor in batch)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * features.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    total = sum(len(chunk) for chunk in all_labels)
    avg_loss = total_loss / total if total else float("nan")
    probs = np.vstack(all_probs) if all_probs else np.empty((0, model.head.out_features))
    labels = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    predictions = np.argmax(probs, axis=1) if len(probs) else np.empty((0,), dtype=np.int64)
    seedable_mask = (labels > 0).astype(int)
    seedable_prob = probs[:, 1] + probs[:, 2] if len(probs) else np.empty((0,))
    if seedable_mask.size == 0 or np.unique(seedable_mask).size == 1:
        pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(seedable_mask, seedable_prob))
    accuracy = float(accuracy_score(labels, predictions)) if labels.size else float("nan")
    return {
        "val_loss": float(avg_loss),
        "val_accuracy": accuracy,
        "val_pr_auc_seedable": pr_auc,
    }


def train_resmlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    model_cfg: Dict,
    training_cfg: Dict,
    device: torch.device,
) -> Tuple[nn.Module, FeatureStandardizer, List[Dict[str, float]], Dict[str, float]]:
    standardizer = FeatureStandardizer.from_array(x_train)
    x_train_std = standardizer.transform(x_train).astype(np.float32)
    x_val_std = standardizer.transform(x_val).astype(np.float32)

    train_ds = _tensor_dataset(x_train_std, y_train)
    val_ds = _tensor_dataset(x_val_std, y_val)
    batch_size = int(training_cfg.get("batch_size", 512))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    resmlp_config = ResMLPConfig(
        input_dim=x_train.shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        num_classes=int(class_weights.size),
        expansion=int(model_cfg.get("expansion", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = build_resmlp(resmlp_config, device=device)

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-2)),
    )

    max_epochs = int(training_cfg.get("epochs", 200))
    patience = int(training_cfg.get("early_stopping_patience", 20))
    history: List[Dict[str, float]] = []
    best_state = None
    best_metric = float("-inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * features.size(0)
            total += features.size(0)
        train_loss = running_loss / total if total else float("nan")
        val_metrics = evaluate_resmlp(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        pr_auc = val_metrics.get("val_pr_auc_seedable", float("nan"))
        if np.isnan(pr_auc):
            metric = -float(val_metrics.get("val_loss", float("inf")))
        else:
            metric = pr_auc
        if metric > best_metric:
            best_metric = metric
            best_state = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": val_metrics,
                "epoch": epoch,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("ResMLP training failed to produce a valid checkpoint.")
    model.load_state_dict(best_state["state_dict"])
    best_metrics = dict(best_state["metrics"])
    best_metrics["epoch"] = best_state["epoch"]
    return model, standardizer, history, best_metrics


def predict_resmlp(
    model: nn.Module,
    features: np.ndarray,
    standardizer: FeatureStandardizer,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    if features.size == 0:
        return np.empty((0, model.head.out_features))
    model.eval()
    transformed = standardizer.transform(features).astype(np.float32)
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, transformed.shape[0], batch_size):
            batch = torch.from_numpy(transformed[start : start + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.vstack(outputs)


def fit_isotonic(seedable_probs: np.ndarray, seedable_labels: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(seedable_probs, seedable_labels)
    return calibrator


def apply_calibration(probs: np.ndarray, calibrator: IsotonicRegression) -> np.ndarray:
    if probs.size == 0:
        return probs
    seedable_mass = probs[:, 1] + probs[:, 2]
    calibrated_mass = calibrator.predict(seedable_mass)
    ratio = np.divide(
        probs[:, 1:3],
        seedable_mass[:, None],
        out=np.full_like(probs[:, 1:3], 0.5),
        where=seedable_mass[:, None] > 1e-8,
    )
    adjusted = probs.copy()
    adjusted[:, 1:3] = np.clip(calibrated_mass[:, None], 0.0, 1.0) * ratio
    adjusted[:, 0] = np.clip(1.0 - calibrated_mass, 0.0, 1.0)
    adjusted_sum = adjusted.sum(axis=1, keepdims=True)
    adjusted = np.divide(
        adjusted,
        adjusted_sum,
        out=np.full_like(adjusted, 1.0 / adjusted.shape[1]),
        where=adjusted_sum > 0,
    )
    return adjusted


def compute_split_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict:
    if labels.size == 0:
        return {}
    preds = probs.argmax(axis=1)
    seedable = (labels > 0).astype(int)
    seedable_prob = probs[:, 1] + probs[:, 2]
    metrics: Dict[str, object] = {}
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    try:
        metrics["log_loss"] = float(log_loss(labels, probs))
    except ValueError:
        eps = 1e-7
        clipped = np.clip(probs, eps, 1 - eps)
        clipped /= clipped.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(log_loss(labels, clipped))
    if seedable.sum() and seedable.sum() != seedable.size:
        metrics["seedable_average_precision"] = float(average_precision_score(seedable, seedable_prob))
    else:
        metrics["seedable_average_precision"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    metrics["classification_report"] = classification_report(labels, preds, output_dict=True, zero_division=0)
    return metrics


def save_predictions(
    frame: pd.DataFrame,
    probs: np.ndarray,
    output_path: Path,
    *,
    prefix: str,
) -> None:
    if probs.size == 0:
        return
    table = frame.copy()
    table[f"{prefix}_prob_not_seedable"] = probs[:, 0]
    table[f"{prefix}_prob_glaciogenic"] = probs[:, 1]
    table[f"{prefix}_prob_hygroscopic"] = probs[:, 2]
    table[f"{prefix}_demo_flag"] = (np.max(probs[:, 1:3], axis=1) >= 0.60).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)


def main(config_path: Path) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    frame = read_table(Path(data_cfg["features_path"]))
    label_column = data_cfg["label_column"]
    if label_column not in frame:
        raise KeyError(f"Label column '{label_column}' not found in data.")
    frame = frame.dropna(subset=[label_column]).copy()
    frame[label_column] = frame[label_column].astype(int)

    split_column = data_cfg.get("split_column", "split")
    split_values = data_cfg.get(
        "splits",
        {
            "train": ["train"],
            "val": ["val", "validation"],
            "test": ["test"],
        },
    )
    granule_column = data_cfg.get("granule_column")

    feature_columns = data_cfg.get("feature_columns")
    if not feature_columns:
        extras = [granule_column] if granule_column else []
        feature_columns = infer_feature_columns(frame, label_column, split_column, extras)
    missing_features = [col for col in feature_columns if col not in frame.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}.")

    splits = split_frame(
        frame,
        split_column=split_column,
        split_values=split_values,
        granule_column=granule_column,
    )
    required_splits = {"train", "val"}
    if not required_splits.issubset(splits) or any(splits[name].empty for name in required_splits):
        raise ValueError("Training requires non-empty 'train' and 'val' splits.")

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits.get("test", pd.DataFrame(columns=frame.columns))

    x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_df[label_column].to_numpy(dtype=np.int64)
    x_val = val_df[feature_columns].to_numpy(dtype=np.float32)
    y_val = val_df[label_column].to_numpy(dtype=np.int64)
    x_test = test_df[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_df[label_column].to_numpy(dtype=np.int64) if not test_df.empty else np.empty((0,), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("training", {}).get("seed", 2024))
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = int(config.get("training", {}).get("num_classes", 3))
    class_weights = compute_class_weights(y_train, num_classes)

    lightgbm_cfg = config.get("lightgbm", {})
    lightgbm_model = train_lightgbm(x_train, y_train, x_val, y_val, class_weights, lightgbm_cfg)

    resmlp_cfg = config.get("resmlp", {})
    resmlp_training_cfg = config.get("resmlp_training", {})
    resmlp_model, standardizer, history, val_metrics = train_resmlp(
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights,
        resmlp_cfg,
        resmlp_training_cfg,
        device,
    )

    lightgbm_train_probs = lightgbm_model.predict_proba(x_train)
    lightgbm_val_probs = lightgbm_model.predict_proba(x_val)
    lightgbm_test_probs = lightgbm_model.predict_proba(x_test) if x_test.size else np.empty((0, num_classes))

    resmlp_train_probs = predict_resmlp(resmlp_model, x_train, standardizer, device)
    resmlp_val_probs = predict_resmlp(resmlp_model, x_val, standardizer, device)
    resmlp_test_probs = predict_resmlp(resmlp_model, x_test, standardizer, device) if x_test.size else np.empty((0, num_classes))

    ensemble_train = 0.5 * (lightgbm_train_probs + resmlp_train_probs)
    ensemble_val = 0.5 * (lightgbm_val_probs + resmlp_val_probs)
    ensemble_test = 0.5 * (lightgbm_test_probs + resmlp_test_probs)

    seedable_val = (y_val > 0).astype(int)
    calibrator = fit_isotonic(ensemble_val[:, 1] + ensemble_val[:, 2], seedable_val)

    calibrated_train = apply_calibration(ensemble_train, calibrator)
    calibrated_val = apply_calibration(ensemble_val, calibrator)
    calibrated_test = apply_calibration(ensemble_test, calibrator)

    metrics = {
        "resmlp": {
            "validation": val_metrics,
            "history": history,
        },
        "lightgbm": {
            "best_iteration": getattr(lightgbm_model, "best_iteration_", None),
        },
        "ensemble": {
            "train": compute_split_metrics(y_train, calibrated_train),
            "val": compute_split_metrics(y_val, calibrated_val),
            "test": compute_split_metrics(y_test, calibrated_test) if y_test.size else {},
        },
        "feature_columns": feature_columns,
    }

    artifacts_dir = Path(config.get("training", {}).get("artifacts_dir", "ml_artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": resmlp_model.state_dict(),
            "config": resmlp_cfg,
            "training_config": resmlp_training_cfg,
            "standardizer": standardizer.to_dict(),
            "class_weights": class_weights.tolist(),
            "feature_columns": feature_columns,
        },
        artifacts_dir / "resmlp.pt",
    )
    joblib.dump(lightgbm_model, artifacts_dir / "lightgbm.joblib")
    joblib.dump(calibrator, artifacts_dir / "isotonic_calibrator.joblib")

    metrics_serialisable = _to_serialisable(metrics)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_serialisable, indent=2))

    save_predictions(train_df, calibrated_train, artifacts_dir / "predictions_train.csv", prefix="ensemble")
    save_predictions(val_df, calibrated_val, artifacts_dir / "predictions_val.csv", prefix="ensemble")
    if calibrated_test.size:
        save_predictions(test_df, calibrated_test, artifacts_dir / "predictions_test.csv", prefix="ensemble")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
