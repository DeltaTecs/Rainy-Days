"""High-level training loop for the MODIS cloud seeding likelihood model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import CloudSeedingDataset, SplitConfig, split_dataset
from .model import ModelConfig, build_model


class StandardizeTransform:
    """Normalize features to zero mean / unit variance."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        std = torch.where(std == 0, torch.ones_like(std), std)
        self.mean = mean
        self.std = std

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std


def load_config(path: Path) -> Dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def prepare_dataset(config: Dict) -> Tuple[CloudSeedingDataset, torch.Tensor, torch.Tensor]:
    data_cfg = config["data"]
    dataset = CloudSeedingDataset(
        Path(data_cfg["features_path"]),
        feature_columns=data_cfg.get("feature_columns"),
        label_column=data_cfg.get("label_column"),
    )
    stats = dataset.frame[dataset.feature_columns].agg(["mean", "std"])
    mean = torch.tensor(stats.loc["mean"].to_numpy(dtype=float), dtype=torch.float32)
    std = torch.tensor(stats.loc["std"].to_numpy(dtype=float), dtype=torch.float32)
    if data_cfg.get("standardize", True):
        dataset.transform = StandardizeTransform(mean, std)
    return dataset, mean, std


def build_dataloaders(dataset: CloudSeedingDataset, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    split_cfg = SplitConfig(**config["data"].get("split", {}))
    train_ds, val_ds, test_ds = split_dataset(dataset, split_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def training_step(
    model: nn.Module,
    batch,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    features, labels = batch
    labels = enforce_labels(labels)
    if labels is None:
        return 0.0
    features = features.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(features).squeeze(-1)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            labels = enforce_labels(labels)
            if labels is None:
                continue
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    metrics: Dict[str, float] = {"loss": float(torch.tensor(losses).mean().item()) if losses else float("nan")}
    if all_outputs and all_labels:
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        metrics.update(compute_task_metrics(outputs, labels))
    return metrics


def compute_task_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if labels.ndim == 1 and labels.unique().numel() <= 10 and labels.dtype.is_floating_point:
        # Attempt to detect binary labels {0,1}
        unique_vals = labels.unique()
        if torch.all((unique_vals == 0) | (unique_vals == 1)):
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            accuracy = (preds == labels).float().mean().item()
            metrics["accuracy"] = float(accuracy)
            metrics["avg_probability"] = float(probs.mean().item())
    metrics["label_mean"] = float(labels.mean().item())
    metrics["prediction_mean"] = float(outputs.mean().item())
    return metrics


def enforce_labels(labels: torch.Tensor) -> torch.Tensor | None:
    if labels is None:
        raise ValueError("Labels are required for supervised training.")
    labels = labels.squeeze(-1)
    mask = ~torch.isnan(labels)
    if mask.sum() == 0:
        return None
    return labels[mask]


def main(config_path: Path) -> None:
    config = load_config(config_path)
    dataset, feature_mean, feature_std = prepare_dataset(config)
    if dataset.label_column is None:
        raise ValueError("Config must specify a label_column for supervised training.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(
        input_dim=len(dataset.feature_columns),
        hidden_dims=tuple(config["model"].get("hidden_dims", [128, 64])),
        dropout=config["model"].get("dropout", 0.1),
        output_dim=config["model"].get("output_dim", 1),
    )
    model = build_model(model_cfg, device=device)

    training_cfg = config["training"]
    task_type = training_cfg.get("task_type", "binary")
    if task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:  # pragma: no cover - config guard
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    optimizer = Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
    )

    train_loader, val_loader, test_loader = build_dataloaders(dataset, config)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = training_cfg.get("early_stopping_patience", 5)
    epochs_without_improvement = 0
    model_dir = Path(training_cfg.get("artifacts_dir", "artifacts"))
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(training_cfg.get("epochs", 20)):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            loss = training_step(model, batch, criterion, optimizer, device)
            if loss:
                epoch_losses.append(loss)
        avg_train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        val_metrics = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics.get("loss", float("nan")))
        if val_metrics.get("loss", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "feature_mean": feature_mean.numpy().tolist(),
                    "feature_std": feature_std.numpy().tolist(),
                },
                model_dir / "best_model.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    test_metrics = evaluate(model, test_loader, criterion, device)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


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
