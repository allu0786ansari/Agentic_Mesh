from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from edge.data_loader import HAIWindowDataset
from edge.models.vae import VAE


def load_checkpoint_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[VAE, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VAE(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_module.") for key in state_dict):
        state_dict = {key.removeprefix("_module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def reconstruction_errors(
    model: VAE,
    dataset: HAIWindowDataset,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, inputs in enumerate(loader):
            if not torch.isfinite(inputs).all():
                raise ValueError(f"Non-finite input values in evaluation batch {batch_index}")
            inputs = inputs.to(device)
            mu, _ = model.encode(inputs)
            reconstruction = model.decode(mu)
            if not torch.isfinite(reconstruction).all():
                raise ValueError(f"Non-finite reconstruction values in evaluation batch {batch_index}")
            batch_errors = torch.mean((reconstruction - inputs).pow(2), dim=1)
            if not torch.isfinite(batch_errors).all():
                raise ValueError(f"Non-finite reconstruction errors in evaluation batch {batch_index}")
            errors.append(batch_errors.cpu().numpy())
    result = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Evaluation produced non-finite reconstruction errors")
    return result


def error_distribution(errors: np.ndarray) -> dict[str, float | int]:
    if errors.size == 0:
        raise ValueError("Cannot summarize an empty error array")
    percentiles = np.percentile(errors, [1, 5, 25, 50, 75, 95, 99, 99.9])
    return {
        "count": int(errors.size),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "p999": float(percentiles[7]),
    }


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if scores.size != labels.size:
        raise ValueError(f"Score/label length mismatch: {scores.size} scores, {labels.size} labels")
    if not np.isfinite(scores).all():
        raise ValueError("Cannot calculate metrics from non-finite reconstruction scores")
    if not np.isfinite(threshold):
        raise ValueError("Validation threshold is non-finite")
    labels = labels.astype(np.int8)
    predictions = (scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(false_positive / max(true_negative + false_positive, 1)),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "positive_windows": int(np.sum(labels == 1)),
        "negative_windows": int(np.sum(labels == 0)),
    }
    return result


def bootstrap_confidence_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    seed: int = 42,
    samples: int = 200,
) -> dict[str, dict[str, float]]:
    """Estimate percentile bootstrap 95% CIs for classification metrics."""
    if samples < 2:
        raise ValueError("bootstrap samples must be at least 2")
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "f1_score": [], "recall": []}
    for _ in range(samples):
        indices = generator.integers(0, len(scores), len(scores))
        if np.unique(labels[indices]).size < 2:
            continue
        metric = evaluate_scores(scores[indices], labels[indices], threshold=threshold)
        for name in values:
            values[name].append(float(metric[name]))
    return {
        name: {
            "lower_2_5pct": float(np.percentile(metric_values, 2.5)),
            "upper_97_5pct": float(np.percentile(metric_values, 97.5)),
            "bootstrap_samples": len(metric_values),
        }
        for name, metric_values in values.items()
        if metric_values
    }


def event_level_metrics(
    raw_labels: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Score detection over contiguous source-row attack episodes."""
    predicted_rows = np.zeros(raw_labels.size, dtype=bool)
    for start in window_starts[scores >= threshold]:
        predicted_rows[int(start) : int(start) + window_size] = True
    attack_rows = raw_labels.astype(bool)
    events = []
    index = 0
    while index < len(raw_labels):
        if attack_rows[index]:
            end = index
            while end + 1 < len(raw_labels) and attack_rows[end + 1]:
                end += 1
            events.append((index, end + 1))
            index = end + 1
        else:
            index += 1
    detected_events = sum(bool(predicted_rows[start:end].any()) for start, end in events)
    normal_rows = ~attack_rows
    false_positive_rows = int(np.sum(predicted_rows & normal_rows))
    return {
        "attack_event_count": len(events),
        "detected_attack_events": detected_events,
        "event_recall": float(detected_events / max(len(events), 1)),
        "normal_row_count": int(np.sum(normal_rows)),
        "false_positive_row_rate": float(false_positive_rows / max(int(np.sum(normal_rows)), 1)),
    }


def fixed_fpr_operating_points(
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    target_fprs: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, dict[str, Any]]:
    """Evaluate thresholds chosen from normal validation scores at target FPRs."""
    if not np.isfinite(validation_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Fixed-FPR analysis requires finite reconstruction scores")
    return {
        f"fpr_{int(target_fpr * 100)}pct": evaluate_scores(
            test_scores,
            labels,
            threshold=float(np.quantile(validation_scores, 1.0 - target_fpr)),
        )
        for target_fpr in target_fprs
    }


def plot_evaluation_curves(
    validation_errors: np.ndarray,
    test_errors: np.ndarray,
    labels: np.ndarray,
    destination: str | Path,
) -> None:
    """Write distribution, ROC, and PR plots for one node evaluation."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, test_errors)
    precision, recall, _ = precision_recall_curve(labels, test_errors)
    normal_test = test_errors[labels == 0]
    attack_test = test_errors[labels == 1]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(normal_test, bins=80, alpha=0.7, density=True, label="normal")
    axes[0].hist(attack_test, bins=80, alpha=0.7, density=True, label="attack")
    axes[0].axvline(np.quantile(validation_errors, 0.95), color="black", linestyle="--", label="95% threshold")
    axes[0].set_title("Reconstruction errors")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].plot(false_positive_rate, true_positive_rate)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].set_title(f"ROC-AUC {roc_auc_score(labels, test_errors):.3f}")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[2].plot(recall, precision)
    axes[2].set_title(f"PR-AUC {average_precision_score(labels, test_errors):.3f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def evaluate_node(
    checkpoint_path: str | Path,
    train_partition: str | Path,
    test_partition: str | Path,
    *,
    window_size: int = 60,
    stride: int = 10,
    batch_size: int = 256,
    validation_quantile: float = 0.95,
    device: str = "cpu",
    plot_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    validation_dataset = HAIWindowDataset(
        train_partition,
        window_size=window_size,
        stride=stride,
        normal_only=True,
        start_fraction=0.80,
        end_fraction=1.0,
        scaler_path=scaler_path,
    )
    test_dataset = HAIWindowDataset(
        test_partition, window_size=window_size, stride=stride, normal_only=False, scaler_path=scaler_path
    )
    validation_errors = reconstruction_errors(model, validation_dataset, batch_size=batch_size, device=device)
    test_errors = reconstruction_errors(model, test_dataset, batch_size=batch_size, device=device)
    threshold = float(np.quantile(validation_errors, validation_quantile))
    labels = test_dataset.window_labels.astype(np.int8)
    if len(labels) != len(test_errors):
        raise ValueError("Test labels and reconstruction scores have different lengths")
    metrics = evaluate_scores(test_errors, labels, threshold=threshold)
    if plot_path is not None:
        plot_evaluation_curves(validation_errors, test_errors, labels, plot_path)
    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "epsilon": checkpoint.get("epsilon"),
            "delta": checkpoint.get("delta"),
            "input_dim": checkpoint.get("input_dim"),
        },
        "validation": {
            "source": str(train_partition),
            "scaler": str(scaler_path) if scaler_path is not None else None,
            "normal_only": True,
            "selection": "last 20% of normal training windows by temporal order",
            "total_normal_windows": int(len(validation_dataset) / 0.20),
            "threshold_windows": int(validation_errors.size),
            "threshold_quantile": validation_quantile,
            "threshold": threshold,
            "reconstruction_errors": error_distribution(validation_errors),
        },
        "test": {
            "source": str(test_partition),
            "reconstruction_errors": error_distribution(test_errors),
            "metrics": metrics,
            "bootstrap_confidence_intervals": bootstrap_confidence_intervals(
                test_errors, labels, threshold, seed=seed, samples=bootstrap_samples
            ),
            "event_level_metrics": event_level_metrics(
                test_dataset.labels,
                test_dataset.window_starts,
                test_dataset.window_size,
                test_errors,
                threshold,
            ),
            "fixed_fpr_operating_points": fixed_fpr_operating_points(
                validation_errors, test_errors, labels
            ),
        },
    }


def write_result(result: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")



