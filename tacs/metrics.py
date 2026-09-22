"""Evaluation uncertainty estimates for held-out Swiss-Prot predictions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    samples: int,
    confidence_level: float,
    seed: int,
    metric_fn: Callable[[np.ndarray, np.ndarray, int], dict[str, Any]],
) -> dict[str, Any]:
    """Return deterministic percentile bootstrap intervals for classifier metrics."""
    if samples <= 0:
        return {}
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(labels) == 0:
        return {}

    rng = np.random.default_rng(seed)
    accuracy = np.empty(samples, dtype=np.float64)
    macro_f1 = np.empty(samples, dtype=np.float64)
    precision = np.empty((samples, num_classes), dtype=np.float64)
    recall = np.empty((samples, num_classes), dtype=np.float64)
    f1 = np.empty((samples, num_classes), dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        metrics = metric_fn(labels[indices], predictions[indices], num_classes)
        accuracy[sample_index] = metrics["accuracy"]
        macro_f1[sample_index] = metrics["macro_f1"]
        precision[sample_index] = metrics["per_class_precision"]
        recall[sample_index] = metrics["per_class_recall"]
        f1[sample_index] = metrics["per_class_f1"]

    lower = (1.0 - confidence_level) / 2.0
    upper = 1.0 - lower

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, lower)), float(np.quantile(values, upper))]

    return {
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(samples),
        "accuracy_ci": interval(accuracy),
        "macro_f1_ci": interval(macro_f1),
        "per_class_precision_ci": [interval(precision[:, index]) for index in range(num_classes)],
        "per_class_recall_ci": [interval(recall[:, index]) for index in range(num_classes)],
        "per_class_f1_ci": [interval(f1[:, index]) for index in range(num_classes)],
    }
