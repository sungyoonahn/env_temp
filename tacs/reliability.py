"""Reliability scoring and curriculum selection for weak TrEMBL labels."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .neighbors import NeighborResult, majority_vote


def label_factor(
    weak_labels: np.ndarray,
    model_predictions: np.ndarray,
    knn_predictions: np.ndarray,
) -> np.ndarray:
    """Agreement bonus from the TACS specification (1.0 / 0.7 / 0.0)."""
    triple = (weak_labels == model_predictions) & (model_predictions == knn_predictions)
    model_knn = model_predictions == knn_predictions
    return np.where(triple, 1.0, np.where(model_knn, 0.7, 0.0)).astype(np.float32)


def thresholds_for_round(selection_config: Mapping[str, Any], round_index: int) -> dict[str, Any]:
    """Apply an optional per-round curriculum entry to base selection settings."""
    resolved = copy.deepcopy(dict(selection_config))
    curriculum = selection_config.get("curriculum") or []
    if round_index <= 0 or not curriculum:
        return resolved
    entry = curriculum[min(round_index - 1, len(curriculum) - 1)]
    aliases = {
        "confidence": "confidence_threshold",
        "knn_agreement": "knn_agreement_threshold",
        "distance": "distance_threshold",
        "percentile": "percentile_per_round",
        "max_trembl": "max_trembl_per_round",
    }
    for key, value in entry.items():
        resolved[aliases.get(key, key)] = value
    return resolved


def acceptance_mask(
    confidence: np.ndarray,
    knn_agreement: np.ndarray,
    distance: np.ndarray,
    weak_labels: np.ndarray,
    model_predictions: np.ndarray,
    knn_predictions: np.ndarray,
    selection_config: Mapping[str, Any],
    ablation: str,
) -> np.ndarray:
    """Implement all requested ablation acceptance rules explicitly."""
    if ablation == "all_trembl":
        return np.ones(len(confidence), dtype=bool)
    if ablation in {"seed_only", "all_swiss"}:
        return np.zeros(len(confidence), dtype=bool)

    accepted = confidence >= float(selection_config["confidence_threshold"])
    if ablation == "confidence_only":
        return accepted

    accepted &= knn_agreement >= float(selection_config["knn_agreement_threshold"])
    accepted &= model_predictions == knn_predictions
    if ablation == "confidence_neighborhood":
        return accepted

    if ablation == "confidence_neighborhood_annotation":
        return accepted & (weak_labels == model_predictions)

    if ablation != "full_tacs":
        raise ValueError(f"Unknown TACS ablation: {ablation}")
    distance_threshold = selection_config.get("distance_threshold")
    if distance_threshold is not None:
        accepted &= distance <= float(distance_threshold)
    if not selection_config.get("require_model_knn_agreement", True):
        # Undoing a prior condition is clearer and safer than relying on an
        # alternative expression above.
        accepted = confidence >= float(selection_config["confidence_threshold"])
        accepted &= knn_agreement >= float(selection_config["knn_agreement_threshold"])
        if distance_threshold is not None:
            accepted &= distance <= float(distance_threshold)
    if selection_config.get("require_trembl_model_agreement", True):
        accepted &= weak_labels == model_predictions
    if selection_config.get("require_trembl_knn_agreement", False):
        accepted &= weak_labels == knn_predictions
    return accepted


def score_predictions(
    probabilities: np.ndarray,
    weak_labels: np.ndarray,
    neighbors: NeighborResult,
    num_classes: int,
    selection_config: Mapping[str, Any],
    ablation: str,
    distance_reduction: str,
) -> pd.DataFrame:
    """Score one batch after classifier probabilities and Swiss KNN are known."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    weak_labels = np.asarray(weak_labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(weak_labels):
        raise ValueError("Probabilities must be [N, C] and align with weak labels.")
    if probabilities.shape[1] != num_classes:
        raise ValueError("Probability class dimension differs from the Swiss label space.")
    if distance_reduction == "mean":
        distance = neighbors.distances.mean(axis=1)
    elif distance_reduction == "nearest":
        distance = neighbors.distances[:, 0]
    else:
        raise ValueError("distance_reduction must be mean or nearest.")
    model_prediction = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1).astype(np.float32)
    knn_prediction, knn_agreement = majority_vote(neighbors.labels, num_classes)
    factor = label_factor(weak_labels, model_prediction, knn_prediction)
    temperature = float(selection_config["distance_temperature"])
    if temperature <= 0:
        raise ValueError("selection.distance_temperature must be positive.")
    reliability = confidence * knn_agreement * np.exp(-distance / temperature) * factor
    accepted = acceptance_mask(
        confidence,
        knn_agreement,
        distance,
        weak_labels,
        model_prediction,
        knn_prediction,
        selection_config,
        ablation,
    )
    return pd.DataFrame(
        {
            "weak_label": weak_labels,
            "model_prediction": model_prediction,
            "confidence": confidence,
            "knn_prediction": knn_prediction,
            "knn_agreement": knn_agreement,
            "swiss_distance": distance.astype(np.float32),
            "model_trembl_agreement": model_prediction == weak_labels,
            "model_knn_agreement": model_prediction == knn_prediction,
            "trembl_knn_agreement": weak_labels == knn_prediction,
            "triple_agreement": (weak_labels == model_prediction) & (model_prediction == knn_prediction),
            "label_factor": factor,
            "reliability": reliability.astype(np.float32),
            "accepted": accepted,
        }
    )


def choose_top_candidates(
    cache_indices: np.ndarray,
    scores: np.ndarray,
    accepted: np.ndarray,
    selection_config: Mapping[str, Any],
) -> np.ndarray:
    """Choose a reproducible top-k or top-percentile set from accepted rows."""
    cache_indices = np.asarray(cache_indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    accepted = np.asarray(accepted, dtype=bool)
    if not (len(cache_indices) == len(scores) == len(accepted)):
        raise ValueError("cache_indices, scores, and accepted must have equal lengths.")
    valid_positions = np.flatnonzero(accepted & np.isfinite(scores))
    if len(valid_positions) == 0:
        return np.empty(0, dtype=np.int64)
    if selection_config["strategy"] == "top_percentile":
        percentile = selection_config.get("percentile_per_round")
        if percentile is None or not 0 < float(percentile) <= 100:
            raise ValueError("top_percentile requires selection.percentile_per_round in (0, 100].")
        take = max(1, math.ceil(len(valid_positions) * float(percentile) / 100.0))
    else:
        maximum = selection_config.get("max_trembl_per_round")
        take = len(valid_positions) if maximum is None else min(len(valid_positions), int(maximum))
    maximum = selection_config.get("max_trembl_per_round")
    if maximum is not None:
        take = min(take, int(maximum))
    # np.lexsort makes score ties deterministic and independent of cache shard
    # boundaries: primary key is -reliability, secondary is original cache id.
    ordering = np.lexsort((cache_indices[valid_positions], -scores[valid_positions]))
    return cache_indices[valid_positions[ordering[:take]]]
