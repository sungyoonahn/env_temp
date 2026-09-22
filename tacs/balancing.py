"""Class-imbalance controls for curriculum selection."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .reliability import choose_top_candidates


def select_curriculum_candidates(
    cache_indices: np.ndarray,
    scores: np.ndarray,
    accepted: np.ndarray,
    class_labels: np.ndarray,
    num_classes: int,
    selection_config: Mapping[str, object],
) -> np.ndarray:
    """Select high-reliability candidates without a majority-class takeover.

    First applies the configured global top-k/top-percentile budget.  When
    class balancing is on, candidates are then visited in global reliability
    order and accepted only up to a per-class cap.  This is deliberately a
    strict cap: unused majority-class capacity is not reassigned, because doing
    so would reintroduce the exact domination this control is meant to prevent.
    """
    cache_indices = np.asarray(cache_indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    accepted = np.asarray(accepted, dtype=bool)
    class_labels = np.asarray(class_labels, dtype=np.int64)
    if not (len(cache_indices) == len(scores) == len(accepted) == len(class_labels)):
        raise ValueError("Candidate indices, scores, acceptance mask, and labels must align.")
    excluded = np.asarray(selection_config.get("excluded_trembl_labels", []), dtype=np.int64)
    if len(excluded):
        if np.any((excluded < 0) | (excluded >= num_classes)):
            raise ValueError("An excluded TrEMBL label is outside the configured class range.")
        accepted &= ~np.isin(class_labels, excluded)
    if not bool(selection_config.get("balance_by_class", False)):
        return choose_top_candidates(cache_indices, scores, accepted, selection_config)

    eligible = np.flatnonzero(accepted & np.isfinite(scores))
    if len(eligible) == 0:
        return np.empty(0, dtype=np.int64)
    if selection_config["strategy"] == "top_percentile":
        percentile = selection_config.get("percentile_per_round")
        if percentile is None or not 0 < float(percentile) <= 100:
            raise ValueError("top_percentile requires percentile_per_round in (0, 100].")
        maximum = max(1, math.ceil(len(eligible) * float(percentile) / 100.0))
    else:
        configured_maximum = selection_config.get("max_trembl_per_round")
        maximum = len(eligible) if configured_maximum is None else min(len(eligible), int(configured_maximum))
    configured_maximum = selection_config.get("max_trembl_per_round")
    if configured_maximum is not None:
        maximum = min(maximum, int(configured_maximum))
    per_class_cap_raw = selection_config.get("max_trembl_per_class_per_round")
    selectable_classes = num_classes - len(np.unique(excluded))
    if selectable_classes < 1:
        raise ValueError("At least one TrEMBL class must remain selectable.")
    per_class_cap = int(per_class_cap_raw) if per_class_cap_raw is not None else math.ceil(maximum / selectable_classes)

    ordered = eligible[np.lexsort((cache_indices[eligible], -scores[eligible]))]
    class_counts = np.zeros(num_classes, dtype=np.int64)
    selected_positions: list[int] = []
    for position in ordered:
        label = int(class_labels[position])
        if label < 0 or label >= num_classes:
            raise ValueError("A selected candidate has an invalid class label.")
        if class_counts[label] >= per_class_cap:
            continue
        selected_positions.append(int(position))
        class_counts[label] += 1
        if len(selected_positions) == maximum:
            break
    return cache_indices[np.asarray(selected_positions, dtype=np.int64)]
