"""Deterministic full-background coverage and hard-negative epoch sampling."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def rotating_background_epoch(
    frame: pd.DataFrame,
    epoch: int,
    sampling_config: Mapping[str, Any],
    seed: int,
    hardness_sum: np.ndarray,
    hardness_count: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one epoch with complete background coverage and balanced positives.

    During the coverage phase, each background shard is mandatory exactly once.
    Later epochs mix the highest mean ``1 - P(background)`` examples with a
    deterministic random refresh. Every non-background record is retained at
    least once, and minority non-background classes are oversampled to equal
    counts before the configured background ratio is applied.
    """
    if epoch < 1:
        raise ValueError("epoch must be positive.")
    required = {"label", "_row_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Rotating background frame is missing: {sorted(missing)}")

    background_label = int(sampling_config["background_label"])
    coverage_epochs = int(sampling_config["coverage_epochs"])
    ratio = float(sampling_config["background_to_non_background_ratio"])
    hard_fraction = float(sampling_config["hard_negative_fraction"])
    labels = frame["label"].to_numpy(dtype=np.int64)
    background = np.flatnonzero(labels == background_label).astype(np.int64)
    non_background = np.flatnonzero(labels != background_label).astype(np.int64)
    if not len(background) or not len(non_background):
        raise ValueError("Rotating background sampling requires both background and non-background rows.")
    if len(hardness_sum) != len(frame) or len(hardness_count) != len(frame):
        raise ValueError("Hardness arrays must align with the full training frame.")

    order_rng = np.random.default_rng(seed + 61_003)
    background_order = background.copy()
    order_rng.shuffle(background_order)
    shards = np.array_split(background_order, coverage_epochs)
    coverage_phase = epoch <= coverage_epochs
    mandatory = shards[epoch - 1] if coverage_phase else np.empty(0, dtype=np.int64)
    average_shard_size = math.ceil(len(background) / coverage_epochs)

    class_groups = [
        non_background[labels[non_background] == label]
        for label in sorted(np.unique(labels[non_background]).tolist())
    ]
    target_reference = max(len(mandatory), average_shard_size)
    target_per_non_background_class = max(
        max(len(group) for group in class_groups),
        math.ceil(target_reference / (ratio * len(class_groups))),
    )
    epoch_rng = np.random.default_rng(seed + 71_003 + epoch)
    selected_non_background: list[np.ndarray] = []
    for group in class_groups:
        repeats = target_per_non_background_class - len(group)
        extra = (
            epoch_rng.choice(group, size=repeats, replace=True).astype(np.int64)
            if repeats > 0
            else np.empty(0, dtype=np.int64)
        )
        selected_non_background.append(np.concatenate([group, extra]))
    non_background_draws = np.concatenate(selected_non_background).astype(np.int64)

    target_background = min(
        len(background),
        max(len(mandatory), math.ceil(ratio * len(non_background_draws))),
    )
    if coverage_phase:
        mandatory_set = set(mandatory.tolist())
        available = np.asarray(
            [index for index in background if int(index) not in mandatory_set], dtype=np.int64
        )
        extra_count = target_background - len(mandatory)
        extra = (
            epoch_rng.choice(available, size=extra_count, replace=False).astype(np.int64)
            if extra_count > 0
            else np.empty(0, dtype=np.int64)
        )
        background_draws = np.concatenate([mandatory, extra]).astype(np.int64)
        hard_count = 0
    else:
        observed = background[hardness_count[background] > 0]
        means = np.divide(
            hardness_sum[observed],
            hardness_count[observed],
            out=np.zeros(len(observed), dtype=np.float64),
            where=hardness_count[observed] > 0,
        )
        ranked = observed[np.lexsort((observed, -means))]
        requested_hard = min(len(ranked), int(round(target_background * hard_fraction)))
        hard = ranked[:requested_hard]
        hard_set = set(hard.tolist())
        random_pool = np.asarray(
            [index for index in background if int(index) not in hard_set], dtype=np.int64
        )
        random_count = target_background - len(hard)
        random_draws = (
            epoch_rng.choice(random_pool, size=random_count, replace=False).astype(np.int64)
            if random_count > 0
            else np.empty(0, dtype=np.int64)
        )
        background_draws = np.concatenate([hard, random_draws]).astype(np.int64)
        hard_count = len(hard)

    epoch_indices = np.concatenate([background_draws, non_background_draws]).astype(np.int64)
    epoch_rng.shuffle(epoch_indices)
    epoch_frame = frame.iloc[epoch_indices].copy().reset_index(drop=True)
    metadata = {
        "background_sampling_phase": "coverage" if coverage_phase else "hard_negative",
        "background_unique": int(len(np.unique(background_draws))),
        "background_draws": int(len(background_draws)),
        "background_mandatory_shard": int(len(mandatory)),
        "background_hard_draws": int(hard_count),
        "non_background_draws": int(len(non_background_draws)),
        "non_background_classes": int(len(class_groups)),
        "target_per_non_background_class": int(target_per_non_background_class),
        "effective_background_ratio": float(len(background_draws) / len(non_background_draws)),
    }
    return epoch_frame, metadata
