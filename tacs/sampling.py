"""Deterministic trusted-data caps used to control background-class dominance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cap_background_indices(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    background_label: int,
    max_background_to_other_ratio: float | None,
    seed: int,
) -> np.ndarray:
    """Keep all non-background rows and a deterministic cap of background rows.

    A null ratio preserves every row.  For example, ratio=1.0 permits at most
    one background record per combined non-background record in the active
    trusted training set.  The cap is applied before both LoRA training and
    trusted-anchor construction, so class 0 cannot dominate either mechanism.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if max_background_to_other_ratio is None or len(indices) == 0:
        return indices.copy()
    ratio = float(max_background_to_other_ratio)
    if ratio <= 0:
        raise ValueError("max_background_to_other_ratio must be positive or null.")
    labels = metadata.iloc[indices]["label"].to_numpy(dtype=np.int64)
    background = indices[labels == int(background_label)]
    other = indices[labels != int(background_label)]
    if len(other) == 0:
        raise ValueError("Cannot cap a background-only trusted training set.")
    maximum = int(np.floor(len(other) * ratio))
    if len(background) <= maximum:
        return indices.copy()
    rng = np.random.default_rng(seed)
    retained_background = background.copy()
    rng.shuffle(retained_background)
    combined = np.concatenate([other, retained_background[:maximum]]).astype(np.int64)
    rng.shuffle(combined)
    return combined
