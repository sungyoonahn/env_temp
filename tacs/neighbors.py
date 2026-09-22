"""Swiss-Prot-only nearest-neighbour anchors for TACS reliability scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class NeighborResult:
    distances: np.ndarray
    labels: np.ndarray


class SwissAnchorIndex:
    """FAISS-first index with sklearn and NumPy fallbacks.

    The class deliberately has no ``add_trembl`` helper.  The caller must opt
    into the high-confidence TrEMBL ablation when constructing its anchor
    embeddings, which keeps the normal path contamination-free.
    """

    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be cosine or euclidean.")
        self.metric = metric
        self._backend = None
        self._index: Any = None
        self._embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    @staticmethod
    def _as_float32(values: Any) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(values, dtype=np.float32))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-12, None)

    def build(self, embeddings: Any, labels: Any) -> "SwissAnchorIndex":
        embeddings = self._as_float32(embeddings)
        labels = np.asarray(labels, dtype=np.int64)
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("Swiss anchor embeddings must be a non-empty [N, D] matrix.")
        if len(embeddings) != len(labels):
            raise ValueError("Swiss anchor embeddings and labels have different row counts.")
        prepared = self._normalize(embeddings) if self.metric == "cosine" else embeddings
        self._labels = labels
        self._embeddings = prepared

        try:
            import faiss

            if self.metric == "cosine":
                index = faiss.IndexFlatIP(prepared.shape[1])
            else:
                index = faiss.IndexFlatL2(prepared.shape[1])
            index.add(prepared)
            self._index = index
            self._backend = "faiss"
            return self
        except ImportError:
            pass

        try:
            from sklearn.neighbors import NearestNeighbors

            index = NearestNeighbors(metric="cosine" if self.metric == "cosine" else "euclidean", algorithm="auto")
            index.fit(prepared)
            self._index = index
            self._backend = "sklearn"
            return self
        except ImportError:
            self._index = None
            self._backend = "numpy"
            return self

    def query(self, embeddings: Any, k: int) -> NeighborResult:
        if self._labels is None or self._embeddings is None or self._backend is None:
            raise RuntimeError("Build the Swiss anchor index before querying it.")
        if k < 1:
            raise ValueError("k must be at least one.")
        queries = self._as_float32(embeddings)
        if queries.ndim != 2 or queries.shape[1] != self._embeddings.shape[1]:
            raise ValueError("Query embeddings have the wrong shape for this anchor index.")
        k = min(int(k), len(self._labels))
        prepared = self._normalize(queries) if self.metric == "cosine" else queries
        if self._backend == "faiss":
            values, indices = self._index.search(prepared, k)
            distances = 1.0 - values if self.metric == "cosine" else np.sqrt(np.maximum(values, 0.0))
        elif self._backend == "sklearn":
            distances, indices = self._index.kneighbors(prepared, n_neighbors=k, return_distance=True)
        else:
            if self.metric == "cosine":
                distances_full = 1.0 - prepared @ self._embeddings.T
            else:
                distances_full = np.linalg.norm(prepared[:, None, :] - self._embeddings[None, :, :], axis=-1)
            indices = np.argpartition(distances_full, kth=k - 1, axis=1)[:, :k]
            row = np.arange(len(queries))[:, None]
            ordering = np.argsort(distances_full[row, indices], axis=1)
            indices = indices[row, ordering]
            distances = distances_full[row, indices]
        return NeighborResult(
            distances=np.asarray(distances, dtype=np.float32),
            labels=self._labels[np.asarray(indices, dtype=np.int64)],
        )


def majority_vote(neighbor_labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return KNN class and majority fraction for each query."""
    neighbor_labels = np.asarray(neighbor_labels, dtype=np.int64)
    if neighbor_labels.ndim != 2 or neighbor_labels.shape[1] == 0:
        raise ValueError("neighbor_labels must be a non-empty [N, K] array.")
    if neighbor_labels.min() < 0 or neighbor_labels.max() >= num_classes:
        raise ValueError("Neighbor labels are outside the configured class range.")
    counts = np.zeros((neighbor_labels.shape[0], num_classes), dtype=np.int32)
    rows = np.repeat(np.arange(neighbor_labels.shape[0]), neighbor_labels.shape[1])
    np.add.at(counts, (rows, neighbor_labels.ravel()), 1)
    prediction = counts.argmax(axis=1).astype(np.int64)
    agreement = counts.max(axis=1).astype(np.float32) / neighbor_labels.shape[1]
    return prediction, agreement


def build_swiss_anchor(
    embeddings: Any,
    labels: Any,
    anchor_config: Mapping[str, Any],
) -> SwissAnchorIndex:
    """Construct an index from explicitly supplied trusted anchor rows."""
    return SwissAnchorIndex(metric=str(anchor_config["metric"])).build(embeddings, labels)
