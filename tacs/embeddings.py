"""Frozen ESM-2 mean-pooled embeddings and reusable sharded caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("TACS embedding extraction requires PyTorch.") from exc
    return torch


def _torch_load(path: Path):
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatible with older Torch installations.
        return torch.load(path, map_location="cpu")


class ESM2MeanPoolEncoder:
    """Frozen ESM-2 encoder that excludes padding and special tokens."""

    def __init__(self, model_config: Mapping[str, Any]):
        if model_config["encoder_mode"] != "frozen":
            raise ValueError("Cached TACS embeddings require model.encoder_mode='frozen'.")
        self.model_name = str(model_config["esm_model_name"])
        self.max_length = int(model_config["max_length"])
        self.batch_size = int(model_config["embedding_batch_size"])
        self.requested_device = str(model_config.get("device", "cuda"))
        self._model = None
        self._tokenizer = None
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._load()
        return self._device

    def _load(self) -> None:
        torch = _torch()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("TACS ESM-2 extraction requires transformers.") from exc
        device = self.requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA was requested but is unavailable; falling back to CPU for ESM-2 extraction.")
            device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval().to(device)
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._device = torch.device(device)

    def encode(self, sequences: Sequence[str]):
        """Return float32 [N, D] protein embeddings in input order."""
        torch = _torch()
        if not sequences:
            raise ValueError("Cannot encode an empty sequence collection.")
        if self._model is None:
            self._load()
        assert self._model is not None and self._tokenizer is not None and self._device is not None

        outputs = []
        special_ids = tuple(int(token_id) for token_id in self._tokenizer.all_special_ids)
        with torch.inference_mode():
            for start in range(0, len(sequences), self.batch_size):
                batch = list(sequences[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
                last_hidden = self._model(**encoded).last_hidden_state
                residue_mask = encoded["attention_mask"].bool()
                for token_id in special_ids:
                    residue_mask &= encoded["input_ids"].ne(token_id)
                # An unexpected tokenizer configuration should not create NaNs.
                empty_rows = residue_mask.sum(dim=1).eq(0)
                if empty_rows.any():
                    residue_mask[empty_rows] = encoded["attention_mask"][empty_rows].bool()
                pooled = (last_hidden * residue_mask.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
                outputs.append(pooled.detach().float().cpu())
        return torch.cat(outputs, dim=0)


@dataclass
class CacheBatch:
    """One contiguous cached embedding batch and its standardized metadata."""

    indices: np.ndarray
    metadata: pd.DataFrame
    embeddings: Any


class ShardedEmbeddingCache:
    """On-disk cache that permits scoring a large TrEMBL pool in batches."""

    manifest_name = "manifest.json"
    required_columns = (
        "uniprot_id",
        "sequence",
        "label",
        "source",
        "sequence_length",
        "sequence_hash",
    )

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest_path = self.directory / self.manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding cache manifest does not exist: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.parts = self.manifest["parts"]
        self.num_records = int(self.manifest["num_records"])

    @property
    def embedding_dim(self) -> int:
        return int(self.manifest["embedding_dim"])

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        return (Path(directory) / cls.manifest_name).is_file()

    @classmethod
    def create(
        cls,
        directory: str | Path,
        frames: Iterable[pd.DataFrame],
        encoder: ESM2MeanPoolEncoder,
        shard_size: int,
        cache_metadata: Mapping[str, Any],
    ) -> "ShardedEmbeddingCache":
        """Encode frames once and create immutable Torch shards.

        Existing manifests are reused deliberately.  A changed model or input
        should use a new output directory, avoiding accidental stale caches.
        """
        directory = Path(directory)
        if cls.exists(directory):
            return cls(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Embedding cache directory is non-empty without a manifest: {directory}. "
                "Choose a new cache directory rather than overwriting it."
            )
        directory.mkdir(parents=True, exist_ok=True)
        if shard_size < 1:
            raise ValueError("cache_shard_size must be at least one.")

        torch = _torch()
        pending: list[pd.DataFrame] = []
        pending_rows = 0
        parts: list[dict[str, Any]] = []
        total = 0
        embedding_dim: int | None = None

        def flush() -> None:
            nonlocal pending_rows, total, embedding_dim
            if not pending:
                return
            frame = pd.concat(pending, ignore_index=True)
            pending.clear()
            pending_rows = 0
            for column in cls.required_columns:
                if column not in frame.columns:
                    raise ValueError(f"Embedding input is missing '{column}'.")
            embeddings = encoder.encode(frame["sequence"].astype(str).tolist())
            if embedding_dim is None:
                embedding_dim = int(embeddings.shape[1])
            elif embedding_dim != int(embeddings.shape[1]):
                raise RuntimeError("Encoder produced inconsistent embedding dimensions.")
            part_index = len(parts)
            filename = f"part-{part_index:05d}.pt"
            metadata_columns = [column for column in frame.columns if column != "sequence"]
            payload = {
                "embeddings": embeddings.contiguous(),
                "metadata": {
                    column: frame[column].astype(str).tolist()
                    if frame[column].dtype == object
                    else frame[column].tolist()
                    for column in metadata_columns
                },
            }
            torch.save(payload, directory / filename)
            rows = len(frame)
            parts.append({"file": filename, "offset": total, "rows": rows})
            total += rows

        for incoming in frames:
            if incoming.empty:
                continue
            incoming = incoming.copy()
            for start in range(0, len(incoming), shard_size):
                pending.append(incoming.iloc[start : start + shard_size].copy())
                pending_rows += min(shard_size, len(incoming) - start)
                if pending_rows >= shard_size:
                    flush()
        flush()
        if total == 0 or embedding_dim is None:
            raise ValueError("No embeddings were written; check dataset preparation and candidate filters.")
        manifest = {
            "format_version": 1,
            "num_records": total,
            "embedding_dim": embedding_dim,
            "parts": parts,
            "cache_metadata": dict(cache_metadata),
        }
        (directory / cls.manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(directory)

    def _load_part(self, part: Mapping[str, Any]) -> tuple[pd.DataFrame, Any]:
        payload = _torch_load(self.directory / str(part["file"]))
        metadata = pd.DataFrame(payload["metadata"])
        return metadata, payload["embeddings"].float()

    def iter_batches(self, batch_size: int) -> Iterator[CacheBatch]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        for part in self.parts:
            metadata, embeddings = self._load_part(part)
            offset = int(part["offset"])
            for start in range(0, len(metadata), batch_size):
                end = min(start + batch_size, len(metadata))
                yield CacheBatch(
                    indices=np.arange(offset + start, offset + end, dtype=np.int64),
                    metadata=metadata.iloc[start:end].reset_index(drop=True),
                    embeddings=embeddings[start:end],
                )

    def load_all(self) -> CacheBatch:
        torch = _torch()
        metadata_frames: list[pd.DataFrame] = []
        embedding_parts = []
        for part in self.parts:
            metadata, embeddings = self._load_part(part)
            metadata_frames.append(metadata)
            embedding_parts.append(embeddings)
        return CacheBatch(
            indices=np.arange(self.num_records, dtype=np.int64),
            metadata=pd.concat(metadata_frames, ignore_index=True),
            embeddings=torch.cat(embedding_parts, dim=0),
        )

    def select(self, indices: Sequence[int]) -> CacheBatch:
        """Load selected cache rows while retaining the caller's order."""
        torch = _torch()
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or len(requested) == 0:
            raise ValueError("select() requires one or more one-dimensional cache indices.")
        if requested.min() < 0 or requested.max() >= self.num_records:
            raise IndexError("Requested embedding-cache index is out of range.")
        output_embeddings = None
        output_rows: list[dict[str, Any] | None] = [None] * len(requested)
        for part in self.parts:
            offset = int(part["offset"])
            stop = offset + int(part["rows"])
            positions = np.flatnonzero((requested >= offset) & (requested < stop))
            if len(positions) == 0:
                continue
            metadata, embeddings = self._load_part(part)
            if output_embeddings is None:
                output_embeddings = torch.empty((len(requested), embeddings.shape[1]), dtype=embeddings.dtype)
            local = requested[positions] - offset
            output_embeddings[torch.as_tensor(positions)] = embeddings[torch.as_tensor(local)]
            for output_position, local_position in zip(positions, local, strict=True):
                output_rows[int(output_position)] = metadata.iloc[int(local_position)].to_dict()
        if output_embeddings is None or any(row is None for row in output_rows):
            raise RuntimeError("Could not materialize every requested embedding-cache row.")
        return CacheBatch(
            indices=requested,
            metadata=pd.DataFrame(output_rows),
            embeddings=output_embeddings,
        )


def dataframe_frames(frame: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Expose an in-memory standardized frame through the cache writer API."""
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size].copy()


def build_cache(
    cache_directory: str | Path,
    frames: Iterable[pd.DataFrame],
    config: Mapping[str, Any],
    source_name: str,
) -> ShardedEmbeddingCache:
    """Build or reuse a frozen embedding cache with provenance metadata."""
    encoder = ESM2MeanPoolEncoder(config["model"])
    return ShardedEmbeddingCache.create(
        cache_directory,
        frames,
        encoder,
        int(config["model"]["cache_shard_size"]),
        {
            "source_name": source_name,
            "model_name": config["model"]["esm_model_name"],
            "max_length": config["model"]["max_length"],
            "encoder_mode": config["model"]["encoder_mode"],
        },
    )
