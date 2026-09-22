"""Dataset preparation, leakage-safe Swiss splits, and TrEMBL streaming.

The local Swiss-Prot files are keyword-labelled multiclass datasets.  The
large TrEMBL export contains a semicolon-separated list of UniProt keyword
identifiers.  This module derives a weak label only when those identifiers map
unambiguously to the chosen Swiss-Prot task.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd


TASK_KEYWORD_COLUMNS = {
    "biological_process": "Biological process Keywords",
    "cellular_component": "Cellular component Keywords",
    "molecular_function": "Molecular function Keywords",
}
_KW_RE = re.compile(r"KW-\d{4}")
_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_AMBIGUOUS_AMINO_ACIDS = str.maketrans({"B": "X", "J": "X", "O": "X", "U": "X", "Z": "X"})


def set_max_csv_field_size() -> None:
    """Allow long protein and annotation fields across Python builds."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def keyword_column_for_task(task: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        return TASK_KEYWORD_COLUMNS[task]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_KEYWORD_COLUMNS))
        raise ValueError(f"Unknown task '{task}'. Supply data.keyword_column or use: {valid}") from exc


def sanitize_sequence(sequence: Any) -> str:
    """Canonicalize a sequence for ESM-2 without silently retaining gaps."""
    if sequence is None or (isinstance(sequence, float) and math.isnan(sequence)):
        return ""
    cleaned = "".join(str(sequence).upper().split()).replace("-", "").translate(_AMBIGUOUS_AMINO_ACIDS)
    return "".join(residue if residue in _VALID_AMINO_ACIDS else "X" for residue in cleaned)


def sequence_digest(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("ascii", errors="replace")).hexdigest()


def extract_keywords(value: Any) -> tuple[str, ...]:
    """Extract canonical UniProt keyword IDs from an annotation cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    return tuple(sorted(set(_KW_RE.findall(str(value).upper()))))


@dataclass(frozen=True)
class LabelSpace:
    """Task-specific mapping from UniProt keyword IDs to integer labels."""

    keyword_to_label: dict[str, int]
    labels: tuple[int, ...]
    background_label: int
    task: str
    keyword_column: str

    def choose_label(self, keywords: Iterable[str], conflict_policy: str) -> int | None:
        matches = sorted({self.keyword_to_label[keyword] for keyword in keywords if keyword in self.keyword_to_label})
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        if conflict_policy == "drop":
            return None
        if conflict_policy == "specific_over_background":
            specific = [label for label in matches if label != self.background_label]
            if len(specific) == 1:
                return specific[0]
            return None
        if conflict_policy == "majority":
            # The mapping itself has already resolved keyword-level ties.  For a
            # multi-keyword protein use the smallest label only as a deterministic
            # fallback; callers should prefer the safer specific_over_background.
            return matches[0]
        raise ValueError(f"Unknown conflict_policy: {conflict_policy}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "keyword_column": self.keyword_column,
            "background_label": self.background_label,
            "labels": list(self.labels),
            "keyword_to_label": self.keyword_to_label,
        }


def build_label_space(swiss: pd.DataFrame, config: Mapping[str, Any]) -> LabelSpace:
    """Infer the exact keyword-to-class mapping from the local Swiss data."""
    data_config = config["data"]
    keyword_column = keyword_column_for_task(data_config["task"], data_config.get("keyword_column"))
    label_column = data_config["label_column"]
    if keyword_column not in swiss.columns:
        raise ValueError(f"Swiss file has no keyword column '{keyword_column}'. Columns: {list(swiss.columns)}")
    if label_column not in swiss.columns:
        raise ValueError(f"Swiss file has no label column '{label_column}'.")

    votes: dict[str, Counter[int]] = defaultdict(Counter)
    for keywords, label in zip(swiss[keyword_column], swiss[label_column], strict=True):
        if pd.isna(label):
            continue
        for keyword in extract_keywords(keywords):
            votes[keyword][int(label)] += 1

    mapping: dict[str, int] = {}
    for keyword, counts in votes.items():
        # Stable tie-breaking makes the resulting task definition reproducible.
        mapping[keyword] = min(counts, key=lambda label: (-counts[label], label))
    if not mapping:
        raise ValueError("No UniProt KW-#### identifiers were found in the Swiss label column.")
    labels = tuple(sorted({int(label) for label in swiss[label_column].dropna().unique()}))
    return LabelSpace(
        keyword_to_label=mapping,
        labels=labels,
        background_label=int(data_config["background_label"]),
        task=data_config["task"],
        keyword_column=keyword_column,
    )


def _resolve_conflicting_swiss_labels(
    swiss: pd.DataFrame,
    label_space: LabelSpace,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Deduplicate exact proteins and remove ambiguous multiclass supervision."""
    policy = config["data"]["conflict_policy"]
    retained_rows: list[int] = []
    label_column = config["data"]["label_column"]
    for _, group in swiss.groupby("sequence_hash", sort=False):
        labels = tuple(sorted({int(label) for label in group[label_column]}))
        if len(labels) == 1:
            retained_rows.append(int(group.index[0]))
            continue
        chosen = label_space.choose_label(
            [keyword for value in group["keywords"] for keyword in extract_keywords(value)],
            policy,
        )
        if chosen is None:
            continue
        matching = group[group[label_column].astype(int) == chosen]
        if not matching.empty:
            retained_rows.append(int(matching.index[0]))
    return swiss.loc[retained_rows].copy().reset_index(drop=True)


def load_swiss_dataset(config: Mapping[str, Any]) -> tuple[pd.DataFrame, LabelSpace]:
    """Load and clean Swiss-Prot while keeping the task's original class IDs."""
    data_config = config["data"]
    path = Path(data_config["swiss_csv"])
    if not path.is_file():
        raise FileNotFoundError(f"Swiss-Prot CSV does not exist: {path}")
    swiss_raw = pd.read_csv(path)
    label_space = build_label_space(swiss_raw, config)
    required = [data_config["id_column"], data_config["sequence_column"], data_config["label_column"]]
    missing = [column for column in required if column not in swiss_raw.columns]
    if missing:
        raise ValueError(f"Swiss file is missing required columns: {missing}")

    keyword_column = label_space.keyword_column
    output = pd.DataFrame(
        {
            "uniprot_id": swiss_raw[data_config["id_column"]].astype(str),
            "sequence": swiss_raw[data_config["sequence_column"]].map(sanitize_sequence),
            "label": swiss_raw[data_config["label_column"]].astype(int),
            "keywords": swiss_raw[keyword_column].fillna("").astype(str),
            "source": "swiss",
        }
    )
    cluster_column = data_config.get("cluster_column")
    if cluster_column:
        if cluster_column not in swiss_raw.columns:
            raise ValueError(f"Configured cluster column '{cluster_column}' is not in Swiss-Prot CSV.")
        output["cluster_id"] = swiss_raw[cluster_column].fillna("").astype(str)

    output = output[output["sequence"].ne("")].copy()
    output["sequence_length"] = output["sequence"].str.len().astype(int)
    output["sequence_hash"] = output["sequence"].map(sequence_digest)
    output = _resolve_conflicting_swiss_labels(output, label_space, config)
    if output.empty:
        raise ValueError("Swiss cleaning removed every row; inspect data.conflict_policy and input columns.")
    return output.reset_index(drop=True), label_space


def _partition_counts(n: int, fractions: list[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    for index in np.argsort(-(raw - counts))[:remainder]:
        counts[index] += 1
    # Preserve a representative of any class in each requested partition where
    # that is mathematically possible.
    nonzero = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if n >= len(nonzero):
        for index in nonzero:
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts.tolist()


def _random_stratified_split(
    swiss: pd.DataFrame,
    fractions: list[float],
    seed: int,
    split_config: Mapping[str, Any],
) -> np.ndarray:
    """Stratify every class, with optional rare-class holdout guarantees."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(swiss), dtype=object)
    min_validation = int(split_config.get("min_validation_per_class", 0))
    min_test = int(split_config.get("min_test_per_class", 0))
    rare_limit = split_config.get("rare_class_max_size")
    rare_seed_fraction = float(split_config.get("rare_class_seed_fraction", 0.5))
    for _, group in swiss.groupby("label", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        if min_validation == 0 and min_test == 0 and rare_limit is None:
            counts = _partition_counts(len(indices), fractions)
        else:
            baseline = _partition_counts(len(indices), fractions)
            validation_count = max(baseline[2], min_validation)
            test_count = max(baseline[3], min_test)
            remaining = len(indices) - validation_count - test_count
            if remaining < 2:
                raise ValueError(
                    f"Class {group.name} has {len(indices)} samples, insufficient for "
                    f"validation={validation_count}, test={test_count}, and both seed/candidate partitions."
                )
            is_rare = rare_limit is not None and len(indices) <= int(rare_limit)
            if not is_rare and validation_count == baseline[2] and test_count == baseline[3]:
                # Keep the normal largest-remainder 20/60/10/10 allocation bit-for-bit unchanged.
                counts = baseline
            else:
                seed_fraction = rare_seed_fraction if is_rare else fractions[0] / (fractions[0] + fractions[1])
                seed_count = int(round(remaining * seed_fraction))
                seed_count = min(max(seed_count, 1), remaining - 1)
                counts = [seed_count, remaining - seed_count, validation_count, test_count]
        start = 0
        for split_index, count in enumerate(counts):
            split[indices[start : start + count]] = split_index
            start += count
    return split


def _cluster_split(swiss: pd.DataFrame, fractions: list[float], seed: int) -> np.ndarray:
    """Greedily assign whole clusters while approximating label proportions."""
    if "cluster_id" not in swiss.columns:
        raise ValueError("split.method='cluster' requires data.cluster_column in the Swiss CSV.")
    if swiss["cluster_id"].eq("").any():
        raise ValueError("cluster split cannot use empty cluster identifiers.")
    labels = sorted(swiss["label"].unique())
    label_to_position = {label: position for position, label in enumerate(labels)}
    target = np.zeros((len(fractions), len(labels)), dtype=float)
    total_per_label = swiss.groupby("label").size()
    for split_index, fraction in enumerate(fractions):
        for label, count in total_per_label.items():
            target[split_index, label_to_position[int(label)]] = fraction * count

    rng = np.random.default_rng(seed)
    groups: list[tuple[str, np.ndarray, np.ndarray]] = []
    for cluster_id, group in swiss.groupby("cluster_id", sort=False):
        counts = np.zeros(len(labels), dtype=float)
        for label, count in group.groupby("label").size().items():
            counts[label_to_position[int(label)]] = count
        groups.append((str(cluster_id), group.index.to_numpy(), counts))
    # Large clusters first gives much better target matching.  The random key is
    # only a deterministic tie breaker, not a source of leakage.
    random_keys = rng.random(len(groups))
    groups = [group for _, group in sorted(zip(random_keys, groups, strict=True), key=lambda item: (-item[1][2].sum(), item[0]))]

    assigned = np.zeros_like(target)
    split = np.empty(len(swiss), dtype=object)
    for _, indices, counts in groups:
        costs = []
        for split_index in range(len(fractions)):
            proposed = assigned.copy()
            proposed[split_index] += counts
            # Relative squared deviation avoids the background class drowning out
            # smaller, important classes.
            costs.append(np.square((proposed - target) / np.maximum(target, 1.0)).sum())
        destination = int(np.argmin(costs))
        assigned[destination] += counts
        split[indices] = destination
    return split


def make_swiss_splits(swiss: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Create seed/candidate/validation/test sets with no duplicate sequences."""
    split_config = config["split"]
    fractions = [
        float(split_config["seed_fraction"]),
        float(split_config["candidate_fraction"]),
        float(split_config["validation_fraction"]),
        float(split_config["test_fraction"]),
    ]
    if split_config["method"] == "random":
        assigned = _random_stratified_split(swiss, fractions, int(config["experiment"]["seed"]), split_config)
    elif split_config["method"] == "cluster":
        if (int(split_config.get("min_validation_per_class", 0)) > 0
                or int(split_config.get("min_test_per_class", 0)) > 0
                or split_config.get("rare_class_max_size") is not None):
            raise ValueError("Rare-class minimum holdouts currently require split.method=random; cluster assignment cannot guarantee exact per-class minima.")
        assigned = _cluster_split(swiss, fractions, int(config["experiment"]["seed"]))
    else:
        raise ValueError("split.method must be random or cluster.")

    names = ("swiss_seed", "swiss_candidate", "swiss_validation", "swiss_test")
    result = {
        name: swiss.iloc[np.flatnonzero(assigned == index)].copy().reset_index(drop=True)
        for index, name in enumerate(names)
    }
    hashes = [set(frame["sequence_hash"]) for frame in result.values()]
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hashes[left].intersection(hashes[right]):
                raise AssertionError("Sequence overlap detected across Swiss splits.")
    return result


class _CandidateWriter:
    """Append candidate records without ever materializing the TrEMBL export."""

    columns = (
        "uniprot_id",
        "sequence",
        "label",
        "matched_keywords",
        "source",
        "sequence_length",
        "sequence_hash",
    )

    def __init__(self, output: Path):
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._csv_handle = None
        self._csv_writer = None
        self._parquet_writer = None
        self._format = output.suffix.lower()
        if self._format not in {".csv", ".parquet"}:
            raise ValueError("TrEMBL candidate output must end in .csv or .parquet.")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate pool: {output}")

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self._format == ".csv":
            if self._csv_writer is None:
                self._csv_handle = self.output.open("w", encoding="utf-8", newline="")
                self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.columns)
                self._csv_writer.writeheader()
            self._csv_writer.writerows(rows)
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Writing Parquet candidates requires pyarrow; use a .csv output instead.") from exc
        table = pa.Table.from_pylist(rows)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self.output, table.schema)
        self._parquet_writer.write_table(table)

    def close(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()
        if self._parquet_writer is not None:
            self._parquet_writer.close()


def prepare_trembl_candidates(
    config: Mapping[str, Any],
    output: str | Path | None = None,
    max_records: int | None = None,
    max_candidates: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream HDD_E TrEMBL and materialize only task-compatible weak labels.

    Candidate labels are inferred solely from the same UniProt keyword mapping
    used by the local Swiss-Prot task.  Exact sequence overlap with Swiss-Prot
    is removed before any cache or model sees the candidate.
    """
    set_max_csv_field_size()
    data_config = config["data"]
    trembl_path = Path(data_config["trembl_source_csv"])
    if not trembl_path.is_file():
        raise FileNotFoundError(f"TrEMBL source CSV does not exist: {trembl_path}")
    output_path = Path(output or data_config["trembl_candidates"])
    if output_path.exists() and overwrite:
        output_path.unlink()

    swiss, label_space = load_swiss_dataset(config)
    swiss_hashes = set(swiss["sequence_hash"]) if data_config["drop_swiss_trembl_sequence_overlap"] else set()
    writer = _CandidateWriter(output_path)
    keyword_column = label_space.keyword_column
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    try:
        with trembl_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Entry", "Sequence", keyword_column}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"TrEMBL source is missing columns: {sorted(missing)}")
            for record_number, row in enumerate(reader, start=1):
                counters["records_seen"] += 1
                if record_number % 250_000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "candidate_preparation",
                                "task": data_config["task"],
                                "records_seen": int(counters["records_seen"]),
                                "candidates_written": int(counters["candidates_written"]),
                                "swiss_overlap": int(counters["swiss_overlap"]),
                                "no_unambiguous_task_label": int(counters["no_unambiguous_task_label"]),
                            }
                        ),
                        flush=True,
                    )
                if max_records is not None and record_number > max_records:
                    break
                if row.get(None):
                    counters["malformed_rows"] += 1
                    continue
                if (row.get("Reviewed") or "unreviewed").strip().lower() not in {"", "unreviewed"}:
                    counters["reviewed_rows_skipped"] += 1
                    continue
                sequence = sanitize_sequence(row.get("Sequence"))
                if not sequence:
                    counters["empty_sequence"] += 1
                    continue
                digest = sequence_digest(sequence)
                if digest in swiss_hashes:
                    counters["swiss_overlap"] += 1
                    continue
                keywords = extract_keywords(row.get(keyword_column))
                label = label_space.choose_label(keywords, data_config["conflict_policy"])
                if label is None:
                    counters["no_unambiguous_task_label"] += 1
                    continue
                rows.append(
                    {
                        "uniprot_id": str(row["Entry"]),
                        "sequence": sequence,
                        "label": int(label),
                        "matched_keywords": ";".join(keyword for keyword in keywords if keyword in label_space.keyword_to_label),
                        "source": "trembl",
                        "sequence_length": len(sequence),
                        "sequence_hash": digest,
                    }
                )
                counters["candidates_written"] += 1
                if len(rows) >= int(data_config["candidate_read_batch_size"]):
                    writer.write(rows)
                    rows.clear()
                if max_candidates is not None and counters["candidates_written"] >= max_candidates:
                    break
        writer.write(rows)
    finally:
        writer.close()

    summary = {
        "source": str(trembl_path),
        "output": str(output_path),
        "label_space": label_space.to_dict(),
        "counts": dict(counters),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def iter_candidate_frames(path: str | Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield standardized TrEMBL candidate frames from CSV or Parquet."""
    path = Path(path)
    required = ["uniprot_id", "sequence", "label", "source", "sequence_length", "sequence_hash"]
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, chunksize=batch_size, usecols=lambda column: column in set(required + ["matched_keywords"]))
        return
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Reading Parquet candidates requires pyarrow.") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
        return
    raise ValueError(f"Unsupported candidate pool format: {path}")


def write_split_manifest(splits: Mapping[str, pd.DataFrame], label_space: LabelSpace, output: str | Path) -> None:
    """Persist reproducibility metadata without duplicating protein sequences."""
    summary = {
        "label_space": label_space.to_dict(),
        "splits": {
            name: {
                "rows": len(frame),
                "label_counts": {str(label): int(count) for label, count in frame["label"].value_counts().sort_index().items()},
                "unique_sequences": int(frame["sequence_hash"].nunique()),
            }
            for name, frame in splits.items()
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
