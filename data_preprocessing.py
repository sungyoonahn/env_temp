#!/usr/bin/env python3
"""Report class distributions in the SwissProt trainable datasets.

Example:
    python data_preprocessing.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "CC": ROOT / "datasets/Swissprot/cellular_component_trainable.csv",
    "BP": ROOT / "datasets/Swissprot/biological_process_trainable.csv",
    "MF": ROOT / "datasets/Swissprot/molecular_function_trainable.csv",
}


def class_counts(path: Path) -> Counter[int]:
    """Return counts for the required integer ``label`` column."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} has no 'label' column.")
        return Counter(int(row["label"]) for row in reader)


def report(dataset_names: list[str]) -> None:
    for name in dataset_names:
        path = DATASETS[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} dataset: {path}")
        counts = class_counts(path)
        total = sum(counts.values())
        print(f"{name}: {len(counts)} classes, {total:,} samples")
        for label, count in sorted(counts.items()):
            print(f"  label {label}: {count:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Datasets to report (default: CC BP MF).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    report(parse_args().datasets)
