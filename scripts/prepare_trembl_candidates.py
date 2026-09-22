#!/usr/bin/env python3
"""Stream the HDD_E TrEMBL export into a task-compatible weak-label pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tacs.config import load_config
from tacs.data import prepare_trembl_candidates


TASK_CONFIGS = {
    "biological_process": PROJECT_ROOT / "configs/tacs_go_lora_capped_matched_split.yaml",
    "molecular_function": PROJECT_ROOT / "configs/tacs_molecular_function_lora.yaml",
    "cellular_component": PROJECT_ROOT / "configs/tacs_cellular_component_lora.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", type=Path, help="TACS YAML/JSON config.")
    selection.add_argument(
        "--task",
        choices=sorted(TASK_CONFIGS),
        help="Use the maintained ESM-2 650M LoRA configuration for this GO category.",
    )
    parser.add_argument("--output", type=Path, help="Candidate .csv or .parquet path (overrides config).")
    parser.add_argument("--max-records", type=int, help="Read at most this many source rows; useful for a smoke test.")
    parser.add_argument("--max-candidates", type=int, help="Stop after writing this many compatible candidates.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing candidate output deliberately.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_records is not None and args.max_records < 1:
        raise SystemExit("--max-records must be positive.")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive.")
    config_path = args.config or TASK_CONFIGS.get(args.task or "", PROJECT_ROOT / "configs/tacs_go.yaml")
    summary = prepare_trembl_candidates(
        load_config(config_path),
        output=args.output,
        max_records=args.max_records,
        max_candidates=args.max_candidates,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
