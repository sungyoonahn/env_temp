#!/usr/bin/env python3
"""Run an ESM-2 TACS experiment for biological process, molecular function, or cellular component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tacs.config import VALID_ABLATIONS, load_config
from tacs.runner import run_tacs


TASK_CONFIGS = {
    "biological_process": PROJECT_ROOT / "configs/tacs_go_lora_capped_matched_split.yaml",
    "molecular_function": PROJECT_ROOT / "configs/tacs_molecular_function_lora.yaml",
    "cellular_component": PROJECT_ROOT / "configs/tacs_cellular_component_lora.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--config",
        type=Path,
        help=f"TACS YAML/JSON config",
        default="../configs/test.yaml",
    )
    selection.add_argument(
        "--task",
        choices=sorted(TASK_CONFIGS),
        help="Use the maintained ESM-2 650M LoRA configuration for this GO category.",
    )
    parser.add_argument("--ablation", choices=sorted(VALID_ABLATIONS), help="Override experiment.ablation.")
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate Swiss preparation and splits without loading ESM-2.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = TASK_CONFIGS[args.task] if args.task else args.config
    config = load_config(config_path)
    if args.ablation:
        config["experiment"]["ablation"] = args.ablation
    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir)
    result = run_tacs(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
