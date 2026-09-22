#!/usr/bin/env python3
"""Run the three full-label ceiling-20 TACS train/inference pairs consecutively."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_bp_inference_two_models import (  # noqa: E402
    artifact_paths,
    checkpoint_metadata,
    evaluate_one_model,
    load_inference_data,
)
from tacs.config import load_config  # noqa: E402


@dataclass(frozen=True)
class Experiment:
    key: str
    display_name: str
    task: str
    config_path: Path


EXPERIMENTS = (
    Experiment(
        "cc_capped",
        "CC capped (ceiling 20)",
        "cellular_component",
        PROJECT_ROOT / "configs/tacs_cellular_component_lora.yaml",
    ),
    Experiment(
        "mf_capped",
        "MF capped (ceiling 20)",
        "molecular_function",
        PROJECT_ROOT / "configs/tacs_molecular_function_lora.yaml",
    ),
    Experiment(
        "bp_capped",
        "BP capped (ceiling 20)",
        "biological_process",
        PROJECT_ROOT / "configs/tacs_go_lora_capped_matched_split.yaml",
    ),
    Experiment(
        "cc_full",
        "CC full label 0 (ceiling 20)",
        "cellular_component",
        PROJECT_ROOT / "configs/tacs_cellular_component_full_lora.yaml",
    ),
    Experiment(
        "mf_full",
        "MF full label 0 (ceiling 20)",
        "molecular_function",
        PROJECT_ROOT / "configs/tacs_molecular_function_full_lora.yaml",
    ),
    Experiment(
        "bp_full",
        "BP full label 0 (ceiling 20)",
        "biological_process",
        PROJECT_ROOT / "configs/tacs_biological_process_full_lora.yaml",
    ),
)

PLAN = (EXPERIMENTS[3], EXPERIMENTS[4], EXPERIMENTS[5])

INFERENCE_INPUTS = {
    "biological_process": PROJECT_ROOT / "datasets/Swissprot/biological_process_inference_processed.csv",
    "molecular_function": PROJECT_ROOT / "datasets/Swissprot/molecular_function_inference_processed.csv",
    "cellular_component": PROJECT_ROOT / "datasets/Swissprot/cellular_component_inference_processed.csv",
}

DEFAULT_SEQUENCE_DIR = PROJECT_ROOT / "outputs/tacs/full_ceiling20_consecutive_sequence"
DEFAULT_INFERENCE_ROOT = PROJECT_ROOT / "outputs/tacs/inference_evaluation/full_ceiling20_sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only.")
    return parser.parse_args()


def run_output(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    return PROJECT_ROOT / experiment["output_dir"] / (
        f"{experiment['name']}_{config['data']['task']}_"
        f"{experiment['ablation']}_seed{experiment['seed']}"
    )


def candidate_path(config: dict[str, Any]) -> Path:
    path = Path(config["data"]["trembl_candidates"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_experiment(experiment: Experiment, config: dict[str, Any]) -> None:
    if config["data"]["task"] != experiment.task:
        raise ValueError(
            f"{experiment.key} config task is {config['data']['task']}, expected {experiment.task}."
        )
    if config["model"]["encoder_mode"] != "lora":
        raise ValueError(f"{experiment.key} is not configured for LoRA.")
    if int(config["training"]["epochs"]) != 20:
        raise ValueError(f"{experiment.key} must use an epoch ceiling of 20.")
    ratio = config["training"].get("max_background_to_other_ratio")
    expected_ratio = 1.0 if experiment.key.endswith("capped") else None
    if ratio != expected_ratio:
        raise ValueError(
            f"{experiment.key} background ratio is {ratio!r}; expected {expected_ratio!r}."
        )
    if experiment.key.endswith("full"):
        sampling = config["training"]["background_sampling"]
        expected_sampling = {
            "strategy": "rotating_hard_negative",
            "background_label": int(config["data"]["background_label"]),
            "background_to_non_background_ratio": 2.0,
            "coverage_epochs": 10,
            "hard_negative_fraction": 0.5,
        }
        for key, expected in expected_sampling.items():
            if sampling.get(key) != expected:
                raise ValueError(
                    f"{experiment.key} background_sampling.{key} is "
                    f"{sampling.get(key)!r}; expected {expected!r}."
                )
        if int(config["training"]["minimum_epochs_before_early_stopping"]) < int(
            sampling["coverage_epochs"]
        ):
            raise ValueError(f"{experiment.key} may stop before full background coverage.")
    background_label = int(config["data"]["background_label"])
    excluded_trembl = [int(label) for label in config["selection"].get("excluded_trembl_labels", [])]
    if excluded_trembl != [background_label]:
        raise ValueError(
            f"{experiment.key} must exclude only TrEMBL background label {background_label}; "
            f"configured exclusion is {excluded_trembl}."
        )


def run_command(command: list[str], phase: str, experiment_key: str) -> None:
    print(
        json.dumps(
            {"phase": phase, "experiment": experiment_key, "command": command},
            indent=2,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_candidate_pool(experiment: Experiment, config: dict[str, Any]) -> None:
    path = candidate_path(config)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    if path.is_file() and path.stat().st_size > 0 and summary_path.is_file():
        print(
            json.dumps(
                {
                    "phase": "candidate_pool_skip",
                    "experiment": experiment.key,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            ),
            flush=True,
        )
        return
    if path.exists() or summary_path.exists():
        raise RuntimeError(
            f"Candidate pool is incomplete or inconsistent: {path}. "
            "Move the partial pool and its summary, if any, before restarting."
        )
    run_command(
        [
            sys.executable,
            "-u",
            "scripts/prepare_trembl_candidates.py",
            "--config",
            str(experiment.config_path),
        ],
        "candidate_pool_start",
        experiment.key,
    )
    if not path.is_file() or path.stat().st_size == 0 or not summary_path.is_file():
        raise RuntimeError(f"Candidate preparation did not create a valid pool: {path}")


def ensure_training(experiment: Experiment, config: dict[str, Any]) -> Path:
    output = run_output(config)
    summary = output / "summary.json"
    if summary.is_file():
        print(
            json.dumps(
                {
                    "phase": "training_skip_completed",
                    "experiment": experiment.key,
                    "summary": str(summary),
                }
            ),
            flush=True,
        )
        return output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite an incomplete run for {experiment.key}: {output}. "
            "Inspect or move that directory before restarting the sequence."
        )
    run_command(
        [sys.executable, "-u", "scripts/train_tacs.py", "--config", str(experiment.config_path)],
        "training_start",
        experiment.key,
    )
    if not summary.is_file():
        raise RuntimeError(f"Training exited without a completion summary: {summary}")
    return output


def inference_output(experiment: Experiment) -> Path:
    return DEFAULT_INFERENCE_ROOT / experiment.key


def run_inference(
    experiment: Experiment,
    run_dir: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    output = inference_output(experiment)
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        print(
            json.dumps(
                {
                    "phase": "inference_skip_completed",
                    "experiment": experiment.key,
                    "metrics": str(metrics_path),
                }
            ),
            flush=True,
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    input_path = INFERENCE_INPUTS[experiment.task]
    frame = load_inference_data(input_path)
    output.mkdir(parents=True, exist_ok=True)
    result, _ = evaluate_one_model(
        experiment.key,
        experiment.display_name,
        run_dir,
        3,
        batch_size,
        device,
        frame,
        output,
    )
    result["input_csv"] = str(input_path)
    metric_columns = ["Model", "Samples", "Accuracy", "F1 Score", "MCC", "AUROC", "AUPRC"]
    pd.DataFrame([result])[metric_columns].to_csv(output / "metrics.csv", index=False)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        pd.DataFrame([result])[metric_columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        flush=True,
    )
    return result


def write_state(completed: list[dict[str, Any]]) -> None:
    DEFAULT_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "plan": [experiment.key for experiment in PLAN],
        "completed_pairs": [row["experiment"] for row in completed],
        "results": completed,
    }
    (DEFAULT_SEQUENCE_DIR / "sequence_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    configs: dict[str, dict[str, Any]] = {}
    plan_rows = []
    for experiment in PLAN:
        config = load_config(experiment.config_path)
        validate_experiment(experiment, config)
        artifact_paths(run_output(config), 3) if (run_output(config) / "summary.json").is_file() else None
        configs[experiment.key] = config
        plan_rows.append(
            {
                "experiment": experiment.key,
                "task": experiment.task,
                "config": str(experiment.config_path),
                "candidate_pool": str(candidate_path(config)),
                "candidate_exists": candidate_path(config).is_file(),
                "run_output": str(run_output(config)),
                "training_complete": (run_output(config) / "summary.json").is_file(),
                "inference_output": str(inference_output(experiment)),
                "inference_complete": (inference_output(experiment) / "metrics.json").is_file(),
            }
        )
    print(json.dumps({"phase": "sequence_plan", "steps": plan_rows}, indent=2), flush=True)
    if args.dry_run:
        return 0

    completed = []
    prepared_tasks: set[str] = set()
    for experiment in PLAN:
        config = configs[experiment.key]
        if experiment.task not in prepared_tasks:
            ensure_candidate_pool(experiment, config)
            prepared_tasks.add(experiment.task)
        run_dir = ensure_training(experiment, config)
        result = run_inference(experiment, run_dir, args.batch_size, args.device)
        completed.append({"experiment": experiment.key, **result})
        write_state(completed)

    metric_columns = [
        "experiment",
        "Model",
        "Samples",
        "Accuracy",
        "F1 Score",
        "MCC",
        "AUROC",
        "AUPRC",
    ]
    pd.DataFrame(completed)[metric_columns].to_csv(
        DEFAULT_SEQUENCE_DIR / "all_inference_metrics.csv", index=False
    )
    print(f"Completed all three full-label training/inference pairs: {DEFAULT_SEQUENCE_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
