#!/usr/bin/env python3
"""Resume the interrupted CC full-ceiling20 run after completed round 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resume_tacs_lora_round3 import label_counts, recover_sequences
from tacs.config import load_config
from tacs.lora import ESM2LoRAClassifier, evaluate_lora_classifier, fit_lora_classifier
from tacs.lora_runner import (
    _anchor_for_round,
    _choose_candidates,
    _ensure_contiguous_labels,
    _round_swiss_indices,
    _score_candidate_round,
    _training_frame,
)
from tacs.reliability import thresholds_for_round
from tacs.trainer import save_evaluation

DEFAULT_CONFIG = ROOT / "configs/tacs_cellular_component_full_lora.yaml"
DEFAULT_RUN = ROOT / "outputs/tacs/cc_full_ceiling20_cellular_component_full_tacs_seed42"
COMPLETED_ROUND = 1


def required_paths(run_dir: Path, candidate_path: Path) -> list[Path]:
    return [
        run_dir / f"lora_adapter_round_{COMPLETED_ROUND}",
        run_dir / f"model_round_{COMPLETED_ROUND}.pt",
        run_dir / "accepted_trembl.csv",
        run_dir / "resolved_config.yaml",
        run_dir / "swiss_seed.csv",
        run_dir / "swiss_candidate.csv",
        run_dir / "swiss_validation.csv",
        run_dir / "swiss_test.csv",
        candidate_path,
        candidate_path.with_suffix(candidate_path.suffix + ".summary.json"),
    ]


def validate(config: dict, run_dir: Path) -> dict:
    if config["data"]["task"] != "cellular_component":
        raise ValueError("This recovery command is only for the interrupted CC run.")
    if config["experiment"]["name"] != "cc_full_ceiling20":
        raise ValueError("Unexpected experiment name; refusing to resume the wrong run.")
    if config["model"]["encoder_mode"] != "lora":
        raise ValueError("Recovery requires model.encoder_mode=lora.")
    if config["experiment"]["ablation"] != "full_tacs":
        raise ValueError("Recovery requires the full_tacs ablation.")
    if int(config["training"]["rounds"]) != 3:
        raise ValueError("Recovery expects exactly three TACS rounds.")

    candidate_path = Path(config["data"]["trembl_candidates"])
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    missing = [str(path) for path in required_paths(run_dir, candidate_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing recovery artifacts: " + ", ".join(missing))
    if (run_dir / "summary.json").exists():
        raise FileExistsError(f"Run already has a completion summary: {run_dir / 'summary.json'}")
    conflicting = [
        run_dir / "model_round_2.pt",
        run_dir / "lora_adapter_round_2",
        run_dir / "model_round_3.pt",
        run_dir / "lora_adapter_round_3",
    ]
    present = [str(path) for path in conflicting if path.exists()]
    if present:
        raise FileExistsError("Refusing to overwrite later-round artifacts: " + ", ".join(present))

    checkpoint = torch.load(
        run_dir / f"model_round_{COMPLETED_ROUND}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if int(checkpoint.get("round", -1)) != COMPLETED_ROUND:
        raise RuntimeError("The recovery checkpoint is not round 1.")
    labels = tuple(int(label) for label in checkpoint.get("labels", []))
    _ensure_contiguous_labels(labels)
    accepted = pd.read_csv(run_dir / "accepted_trembl.csv")
    return {
        "valid": True,
        "run_dir": str(run_dir),
        "candidate_path": str(candidate_path),
        "checkpoint_round": COMPLETED_ROUND,
        "labels": list(labels),
        "accepted_trembl": int(len(accepted)),
        "resume_rounds": [2, 3],
    }


def run(config: dict, run_dir: Path) -> dict:
    state = validate(config, run_dir)
    candidate_path = Path(state["candidate_path"])
    labels = tuple(state["labels"])

    split_names = ("swiss_seed", "swiss_candidate", "swiss_validation", "swiss_test")
    splits = {
        name: pd.read_csv(run_dir / f"{name}.csv")
        for name in split_names
    }
    swiss_metadata = pd.concat(
        [splits[name].assign(split=name) for name in split_names],
        ignore_index=True,
    )
    validation = splits["swiss_validation"][["sequence", "label"]].reset_index(drop=True)
    test = splits["swiss_test"][["sequence", "label"]].reset_index(drop=True)
    all_train = np.flatnonzero(
        swiss_metadata["split"].isin(["swiss_seed", "swiss_candidate"]).to_numpy()
    )

    selected = recover_sequences(
        candidate_path,
        run_dir / "accepted_trembl.csv",
        int(config["data"]["candidate_read_batch_size"]),
    )
    selected_indices = set(selected["candidate_index"].to_numpy(dtype=np.int64).tolist())

    model = ESM2LoRAClassifier(config["model"], config["lora"], len(labels))
    model.load_saved_adapter(
        run_dir / f"lora_adapter_round_{COMPLETED_ROUND}",
        run_dir / f"model_round_{COMPLETED_ROUND}.pt",
        labels,
    )
    round_rows = []
    for round_index in range(COMPLETED_ROUND + 1, int(config["training"]["rounds"]) + 1):
        active, new_swiss = _round_swiss_indices(
            swiss_metadata,
            round_index,
            int(config["training"]["rounds"]),
            str(config["experiment"]["ablation"]),
            int(config["experiment"]["seed"]),
            int(config["data"]["background_label"]),
            config["training"].get("max_background_to_other_ratio"),
        )
        anchor = _anchor_for_round(
            model, swiss_metadata, active, all_train, selected, config
        )
        eligible, score_summary = _score_candidate_round(
            model,
            candidate_path,
            selected_indices,
            anchor,
            labels,
            config,
            round_index,
            run_dir / f"trembl_scores_round_{round_index}.csv",
        )
        new_selected = _choose_candidates(
            eligible,
            len(labels),
            thresholds_for_round(config["selection"], round_index),
        )
        if not new_selected.empty:
            selected = pd.concat([selected, new_selected], ignore_index=True)
            selected_indices.update(
                new_selected["candidate_index"].to_numpy(dtype=np.int64).tolist()
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_frame = _training_frame(swiss_metadata, active, selected, config)
        print(
            json.dumps(
                {
                    "phase": "resume_training",
                    "round": round_index,
                    "train_samples": int(len(train_frame)),
                    "new_selected_trembl": int(len(new_selected)),
                    "total_selected_trembl": int(len(selected)),
                }
            ),
            flush=True,
        )
        fit = fit_lora_classifier(
            train_frame,
            validation,
            len(labels),
            config["model"],
            config["lora"],
            config["training"],
            int(config["experiment"]["seed"]) + round_index,
            run_dir / f"round_{round_index}_epochs.jsonl",
        )
        model = fit.model
        model.save_adapter(run_dir, round_index, config, labels)
        selected.drop(columns=["sequence"], errors="ignore").to_csv(
            run_dir / "accepted_trembl.csv", index=False
        )
        round_rows.append(
            {
                "round": round_index,
                "resumed_from_completed_round": COMPLETED_ROUND,
                "total_swiss_samples": int(len(active)),
                "new_swiss_samples": int(len(new_swiss)),
                "total_accepted_trembl": int(len(selected)),
                "new_accepted_trembl": int(len(new_selected)),
                "accepted_trembl_by_class": label_counts(selected, "training_label"),
                "new_accepted_trembl_by_class": label_counts(
                    new_selected, "training_label"
                ),
                "training_class_counts": label_counts(train_frame, "label"),
                **score_summary,
                **{
                    f"validation_{key}": value
                    for key, value in fit.best_validation.items()
                    if key != "confusion_matrix"
                },
            }
        )

    evaluation = evaluate_lora_classifier(
        model,
        test,
        len(labels),
        int(config["lora"]["inference_batch_size"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        confidence_level=float(config["evaluation"]["confidence_level"]),
        bootstrap_seed=int(config["experiment"]["seed"]),
        progress_description="Swiss test",
    )
    save_evaluation(evaluation, run_dir, "swiss_test", list(labels))
    with (run_dir / "round_metrics_resumed.jsonl").open("w", encoding="utf-8") as handle:
        for row in round_rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "source_run": str(run_dir),
        "resumed_from_completed_round": COMPLETED_ROUND,
        "executed_rounds": [row["round"] for row in round_rows],
        "checkpoint": str(run_dir / "model_round_1.pt"),
        "adapter": str(run_dir / "lora_adapter_round_1"),
    }
    (run_dir / "resume_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "output": str(run_dir),
        "resume": manifest,
        "rounds": round_rows,
        "swiss_test": evaluation,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate(config, args.run_dir) if args.validate_only else run(config, args.run_dir)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
