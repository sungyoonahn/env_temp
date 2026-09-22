#!/usr/bin/env python3
"""Run a saved SwissProt model checkpoint on the separate inference dataset only."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import main as training
import train as training_module
from confusion_matrix import INFERENCE_LABELS, inference_report


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "outputs" / "inference_only"
# These are intentionally independent from training batch sizes: this script
# runs only forward passes under torch.inference_mode().
INFERENCE_BATCH_SIZES = {
    "ProtBERT": 32,
    "ESM2_650M": 16,
    "ESM2_3B": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=training.MODELS,
        help="Checkpoint architecture: ProtBERT, ESM2_650M, or ESM2_3B.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=training.TASKS,
        help="SwissProt category: CC, BP, or MF.",
    )
    parser.add_argument(
        "--normal-size",
        type=int,
        required=True,
        metavar="N",
        help="Normal-sample count used for the training checkpoint (for lookup/output grouping).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to best_weights.pth. Defaults to the matching main.py output path.",
    )
    parser.add_argument(
        "--training-output-root",
        type=Path,
        default=training.OUTPUT_ROOT,
        help="Root containing main.py checkpoints when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root for inference-only results.",
    )
    parser.add_argument(
        "--test-type",
        default="swissprot_inference",
        help="Result-group name, e.g. swissprot_inference (default).",
    )
    parser.add_argument(
        "--experiment-type",
        default="baseline",
        help="Experiment-group name, e.g. baseline (default).",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional extra result-group name to distinguish checkpoints/settings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size. Defaults: ProtBERT=32, ESM2_650M=16, ESM2_3B=8.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def path_component(value: str) -> str:
    """Return a safe, readable directory component without changing common names."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not result:
        raise ValueError("--test-type and --run-tag must contain at least one valid character.")
    return result


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint or (
        args.training_output_root
        / f"normal_{args.normal_size}"
        / args.model
        / args.task
        / "best_weights.pth"
    )
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint}\nUse --checkpoint PATH or select a completed model/task/normal-size run."
        )
    return checkpoint


def result_directory(args: argparse.Namespace) -> Path:
    """Separate every output by inference/test type and selected hyperparameters."""
    directory = (
        args.results_root
        / path_component(args.experiment_type)
        / path_component(args.test_type)
        / args.model
        / args.task
        / f"normal_{args.normal_size}"
    )
    if args.run_tag:
        directory /= path_component(args.run_tag)
    return directory


def build_confusion_matrix(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    class_ids = list(range(training.TASKS[task]["classes"]))
    matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(
        index=class_ids, columns=class_ids, fill_value=0
    )
    labels = INFERENCE_LABELS[task]
    matrix.index = [f"True: {label}" for label in labels]
    matrix.columns = [f"Pred: {label}" for label in labels]
    return matrix


def metric_summary(frame: pd.DataFrame, task: str) -> dict[str, float]:
    """Metrics follow the project reporting convention used in confusion_matrix.py."""
    classes = training.TASKS[task]["classes"]
    class_ids = np.arange(classes)
    true = frame["label"].to_numpy(dtype=int)
    predicted = frame["predicted_label"].to_numpy(dtype=int)
    probabilities = frame[[f"probability_{index}" for index in class_ids]].to_numpy(
        dtype=float
    )
    counts = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(
        index=class_ids, columns=class_ids, fill_value=0
    ).to_numpy(dtype=float)
    recalls = np.divide(
        np.diag(counts),
        counts.sum(axis=1),
        out=np.zeros(classes, dtype=float),
        where=counts.sum(axis=1) != 0,
    )
    smoothed_recalls = np.maximum(recalls, 1e-3)
    one_hot = label_binarize(true, classes=class_ids)

    try:
        auroc = float(
            roc_auc_score(
                true, probabilities, labels=class_ids, average="macro", multi_class="ovr"
            )
        )
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        auprc = float("nan")

    return {
        # Per the requested convention, Accuracy is balanced accuracy—the
        # arithmetic mean of actual per-class recalls (zeros stay zero here).
        "Accuracy": float(recalls.mean()),
        "F1 Score": float(
            f1_score(true, predicted, labels=class_ids, average="macro", zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(true, predicted)),
        "AUROC": auroc,
        "AUPRC": auprc,
        # Only H-ACC/G-ACC floor a zero recall to 1e-3.
        "H-ACC": float(classes / np.sum(1.0 / smoothed_recalls)),
        "G-ACC": float(np.prod(smoothed_recalls) ** (1.0 / classes)),
    }


def main() -> int:
    args = parse_args()
    if args.normal_size < 1:
        raise ValueError("--normal-size must be positive.")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    checkpoint = resolve_checkpoint(args)
    batch_size = args.batch_size or INFERENCE_BATCH_SIZES[args.model]
    output = result_directory(args)
    output.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu to run on CPU.")

    model, tokenizer, bert_style = training.make_model(args.model, args.task, args)
    state_dict = torch.load(checkpoint, map_location=training_module.device)
    model.load_state_dict(state_dict, strict=False)
    # A newly constructed model starts in train mode; dropout must be disabled
    # before exporting reproducible inference probabilities.
    model.eval()

    inference_frame = training.load_inference_frame(args.task)
    predictions, _ = training.predict_inference(
        model,
        inference_frame,
        tokenizer,
        bert_style,
        batch_size,
        workers=0,
        classes=training.TASKS[args.task]["classes"],
    )
    predictions_path = output / "inference_predictions.csv"
    confusion_path = output / "inference_confusion_matrix.csv"
    report_path = output / "inference_report.md"
    metadata_path = output / "inference_metadata.json"

    predictions.to_csv(predictions_path, index=False)
    build_confusion_matrix(predictions, args.task).to_csv(confusion_path)
    report_path.write_text(inference_report(predictions, args.task), encoding="utf-8")
    metadata: dict[str, Any] = {
        "experiment_type": args.experiment_type,
        "test_type": args.test_type,
        "model": args.model,
        "task": args.task,
        "normal_samples": args.normal_size,
        "device": args.device,
        "checkpoint": str(checkpoint),
        "inference_data": str(training.TASKS[args.task]["inference"]),
        "samples": len(predictions),
        "metrics": metric_summary(predictions, args.task),
        "outputs": {
            "predictions": str(predictions_path),
            "confusion_matrix": str(confusion_path),
            "report": str(report_path),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print(inference_report(predictions, args.task), end="")
    print(f"Saved results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
