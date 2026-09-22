#!/usr/bin/env python3
"""Compare the uncapped and label-0-capped BP ESM-2 LoRA models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tacs.data import sanitize_sequence  # noqa: E402
from tacs.lora import ESM2LoRAClassifier  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "datasets/Swissprot/biological_process_inference_processed.csv"
DEFAULT_UNCAPPED = (
    PROJECT_ROOT
    / "outputs/tacs/bp_full_ceiling8_resumed_round3/completed_round_3"
)
DEFAULT_CAPPED = (
    PROJECT_ROOT
    / "outputs/tacs/bp_capped_ceiling8_biological_process_full_tacs_seed42"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/tacs/inference_evaluation/biological_process_two_models"

MODEL_RUNS = (
    ("uncapped_background", "Uncapped label 0", DEFAULT_UNCAPPED),
    ("capped_background_1_to_1", "Label 0 capped 1:1", DEFAULT_CAPPED),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--uncapped-run", type=Path, default=DEFAULT_UNCAPPED)
    parser.add_argument("--capped-run", type=Path, default=DEFAULT_CAPPED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round", type=int, default=3, dest="round_index")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the data and model artifacts without loading ESM-2.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_inference_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Inference CSV does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"Sequence", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Inference CSV is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["sequence"] = frame["Sequence"].map(sanitize_sequence)
    empty = frame["sequence"].eq("")
    if empty.any():
        raise ValueError(f"Inference CSV contains {int(empty.sum())} empty sequences after cleaning.")
    numeric_labels = pd.to_numeric(frame["label"], errors="raise")
    if not np.allclose(numeric_labels.to_numpy(), numeric_labels.astype(np.int64).to_numpy()):
        raise ValueError("Inference labels must be integers.")
    frame["label"] = numeric_labels.astype(np.int64)
    return frame


def artifact_paths(run_dir: Path, round_index: int) -> tuple[Path, Path]:
    adapter = run_dir / f"lora_adapter_round_{round_index}"
    checkpoint = run_dir / f"model_round_{round_index}.pt"
    missing = [str(path) for path in (adapter, checkpoint) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))
    return adapter, checkpoint


def checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"labels", "model_config", "lora_config", "classifier_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing fields: {sorted(missing)}")
    return checkpoint


def calculate_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_labels: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    one_hot = label_binarize(true_labels, classes=class_labels)
    if one_hot.shape[1] != len(class_labels):
        raise ValueError("Could not construct one-vs-rest targets for every model class.")

    metrics: dict[str, float | int] = {
        "Samples": int(len(true_labels)),
        "Accuracy": float(accuracy_score(true_labels, predictions)),
        "F1 Score": float(
            f1_score(
                true_labels,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(true_labels, predictions)),
        "AUROC": float(
            roc_auc_score(
                true_labels,
                probabilities,
                labels=class_labels,
                average="macro",
                multi_class="ovr",
            )
        ),
        "AUPRC": float(average_precision_score(one_hot, probabilities, average="macro")),
    }
    matrix = confusion_matrix(true_labels, predictions, labels=class_labels)
    return metrics, matrix


def save_confusion_matrix(path: Path, matrix: np.ndarray, class_labels: np.ndarray) -> None:
    columns = [f"pred_{label}" for label in class_labels]
    index = [f"true_{label}" for label in class_labels]
    pd.DataFrame(matrix, index=index, columns=columns).rename_axis("true_label").to_csv(path)


def evaluate_one_model(
    model_key: str,
    display_name: str,
    run_dir: Path,
    round_index: int,
    batch_size: int,
    device: str,
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    adapter, checkpoint_path = artifact_paths(run_dir, round_index)
    checkpoint = checkpoint_metadata(checkpoint_path)
    class_labels = tuple(int(label) for label in checkpoint["labels"])
    if not class_labels:
        raise ValueError(f"Checkpoint has no class labels: {checkpoint_path}")
    unknown = sorted(set(frame["label"].tolist()).difference(class_labels))
    if unknown:
        raise ValueError(f"Inference CSV contains labels absent from {model_key}: {unknown}")

    model_config = dict(checkpoint["model_config"])
    lora_config = dict(checkpoint["lora_config"])
    model_config["device"] = device
    lora_config["inference_batch_size"] = batch_size

    print(
        json.dumps(
            {
                "phase": "load_model",
                "model": model_key,
                "run_dir": str(run_dir),
                "round": round_index,
                "samples": len(frame),
                "labels": list(class_labels),
            }
        ),
        flush=True,
    )
    model = ESM2LoRAClassifier(model_config, lora_config, len(class_labels))
    model.load_saved_adapter(adapter, checkpoint_path, class_labels)
    probability_tensor, embedding_tensor = model.predict_sequences(
        frame["sequence"].tolist(),
        batch_size,
        progress_description=f"{display_name} inference",
    )
    probabilities = probability_tensor.numpy()
    label_array = np.asarray(class_labels, dtype=np.int64)
    predictions = label_array[probabilities.argmax(axis=1)]
    true_labels = frame["label"].to_numpy(dtype=np.int64)
    metrics, matrix = calculate_metrics(true_labels, predictions, probabilities, label_array)

    prediction_columns = [column for column in ("Entry", "Organism") if column in frame.columns]
    prediction_frame = frame[prediction_columns].copy()
    prediction_frame["true_label"] = true_labels
    prediction_frame["predicted_label"] = predictions
    for position, label in enumerate(class_labels):
        prediction_frame[f"probability_label_{label}"] = probabilities[:, position]
    prediction_frame.to_csv(output_dir / f"{model_key}_predictions.csv", index=False)
    save_confusion_matrix(output_dir / f"{model_key}_confusion_matrix.csv", matrix, label_array)

    result: dict[str, Any] = {
        "Model": display_name,
        **metrics,
        "model_key": model_key,
        "run_dir": str(run_dir),
        "round": round_index,
        "class_labels": list(class_labels),
        "confusion_matrix": matrix.tolist(),
    }

    del model, probability_tensor, embedding_tensor, probabilities
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, class_labels


def main() -> int:
    args = parse_args()
    if args.round_index < 0:
        raise ValueError("--round must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    input_path = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    run_specs = (
        ("uncapped_background", "Uncapped label 0", resolve_path(args.uncapped_run)),
        ("capped_background_1_to_1", "Label 0 capped 1:1", resolve_path(args.capped_run)),
    )
    frame = load_inference_data(input_path)

    metadata = []
    expected_labels: tuple[int, ...] | None = None
    for model_key, display_name, run_dir in run_specs:
        _, checkpoint_path = artifact_paths(run_dir, args.round_index)
        checkpoint = checkpoint_metadata(checkpoint_path)
        labels = tuple(int(label) for label in checkpoint["labels"])
        if expected_labels is None:
            expected_labels = labels
        elif labels != expected_labels:
            raise ValueError(
                f"The two checkpoints use different label spaces: {expected_labels} versus {labels}."
            )
        metadata.append(
            {
                "model_key": model_key,
                "model": display_name,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "labels": list(labels),
            }
        )

    label_counts = {
        str(int(label)): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }
    preflight = {
        "input_csv": str(input_path),
        "samples": int(len(frame)),
        "label_counts": label_counts,
        "models": metadata,
        "metric_definitions": {
            "F1 Score": "macro F1 across all checkpoint labels",
            "MCC": "multiclass Matthews correlation coefficient",
            "AUROC": "macro one-vs-rest AUROC",
            "AUPRC": "macro one-vs-rest average precision",
        },
    }
    print(json.dumps({"phase": "preflight", **preflight}, indent=2), flush=True)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_key, display_name, run_dir in run_specs:
        result, _ = evaluate_one_model(
            model_key,
            display_name,
            run_dir,
            args.round_index,
            args.batch_size,
            args.device,
            frame,
            output_dir,
        )
        results.append(result)

    metric_columns = ["Model", "Samples", "Accuracy", "F1 Score", "MCC", "AUROC", "AUPRC"]
    metric_frame = pd.DataFrame(results)[metric_columns]
    metric_frame.to_csv(output_dir / "comparison_metrics.csv", index=False)
    report = {"evaluation": preflight, "results": results}
    (output_dir / "comparison_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nComparison metrics (F1/AUROC/AUPRC are macro):", flush=True)
    print(metric_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    for result in results:
        print(f"\n{result['Model']} confusion matrix (rows=true, columns=predicted):", flush=True)
        print(np.asarray(result["confusion_matrix"]), flush=True)
    print(f"\nSaved results to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
