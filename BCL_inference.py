#!/usr/bin/env python3
"""Run a saved BCL protein classifier on a SwissProt inference dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import BCL_main as bcl
from confusion_matrix import inference_report
from inference_only import build_confusion_matrix, metric_summary


ROOT = Path(__file__).resolve().parent
INFERENCE_FILES = {
    "CC": ROOT / "datasets/Swissprot/cellular_component_inference_processed.csv",
    "BP": ROOT / "datasets/Swissprot/biological_process_inference_processed.csv",
    "MF": ROOT / "datasets/Swissprot/molecular_function_inference_processed.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=bcl.TASKS, required=True)
    parser.add_argument("--model", choices=bcl.MODELS, default="ESM2_650M")
    parser.add_argument("--normal-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[torch.nn.Module, Any, bool]:
    saved_args = checkpoint.get("args", {})
    if saved_args.get("model", args.model) != args.model:
        raise ValueError("--model does not match the model saved in the BCL checkpoint.")
    saved_lora_layers = saved_args.get("lora_last_n_layers")
    if saved_lora_layers is None:
        raise ValueError("Checkpoint does not record --lora-last-n-layers.")

    encoder, tokenizer, bert_style, _ = bcl.load_encoder(args.model)
    lora_targets, _, _ = bcl.final_layer_lora_targets(encoder, int(saved_lora_layers))
    if lora_targets:
        encoder = get_peft_model(
            encoder,
            LoraConfig(
                inference_mode=True,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=lora_targets,
            ),
        )
    else:
        encoder.requires_grad_(False)

    num_classes = bcl.TASKS[args.task][1]
    feature_dim = int(saved_args.get("feature_dim", 1024))
    model = bcl.BCLProteinClassifier(encoder, num_classes, feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, tokenizer, bert_style


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, frame: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    logits_batches = []
    for batch in tqdm(loader, desc="SwissProt inference", unit="batch", dynamic_ncols=True):
        tokens = {name: value.to(device) for name, value in batch["tokens"].items()}
        logits, _, _ = model(**tokens)
        logits_batches.append(logits.cpu())
    probabilities = torch.softmax(torch.cat(logits_batches), dim=1).numpy()
    result = frame.copy()
    result["predicted_label"] = probabilities.argmax(axis=1)
    for label in range(probabilities.shape[1]):
        result[f"probability_{label}"] = probabilities[:, label]
    return result


def main() -> int:
    args = parse_args()
    if args.normal_size < 1 or args.batch_size < 1:
        raise ValueError("--normal-size and --batch-size must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    # These are locally produced, trusted checkpoints; args contains standard
    # Python values such as output paths, so weights_only must be disabled.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, tokenizer, bert_style = load_model(args, checkpoint)
    model.to(device)

    frame = pd.read_csv(INFERENCE_FILES[args.task])
    required = {"Sequence", "label"}
    if required.difference(frame.columns):
        raise ValueError(f"Inference CSV is missing columns: {sorted(required.difference(frame.columns))}")
    loader = DataLoader(
        bcl.ProteinDataset(frame, bert_style),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bcl.single_collate(tokenizer),
        pin_memory=device.type == "cuda",
    )
    predictions = predict(model, loader, device, frame)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "inference_predictions.csv"
    confusion_path = output / "inference_confusion_matrix.csv"
    report_path = output / "inference_report.md"
    metadata_path = output / "inference_metadata.json"
    predictions.to_csv(predictions_path, index=False)
    build_confusion_matrix(predictions, args.task).to_csv(confusion_path)
    report_path.write_text(inference_report(predictions, args.task), encoding="utf-8")
    metadata = {
        "experiment_type": "bcl",
        "model": args.model,
        "task": args.task,
        "normal_samples": args.normal_size,
        "checkpoint": str(checkpoint_path),
        "inference_data": str(INFERENCE_FILES[args.task]),
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
