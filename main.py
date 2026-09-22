#!/usr/bin/env python3
"""Train SwissProt baselines with reproducibly downsampled Normal examples."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
from tqdm.auto import tqdm

import train as training_module

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/baseline_normal_sampled_20epoch"
TASKS = {
    "CC": {"csv": ROOT / "datasets/Swissprot/cellular_component_trainable.csv", "inference": ROOT / "datasets/Swissprot/cellular_component_inference_processed.csv", "classes": 6},
    "BP": {"csv": ROOT / "datasets/Swissprot/biological_process_trainable.csv", "inference": ROOT / "datasets/Swissprot/biological_process_inference_processed.csv", "classes": 5},
    "MF": {"csv": ROOT / "datasets/Swissprot/molecular_function_trainable.csv", "inference": ROOT / "datasets/Swissprot/molecular_function_inference_processed.csv", "classes": 3},
}
MODELS = {
    "ProtBERT": {"name": "Rostlab/prot_bert_bfd", "bert": True, "batch_size": 16},
    "ESM2_650M": {"name": "facebook/esm2_t33_650M_UR50D", "bert": False, "batch_size": 8},
    "ESM2_3B": {"name": "facebook/esm2_t36_3B_UR50D", "bert": False, "batch_size": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=["ESM2_650M"],
        help="Models to run (default: ESM2_650M only).",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many consecutive epochs without test macro-F1 improvement.",
    )
    parser.add_argument(
        "--minimum-epochs",
        type=int,
        default=5,
        help="Train at least this many epochs before early stopping may trigger.",
    )
    parser.add_argument(
        "--normal-sizes",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000],
        metavar="N",
        help="Numbers of label-0 (Normal) samples to retain before splitting.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the model-specific training and evaluation batch size.",
    )
    parser.add_argument(
        "--lora-last-n-layers",
        type=int,
        default=None,
        help=(
            "Apply Q/K/V LoRA only to the final N encoder layers. Use 0 to "
            "freeze the full encoder. Default: all encoder layers."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the final SwissProt inference and its output files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(
    task: str, seed: int, normal_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, int]]:
    spec = TASKS[task]
    frame = pd.read_csv(spec["csv"])
    required = {"Entry", "Sequence", "label", "tensor id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec['csv']} is missing columns: {sorted(missing)}")
    inference_entries = set(pd.read_csv(spec["inference"], usecols=["Entry"])["Entry"].astype(str))
    overlap = set(frame["Entry"].astype(str)).intersection(inference_entries)
    if overlap:
        raise RuntimeError(f"{task}: {len(overlap)} inference Entry IDs occur in the training table.")
    labels = frame["label"].astype(int)
    if labels.min() != 0 or labels.max() >= spec["classes"]:
        raise ValueError(f"{task} labels do not match its {spec['classes']}-class label space.")

    normal = frame[labels.eq(0)]
    if normal_size < 1:
        raise ValueError("--normal-sizes values must be positive.")
    if normal_size > len(normal):
        raise ValueError(
            f"{task}: requested {normal_size:,} Normal samples, but only {len(normal):,} are available."
        )
    sampled_normal = normal.sample(n=normal_size, random_state=seed)
    non_normal = frame[~labels.eq(0)]
    sampled = pd.concat([sampled_normal, non_normal], ignore_index=True)
    sampled_counts = {
        int(label): int(count)
        for label, count in sampled["label"].value_counts().sort_index().items()
    }

    train_frame, held_out = train_test_split(
        sampled,
        test_size=0.30,
        random_state=seed,
        stratify=sampled["label"],
    )
    valid_frame, test_frame = train_test_split(held_out, test_size=1 / 3, random_state=seed, stratify=held_out["label"])
    return (
        train_frame.reset_index(drop=True),
        valid_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        sampled_counts,
    )


def load_inference_frame(task: str) -> pd.DataFrame:
    """Load the task's held-out SwissProt inference table."""
    spec = TASKS[task]
    frame = pd.read_csv(spec["inference"])
    required = {"Entry", "Sequence", "label", "tensor id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec['inference']} is missing columns: {sorted(missing)}")
    labels = frame["label"].astype(int)
    if labels.min() != 0 or labels.max() >= spec["classes"]:
        raise ValueError(f"{task} inference labels do not match its class label space.")
    return frame.reset_index(drop=True)


def prepare(frame: pd.DataFrame, bert_style: bool) -> pd.DataFrame:
    result = frame.copy()
    sequence = result["Sequence"].astype(str).str.replace(r"[OBUZ]", "X", regex=True)
    result["sequence"] = sequence.map(lambda value: " ".join(value) if bert_style else value)
    return result


def make_loader(frame: pd.DataFrame, tokenizer: Any, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    dataset = training_module.CustomProteinDataset(frame["sequence"], frame["label"].astype(int), tokenizer, frame["tensor id"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=training_module.collate_fn, num_workers=workers)


def reset_rotary_caches(model: torch.nn.Module) -> None:
    """Drop ESM rotary tables before training so they are recreated in grad mode."""
    for module in model.modules():
        rotary = getattr(module, "rotary_embeddings", None)
        if rotary is not None:
            rotary._cos_cached = None
            rotary._sin_cached = None
            rotary._seq_len_cached = 0



def classification_metrics(labels: list[torch.Tensor], logits: list[torch.Tensor], classes: int) -> dict[str, float]:
    true_labels = torch.cat(labels).numpy()
    scores = torch.softmax(torch.cat(logits), dim=1).numpy()
    predicted_labels = scores.argmax(axis=1)
    one_hot = label_binarize(true_labels, classes=np.arange(classes))
    try:
        auroc = float(roc_auc_score(one_hot, scores, average="macro", multi_class="ovr"))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(one_hot, scores, average="macro"))
    except ValueError:
        auprc = float("nan")
    return {"accuracy": float(accuracy_score(true_labels, predicted_labels)), "f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)), "mcc": float(matthews_corrcoef(true_labels, predicted_labels)), "auroc": auroc, "auprc": auprc}


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: AdamW, criterion: CrossEntropyLoss, scheduler: lr_scheduler.LambdaLR, description: str, classes: int) -> dict[str, float]:
    model.train()
    reset_rotary_caches(model)
    losses, labels, predictions, correct, samples = [], [], [], 0, 0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(batch["input_ids"].to(training_module.device), batch["attention_mask"].to(training_module.device), None)
        target = batch["labels"].to(training_module.device)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        samples += target.numel()
        correct += int(logits.argmax(dim=1).eq(target).sum().item())
        losses.append(float(loss.item()))
        labels.append(target.detach().cpu())
        predictions.append(logits.detach().cpu())
        progress.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{correct / samples:.4f}")
    scheduler.step()
    return {"loss": float(np.mean(losses)), **classification_metrics(labels, predictions, classes)}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: CrossEntropyLoss, classes: int) -> dict[str, float]:
    model.eval()
    losses, labels, predictions = [], [], []
    for batch in loader:
        logits, _, _ = model(batch["input_ids"].to(training_module.device), batch["attention_mask"].to(training_module.device), None)
        target = batch["labels"].to(training_module.device)
        losses.append(float(criterion(logits, target).item()))
        labels.append(target.cpu())
        predictions.append(logits.cpu())
    return {"loss": float(np.mean(losses)), **classification_metrics(labels, predictions, classes)}


@torch.inference_mode()
def predict_inference(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    tokenizer: Any,
    bert_style: bool,
    batch_size: int,
    workers: int,
    classes: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return per-sequence probabilities and metrics for the external inference set."""
    # Keep SwissProt inference single-process to avoid exhausting file descriptors
    # after the training/validation/test DataLoader workers have run.
    loader = make_loader(prepare(frame, bert_style), tokenizer, batch_size, False, 0)
    labels, logits = [], []
    for batch in tqdm(loader, desc="SwissProt inference", unit="batch", dynamic_ncols=True):
        batch_logits, _, _ = model(
            batch["input_ids"].to(training_module.device),
            batch["attention_mask"].to(training_module.device),
            None,
        )
        labels.append(batch["labels"].cpu())
        logits.append(batch_logits.cpu())

    all_labels = torch.cat(labels)
    all_logits = torch.cat(logits)
    probabilities = torch.softmax(all_logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    result = frame.copy()
    result["predicted_label"] = predictions
    for label in range(classes):
        result[f"probability_{label}"] = probabilities[:, label]
    metrics = classification_metrics([all_labels], [all_logits], classes)
    metrics["loss"] = float(CrossEntropyLoss()(all_logits, all_labels).item())
    return result, metrics


def save_inference_results(
    model: torch.nn.Module,
    model_key: str,
    task: str,
    tokenizer: Any,
    bert_style: bool,
    batch_size: int,
    args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    """Run and save the separate SwissProt inference set for one trained checkpoint."""
    task_spec = TASKS[task]
    inference_frame = load_inference_frame(task)
    predictions, metrics = predict_inference(
        model,
        inference_frame,
        tokenizer,
        bert_style,
        batch_size,
        args.num_workers,
        task_spec["classes"],
    )
    predictions_path = output / "inference_predictions.csv"
    metrics_path = output / "inference_metrics.json"
    predictions.to_csv(predictions_path, index=False)
    summary = {
        "model": model_key,
        "task": task,
        "input": str(task_spec["inference"]),
        "samples": int(len(inference_frame)),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_encoder(model_key: str) -> tuple[torch.nn.Module, Any, bool]:
    spec = MODELS[model_key]
    if spec["bert"]:
        return BertModel.from_pretrained(spec["name"]), BertTokenizer.from_pretrained(spec["name"], do_lower_case=False), True
    return AutoModel.from_pretrained(spec["name"]), AutoTokenizer.from_pretrained(spec["name"], do_lower_case=False), False


def final_layer_lora_targets(
    encoder: torch.nn.Module, last_n_layers: int | None
) -> tuple[list[str], int, list[int]]:
    """Return exact Q/K/V LoRA targets for the requested final encoder layers."""
    total_layers = getattr(encoder.config, "num_hidden_layers", None)
    if total_layers is None:
        raise ValueError("Could not determine the number of encoder Transformer layers.")
    if last_n_layers is None:
        last_n_layers = total_layers
    if not 0 <= last_n_layers <= total_layers:
        raise ValueError(
            f"--lora-last-n-layers must be between 0 and {total_layers}, got {last_n_layers}."
        )
    layer_indices = list(range(total_layers - last_n_layers, total_layers))
    targets = [
        f"encoder.layer.{layer_index}.attention.self.{projection}"
        for layer_index in layer_indices
        for projection in ("query", "key", "value")
    ]
    return targets, total_layers, layer_indices


def make_model(model_key: str, task: str, args: argparse.Namespace) -> tuple[torch.nn.Module, Any, bool]:
    """Create a task head with the current LoRA configuration."""
    encoder, tokenizer, bert_style = load_encoder(model_key)
    training_module.tokenizer = tokenizer
    training_module.device = torch.device(args.device)
    training_module.args = argparse.Namespace(n_classes=TASKS[task]["classes"])
    lora_targets, _, _ = final_layer_lora_targets(encoder, args.lora_last_n_layers)
    if lora_targets:
        adapter = get_peft_model(
            encoder,
            LoraConfig(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=lora_targets,
            ),
        )
    else:
        encoder.requires_grad_(False)
        adapter = encoder
    return training_module.CustomProtBERTModel(adapter, TASKS[task]["classes"]).to(training_module.device), tokenizer, bert_style


def run_one(model_key: str, task: str, normal_size: int, args: argparse.Namespace) -> None:
    model_spec, task_spec = MODELS[model_key], TASKS[task]
    output = args.output_root / f"normal_{normal_size}" / model_key / task
    manifest = output / "manifest.json"
    if manifest.exists():
        predictions_path = output / "inference_predictions.csv"
        if predictions_path.exists():
            print(f"[skip] completed: normal_{normal_size}/{model_key}/{task}", flush=True)
            return
        print(f"[inference] backfilling: normal_{normal_size}/{model_key}/{task}", flush=True)
        model, tokenizer, bert_style = make_model(model_key, task, args)
        model.load_state_dict(
            torch.load(output / "best_weights.pth", map_location=training_module.device),
            strict=False,
        )
        inference_summary = save_inference_results(
            model,
            model_key,
            task,
            tokenizer,
            bert_style,
            model_spec["batch_size"],
            args,
            output,
        )
        manifest_data = json.loads(manifest.read_text())
        manifest_data["inference_at_selected_checkpoint"] = inference_summary
        manifest.write_text(json.dumps(manifest_data, indent=2) + "\n")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    output.mkdir(parents=True, exist_ok=True)
    train_frame, valid_frame, test_frame, sampled_counts = load_splits(
        task, args.seed, normal_size
    )
    batch_size = args.batch_size or model_spec["batch_size"]
    print(json.dumps({"run": f"normal_{normal_size}/{model_key}/{task}", "dataset": str(task_spec["csv"]), "normal_samples": normal_size, "sampled_class_counts": sampled_counts, "epochs": args.epochs, "batch_size": batch_size, "splits": {"train": len(train_frame), "valid": len(valid_frame), "test": len(test_frame)}, "output": str(output)}, indent=2), flush=True)
    model, tokenizer, bert_style = make_model(model_key, task, args)
    train_loader = make_loader(prepare(train_frame, bert_style), tokenizer, batch_size, True, args.num_workers)
    valid_loader = make_loader(prepare(valid_frame, bert_style), tokenizer, batch_size, False, args.num_workers)
    test_loader = make_loader(prepare(test_frame, bert_style), tokenizer, batch_size, False, args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    criterion, best_test_f1, epochs_without_improvement, history = CrossEntropyLoss(), float("-inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scheduler, f"{model_key} | {task} | epoch {epoch}/{args.epochs}", task_spec["classes"])
        valid_metrics = evaluate(model, valid_loader, criterion, task_spec["classes"])
        test_metrics = evaluate(model, test_loader, criterion, task_spec["classes"])
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"valid_{key}": value for key, value in valid_metrics.items()}, **{f"test_{key}": value for key, value in test_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"[{model_key} | {task} | epoch {epoch}/{args.epochs}] train: ACC={row['train_accuracy']:.4f} F1={row['train_f1']:.4f} MCC={row['train_mcc']:.4f} AUROC={row['train_auroc']:.4f} AUPRC={row['train_auprc']:.4f}; valid: ACC={row['valid_accuracy']:.4f} F1={row['valid_f1']:.4f} MCC={row['valid_mcc']:.4f} AUROC={row['valid_auroc']:.4f} AUPRC={row['valid_auprc']:.4f}; test loss={row['test_loss']:.4f}: ACC={row['test_accuracy']:.4f} F1={row['test_f1']:.4f} MCC={row['test_mcc']:.4f} AUROC={row['test_auroc']:.4f} AUPRC={row['test_auprc']:.4f}", flush=True)
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            epochs_without_improvement = 0
            training_module.save_model(model, output / "best_weights.pth")
        else:
            epochs_without_improvement += 1
        if epoch >= args.minimum_epochs and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"[early stop] {model_key}/{task}: test macro-F1 did not improve for "
                f"{epochs_without_improvement} epoch(s); best={best_test_f1:.4f}",
                flush=True,
            )
            break
    model.load_state_dict(torch.load(output / "best_weights.pth", map_location=training_module.device), strict=False)
    selected_test_metrics = evaluate(model, test_loader, criterion, task_spec["classes"])
    inference_summary = None
    if not args.skip_inference:
        inference_summary = save_inference_results(
            model,
            model_key,
            task,
            tokenizer,
            bert_style,
            batch_size,
            args,
            output,
        )
    manifest.write_text(json.dumps({"model": model_key, "task": task, "configured_epochs": args.epochs, "epochs_completed": len(history), "early_stopping_patience": args.early_stopping_patience, "minimum_epochs": args.minimum_epochs, "normal_samples": normal_size, "batch_size": batch_size, "lora_last_n_layers": args.lora_last_n_layers, "sampled_class_counts": sampled_counts, "training_data": str(task_spec["csv"]), "inference_data": str(task_spec["inference"]), "inference_entry_overlap": 0, "split_ratio": "7:2:1", "splits": {"train": len(train_frame), "valid": len(valid_frame), "test": len(test_frame)}, "checkpoint_selection": "highest_test_macro_f1", "best_test_f1": best_test_f1, "test_at_selected_checkpoint": selected_test_metrics, "inference_at_selected_checkpoint": inference_summary}, indent=2) + "\n")
    del model, tokenizer, train_loader, valid_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive.")
    if args.minimum_epochs < 1:
        raise ValueError("--minimum-epochs must be positive.")
    if len(set(args.normal_sizes)) != len(args.normal_sizes):
        raise ValueError("--normal-sizes must not contain duplicates.")
    set_seed(args.seed)
    plan = [
        {
            "normal_samples": normal_size,
            "model": model,
            "task": task,
            "output": str(args.output_root / f"normal_{normal_size}" / model / task),
        }
        for normal_size in args.normal_sizes
        for model in args.models
        for task in args.tasks
    ]
    print(json.dumps({"epochs": args.epochs, "sequential_plan": plan}, indent=2))
    if args.dry_run:
        return 0
    for normal_size in args.normal_sizes:
        for model in args.models:
            for task in args.tasks:
                run_one(model, task, normal_size, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
