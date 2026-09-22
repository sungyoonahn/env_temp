#!/usr/bin/env python3
"""BCL for SwissProt protein classification without sequence augmentation.

Each anchor is paired with a distinct random protein sequence from the same
class.  The pair replaces the two augmented image views in the original BCL.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from losses.BCL_loss import BalancedSupConLoss, LogitAdjustedCrossEntropy

ROOT = Path(__file__).resolve().parent
TASKS = {
    "CC": (ROOT / "datasets/Swissprot/cellular_component_trainable.csv", 6),
    "BP": (ROOT / "datasets/Swissprot/biological_process_trainable.csv", 5),
    "MF": (ROOT / "datasets/Swissprot/molecular_function_trainable.csv", 3),
}
MODELS = {
    "ProtBERT": ("Rostlab/prot_bert_bfd", True, 16),
    "ESM2_650M": ("facebook/esm2_t33_650M_UR50D", False, 4),
    "ESM2_3B": ("facebook/esm2_t36_3B_UR50D", False, 4),
}


def format_sequence(sequence: str, bert_style: bool) -> str:
    sequence = str(sequence).translate(str.maketrans({"O": "X", "B": "X", "U": "X", "Z": "X"}))
    return " ".join(sequence) if bert_style else sequence


class SameClassPairDataset(Dataset):
    """An anchor and a *different* random member of its class per item."""

    def __init__(self, frame: pd.DataFrame, bert_style: bool) -> None:
        self.sequences = frame.Sequence.astype(str).tolist()
        self.labels = frame.label.astype(int).tolist()
        self.bert_style = bert_style
        self.indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.indices_by_label[label].append(index)
        singletons = [label for label, items in self.indices_by_label.items() if len(items) < 2]
        if singletons:
            raise ValueError(f"BCL needs two distinct training sequences per class; singletons: {sorted(singletons)}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        label, positive_index = self.labels[index], index
        while positive_index == index:
            positive_index = random.choice(self.indices_by_label[label])
        return {"anchor": format_sequence(self.sequences[index], self.bert_style),
                "positive": format_sequence(self.sequences[positive_index], self.bert_style), "label": label}


class ProteinDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, bert_style: bool) -> None:
        self.sequences, self.labels, self.bert_style = frame.Sequence.astype(str).tolist(), frame.label.astype(int).tolist(), bert_style

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"sequence": format_sequence(self.sequences[index], self.bert_style), "label": self.labels[index]}


def pair_collate(tokenizer):
    def collate(samples):
        tokenize = lambda field: tokenizer([item[field] for item in samples], padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return {"anchor": tokenize("anchor"), "positive": tokenize("positive"), "labels": torch.tensor([item["label"] for item in samples])}
    return collate


def single_collate(tokenizer):
    def collate(samples):
        return {"tokens": tokenizer([item["sequence"] for item in samples], padding=True, truncation=True, max_length=1024, return_tensors="pt"), "labels": torch.tensor([item["label"] for item in samples])}
    return collate


class BCLProteinClassifier(nn.Module):
    """Sequence encoder with the BCL projection head and class centres."""

    def __init__(self, encoder: nn.Module, num_classes: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        config = getattr(encoder, "config", None)
        hidden_size = getattr(config, "hidden_size", getattr(config, "embed_dim", None))
        if hidden_size is None and hasattr(encoder, "get_base_model"):
            config = encoder.get_base_model().config
            hidden_size = getattr(config, "hidden_size", getattr(config, "embed_dim", None))
        if hidden_size is None:
            raise ValueError("Could not determine encoder hidden size.")
        self.dropout, self.classifier = nn.Dropout(0.2), nn.Linear(hidden_size, num_classes)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, feature_dim))
        self.center_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, feature_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(self.dropout(pooled))
        features = nn.functional.normalize(self.projection(pooled), dim=1)
        centers = nn.functional.normalize(self.center_projection(self.classifier.weight), dim=1)
        return logits, features, centers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="CC")
    parser.add_argument("--csv", type=Path, help="Optional replacement for the task's trainable CSV.")
    parser.add_argument("--model", choices=MODELS, default="ESM2_650M")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=1.0, help="Logit-adjusted CE weight.")
    parser.add_argument("--beta", type=float, default=0.35, help="BCL loss weight.")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument(
        "--lora-last-n-layers",
        type=int,
        default=10,
        help=(
            "Apply Q/K/V LoRA only to the final N encoder Transformer layers "
            "(default: 10). For ESM2-650M, 10 selects "
            "zero-based layers 23-32 / human-readable layers 24-33."
        ),
    )
    parser.add_argument("--normal-size", type=int, help="Optional count to retain for label 0 before splitting.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bcl")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, logits) -> dict[str, float]:
    actual, predicted = torch.cat(labels).numpy(), torch.cat(logits).argmax(1).numpy()
    return {"accuracy": float(accuracy_score(actual, predicted)), "f1": float(f1_score(actual, predicted, average="macro", zero_division=0)), "mcc": float(matthews_corrcoef(actual, predicted))}


def load_splits(args):
    default_csv, num_classes = TASKS[args.task]
    frame = pd.read_csv(args.csv or default_csv).copy()
    if {"Sequence", "label"}.difference(frame.columns): raise ValueError("CSV must contain Sequence and label columns.")
    frame.label = frame.label.astype(int)
    if frame.label.min() != 0 or frame.label.max() >= num_classes: raise ValueError(f"Labels must be in [0, {num_classes - 1}].")
    if args.normal_size is not None:
        normal = frame[frame.label.eq(0)]
        if not 1 <= args.normal_size <= len(normal): raise ValueError("--normal-size is out of range.")
        frame = pd.concat((normal.sample(args.normal_size, random_state=args.seed), frame[~frame.label.eq(0)]), ignore_index=True)
    train, held_out = train_test_split(frame, test_size=.30, random_state=args.seed, stratify=frame.label)
    valid, test = train_test_split(held_out, test_size=1/3, random_state=args.seed, stratify=held_out.label)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True), num_classes


def load_encoder(model_key):
    name, bert_style, batch_size = MODELS[model_key]
    if bert_style: return BertModel.from_pretrained(name), BertTokenizer.from_pretrained(name, do_lower_case=False), bert_style, batch_size
    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name, do_lower_case=False), bert_style, batch_size


def final_layer_lora_targets(encoder: nn.Module, last_n_layers: int) -> tuple[list[str], int, list[int]]:
    """Return exact Q/K/V module names for the final requested encoder layers."""
    config = getattr(encoder, "config", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine the number of encoder Transformer layers.")
    if not 0 <= last_n_layers <= num_layers:
        raise ValueError(
            f"--lora-last-n-layers must be between 0 and {num_layers}, got {last_n_layers}."
        )
    layer_indices = list(range(num_layers - last_n_layers, num_layers))
    targets = [
        f"encoder.layer.{layer_index}.attention.self.{projection}"
        for layer_index in layer_indices
        for projection in ("query", "key", "value")
    ]
    return targets, num_layers, layer_indices


def train_epoch(model, loader, optimizer, ce_criterion, scl_criterion, device, alpha, beta):
    model.train(); collected = {"loss": [], "ce_loss": [], "scl_loss": []}; labels, logits = [], []
    for batch in tqdm(loader, desc="Train", unit="batch", dynamic_ncols=True):
        optimizer.zero_grad(set_to_none=True); target = batch["labels"].to(device)
        anchor = {key: value.to(device) for key, value in batch["anchor"].items()}; positive = {key: value.to(device) for key, value in batch["positive"].items()}
        anchor_logits, anchor_features, centers = model(**anchor); positive_logits, positive_features, _ = model(**positive)
        ce_loss = .5 * (ce_criterion(anchor_logits, target) + ce_criterion(positive_logits, target))
        scl_loss = scl_criterion(centers, torch.stack((anchor_features, positive_features), dim=1), target)
        loss = alpha * ce_loss + beta * scl_loss; loss.backward(); optimizer.step()
        for name, value in (("loss", loss), ("ce_loss", ce_loss), ("scl_loss", scl_loss)): collected[name].append(float(value.detach()))
        labels.append(target.detach().cpu()); logits.append(anchor_logits.detach().cpu())
    return {key: float(np.mean(value)) for key, value in collected.items()} | compute_metrics(labels, logits)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); losses, labels, logits_list = [], [], []
    for batch in loader:
        tokens, target = {key: value.to(device) for key, value in batch["tokens"].items()}, batch["labels"].to(device)
        logits, _, _ = model(**tokens); losses.append(float(criterion(logits, target))); labels.append(target.cpu()); logits_list.append(logits.cpu())
    return {"loss": float(np.mean(losses))} | compute_metrics(labels, logits_list)


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    set_seed(args.seed); train_frame, valid_frame, test_frame, num_classes = load_splits(args)
    encoder, tokenizer, bert_style, default_batch_size = load_encoder(args.model); device = torch.device(args.device)
    lora_targets, total_encoder_layers, lora_layer_indices = final_layer_lora_targets(
        encoder, args.lora_last_n_layers
    )
    if lora_targets:
        encoder = get_peft_model(
            encoder,
            LoraConfig(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=.1,
                target_modules=lora_targets,
            ),
        )
    else:
        # The classification and BCL heads are added below and remain trainable.
        encoder.requires_grad_(False)
    model = BCLProteinClassifier(encoder, num_classes, args.feature_dim).to(device); batch_size = args.batch_size or default_batch_size
    class_counts = train_frame.label.value_counts().sort_index().reindex(range(num_classes), fill_value=0).tolist()
    if min(class_counts) < 2: raise ValueError("Each class needs at least two training sequences for same-class BCL pairs.")
    loader_args = {"batch_size": batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(SameClassPairDataset(train_frame, bert_style), shuffle=True, collate_fn=pair_collate(tokenizer), **loader_args)
    valid_loader = DataLoader(ProteinDataset(valid_frame, bert_style), shuffle=False, collate_fn=single_collate(tokenizer), **loader_args)
    test_loader = DataLoader(ProteinDataset(test_frame, bert_style), shuffle=False, collate_fn=single_collate(tokenizer), **loader_args)
    ce_criterion, scl_criterion = LogitAdjustedCrossEntropy(class_counts).to(device), BalancedSupConLoss(args.temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate); output = args.output_dir / args.task / args.model; output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "task": args.task,
        "model": args.model,
        "class_counts": class_counts,
        "pairing": "distinct same-class protein sequences",
        "lora_encoder_layers_total": total_encoder_layers,
        "lora_layer_indices_zero_based": lora_layer_indices,
        "lora_layer_numbers_one_based": [index + 1 for index in lora_layer_indices],
        "lora_target_modules": lora_targets,
    }, indent=2))
    best_f1, history = float("-inf"), []
    for epoch in range(1, args.epochs + 1):
        train_results = train_epoch(model, train_loader, optimizer, ce_criterion, scl_criterion, device, args.alpha, args.beta)
        valid_results, test_results = evaluate(model, valid_loader, ce_criterion, device), evaluate(model, test_loader, ce_criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_results.items()}, **{f"valid_{k}": v for k, v in valid_results.items()}, **{f"test_{k}": v for k, v in test_results.items()}}
        history.append(row); pd.DataFrame(history).to_csv(output / "history.csv", index=False); print(json.dumps(row))
        if valid_results["f1"] > best_f1:
            best_f1 = valid_results["f1"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "class_counts": class_counts}, output / "best_bcl_weights.pt")
    print(f"Best validation macro-F1: {best_f1:.4f}; checkpoint: {output / 'best_bcl_weights.pt'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
