"""Weighted embedding-classifier training and Swiss-only evaluation metrics."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .models import EmbeddingClassifier
from .metrics import bootstrap_confidence_intervals


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but unavailable; classifier training will use CPU.")
        return torch.device("cpu")
    return torch.device(requested)


class WeightedEmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray, weights: np.ndarray):
        if len(embeddings) != len(labels) or len(labels) != len(weights):
            raise ValueError("Embedding, label, and weight arrays must align.")
        self.embeddings = embeddings.float().contiguous()
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.embeddings[index], self.labels[index], self.weights[index]


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
    recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positive.sum() / max(len(labels), 1)),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate_classifier(
    model: EmbeddingClassifier,
    embeddings: torch.Tensor,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    batch_size: int = 4096,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    probabilities = model.predict_proba(embeddings, device=device, batch_size=batch_size)
    logits = torch.log(probabilities.clamp_min(1e-12))
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    loss = nn.functional.nll_loss(logits, label_tensor).item()
    predictions = probabilities.argmax(dim=1).numpy()
    result = classification_metrics(labels, predictions, num_classes)
    result["loss"] = float(loss)
    result.update(
        bootstrap_confidence_intervals(
            labels,
            predictions,
            num_classes,
            int(bootstrap_samples),
            float(confidence_level),
            int(bootstrap_seed),
            classification_metrics,
        )
    )
    return result


@dataclass
class FitResult:
    model: EmbeddingClassifier
    best_validation: dict[str, Any]
    history: list[dict[str, Any]]


def fit_classifier(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_embeddings: torch.Tensor,
    validation_labels: np.ndarray,
    num_classes: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    seed: int,
    history_path: str | Path | None = None,
) -> FitResult:
    """Fit on frozen embeddings with the requested mean weighted CE loss."""
    if len(train_embeddings) == 0:
        raise ValueError("Cannot train TACS classifier with no training examples.")
    set_reproducible_seed(seed)
    device = resolve_device(str(model_config["device"]))
    model = EmbeddingClassifier(
        int(train_embeddings.shape[1]),
        num_classes,
        classifier=str(model_config["classifier"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    dataset = WeightedEmbeddingDataset(train_embeddings, train_labels, train_weights)
    generator = torch.Generator().manual_seed(seed)
    sampling_strategy = str(training_config["sampling_strategy"])
    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "generator": generator,
        "pin_memory": device.type == "cuda",
    }
    if sampling_strategy == "none":
        loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    else:
        counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=num_classes)
        exponent = 1.0 if sampling_strategy == "class_balanced" else 0.5
        sample_probabilities = np.maximum(counts[train_labels], 1).astype(np.float64) ** (-exponent)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_probabilities, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_state = copy.deepcopy(model.state_dict())
    best_validation: dict[str, Any] = {"macro_f1": float("-inf"), "loss": float("inf")}
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    history_handle = None
    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_handle = history_path.open("w", encoding="utf-8")

    try:
        for epoch in range(1, int(training_config["epochs"]) + 1):
            model.train()
            losses = []
            for embeddings, labels, weights in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(embeddings.to(device))
                per_sample = criterion(logits, labels.to(device))
                # This is intentionally mean(weight_i * CE_i), matching the TACS
                # specification.  It prevents weak labels from contributing full
                # supervised strength.
                loss = (per_sample * weights.to(device)).mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation = evaluate_classifier(
                model,
                validation_embeddings,
                validation_labels,
                num_classes,
                device,
            )
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"validation_{key}": value for key, value in validation.items() if key != "confusion_matrix"}}
            history.append(row)
            if history_handle is not None:
                history_handle.write(json.dumps(row) + "\n")
                history_handle.flush()
            improved = validation["macro_f1"] > best_validation["macro_f1"] + 1e-12
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(training_config["early_stopping_patience"]):
                    break
    finally:
        if history_handle is not None:
            history_handle.close()
    model.load_state_dict(best_state)
    return FitResult(model=model, best_validation=best_validation, history=history)


def save_evaluation(evaluation: Mapping[str, Any], output_dir: str | Path, stem: str, labels: list[int]) -> None:
    """Save overall, per-class, and confusion-matrix evaluation artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(dict(evaluation), indent=2) + "\n", encoding="utf-8")
    matrix = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
    np.savetxt(output_dir / f"{stem}_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    interval_keys = ("per_class_precision_ci", "per_class_recall_ci", "per_class_f1_ci")
    has_intervals = all(key in evaluation for key in interval_keys)
    rows = [
        "label,precision,recall,f1,precision_ci_low,precision_ci_high,recall_ci_low,recall_ci_high,f1_ci_low,f1_ci_high"
        if has_intervals else "label,precision,recall,f1"
    ]
    for index, (label, precision, recall, f1) in enumerate(zip(
        labels,
        evaluation["per_class_precision"],
        evaluation["per_class_recall"],
        evaluation["per_class_f1"],
        strict=True,
    )):
        row = f"{label},{precision:.8f},{recall:.8f},{f1:.8f}"
        if has_intervals:
            precision_ci = evaluation["per_class_precision_ci"][index]
            recall_ci = evaluation["per_class_recall_ci"][index]
            f1_ci = evaluation["per_class_f1_ci"][index]
            row += (
                f",{precision_ci[0]:.8f},{precision_ci[1]:.8f}"
                f",{recall_ci[0]:.8f},{recall_ci[1]:.8f}"
                f",{f1_ci[0]:.8f},{f1_ci[1]:.8f}"
            )
        rows.append(row)
    (output_dir / f"{stem}_per_class.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
