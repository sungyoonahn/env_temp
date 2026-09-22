"""Sequence-level ESM-2 LoRA training for the non-cached TACS path."""

from __future__ import annotations

import copy
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .background_sampling import rotating_background_epoch
from .metrics import bootstrap_confidence_intervals
from .models import EmbeddingClassifier
from .trainer import classification_metrics, resolve_device, set_reproducible_seed


class SequenceDataset(Dataset):
    """Protein sequences with hard or reliability-weighted multiclass labels."""

    def __init__(self, frame: pd.DataFrame):
        required = {"sequence", "label", "weight"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Sequence training frame is missing: {sorted(missing)}")
        self.sequences = frame["sequence"].astype(str).tolist()
        self.labels = frame["label"].to_numpy(dtype=np.int64)
        self.weights = frame["weight"].to_numpy(dtype=np.float32)
        self.row_ids = (
            frame["_row_id"].to_numpy(dtype=np.int64)
            if "_row_id" in frame.columns
            else np.arange(len(frame), dtype=np.int64)
        )
        if not len(self.sequences):
            raise ValueError("Cannot train LoRA with no protein sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return (
            self.sequences[index],
            int(self.labels[index]),
            float(self.weights[index]),
            int(self.row_ids[index]),
        )


def _collate_sequences(batch):
    sequences, labels, weights, row_ids = zip(*batch, strict=True)
    return (
        list(sequences),
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(weights, dtype=torch.float32),
        torch.as_tensor(row_ids, dtype=torch.long),
    )


class ESM2LoRAClassifier(nn.Module):
    """Frozen ESM-2 backbone plus trainable LoRA attention adapters and head."""

    def __init__(self, model_config: Mapping[str, Any], lora_config: Mapping[str, Any], num_classes: int):
        super().__init__()
        try:
            # The OT environment has a partially upgraded NumPy installation:
            # package metadata reports NumPy 2 while its import surface is 1.x.
            # Current Accelerate selects ``numpy._core.multiarray`` from the
            # metadata alone, so load that compatibility module before PEFT
            # imports Transformers/Accelerate. On normal NumPy installs this
            # is a harmless no-op.
            try:  # pragma: no cover - environment-specific compatibility
                import numpy._core.multiarray  # noqa: F401
            except ModuleNotFoundError:
                pass
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("LoRA mode requires transformers and peft in the selected environment.") from exc

        requested_device = str(model_config.get("device", "cuda"))
        self.device = resolve_device(requested_device)
        self.max_length = int(model_config["max_length"])
        self.mixed_precision = str(lora_config.get("mixed_precision", "bf16"))
        if self.mixed_precision not in {"bf16", "fp16", "none"}:
            raise ValueError("lora.mixed_precision must be bf16, fp16, or none.")
        dtype = None
        if self.device.type == "cuda" and self.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.device.type == "cuda" and self.mixed_precision == "fp16":
            dtype = torch.float16

        model_name = str(model_config["esm_model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        targets = list(lora_config.get("target_modules", ["query", "value"]))
        if not targets:
            raise ValueError("lora.target_modules must name at least one ESM attention module.")
        adapter = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=targets,
            bias="none",
        )
        self.encoder = get_peft_model(backbone, adapter)
        if bool(lora_config.get("gradient_checkpointing", True)):
            self.encoder.gradient_checkpointing_enable()
            enable_inputs = getattr(self.encoder, "enable_input_require_grads", None)
            if enable_inputs is not None:
                enable_inputs()
        self.classifier = EmbeddingClassifier(
            int(backbone.config.hidden_size),
            num_classes,
            classifier=str(model_config["classifier"]),
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
        )
        self.special_ids = tuple(int(token_id) for token_id in self.tokenizer.all_special_ids)
        self.to(self.device)

    def load_saved_adapter(self, adapter_dir: str | Path, checkpoint_path: str | Path, labels: Sequence[int]) -> None:
        """Restore a saved PEFT adapter and its separately stored classifier head.

        TACS saves the PEFT adapter in a directory and the MLP head in a small
        ``model_round_*.pt`` file. Both must be restored to continue a round
        rather than accidentally starting again from the base ESM-2 weights.
        """
        adapter_dir = Path(adapter_dir)
        checkpoint_path = Path(checkpoint_path)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"LoRA classifier checkpoint does not exist: {checkpoint_path}")
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Restoring a LoRA checkpoint requires peft.") from exc

        # Add the saved adapter to the existing PEFT model and make it the sole
        # active, trainable adapter. Wrapping get_base_model() in a second
        # PeftModel would create nested adapters and is not a faithful resume.
        resume_adapter = "resume_round"
        self.encoder.load_adapter(
            str(adapter_dir),
            adapter_name=resume_adapter,
            is_trainable=True,
        )
        self.encoder.set_adapter(resume_adapter)
        self.encoder.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_labels = tuple(int(label) for label in checkpoint.get("labels", []))
        expected_labels = tuple(int(label) for label in labels)
        if saved_labels != expected_labels:
            raise RuntimeError(
                f"Checkpoint labels {saved_labels} do not match this run's label space {expected_labels}."
            )
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        self.classifier.to(self.device)

    def autocast(self):
        if self.device.type != "cuda" or self.mixed_precision == "none":
            return nullcontext()
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _tokens(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def _pool(self, last_hidden: torch.Tensor, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        residue_mask = encoded["attention_mask"].bool()
        for token_id in self.special_ids:
            residue_mask &= encoded["input_ids"].ne(token_id)
        empty_rows = residue_mask.sum(dim=1).eq(0)
        if empty_rows.any():
            residue_mask[empty_rows] = encoded["attention_mask"][empty_rows].bool()
        pooled = (last_hidden * residue_mask.unsqueeze(-1)).sum(dim=1)
        return pooled / residue_mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward_sequences(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._tokens(sequences)
        hidden = self.encoder(**encoded).last_hidden_state
        pooled = self._pool(hidden, encoded)
        return self.classifier(pooled.float()), pooled

    @torch.inference_mode()
    def predict_sequences(
        self, sequences: Sequence[str], batch_size: int, progress_description: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("LoRA inference batch size must be positive.")
        self.eval()
        probabilities = []
        embeddings = []
        starts = range(0, len(sequences), batch_size)
        iterator = (
            tqdm(
                starts,
                total=(len(sequences) + batch_size - 1) // batch_size,
                desc=progress_description,
                unit="batch",
                dynamic_ncols=True,
                mininterval=2.0,
                leave=True,
            )
            if progress_description
            else starts
        )
        for start in iterator:
            with self.autocast():
                logits, pooled = self.forward_sequences(sequences[start : start + batch_size])
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
            embeddings.append(pooled.float().cpu())
        return torch.cat(probabilities, dim=0), torch.cat(embeddings, dim=0)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state(self, state: Mapping[str, torch.Tensor]) -> None:
        current = dict(self.named_parameters())
        if set(state) != {name for name, parameter in current.items() if parameter.requires_grad}:
            raise RuntimeError("LoRA trainable parameter set changed while restoring the best checkpoint.")
        for name, value in state.items():
            current[name].data.copy_(value.to(device=current[name].device, dtype=current[name].dtype))

    def save_adapter(self, output: Path, round_index: int, config: Mapping[str, Any], labels: Sequence[int]) -> None:
        adapter_dir = output / f"lora_adapter_round_{round_index}"
        self.encoder.save_pretrained(adapter_dir, safe_serialization=True)
        torch.save(
            {
                "round": int(round_index),
                "classifier_state_dict": self.classifier.state_dict(),
                "labels": list(labels),
                "model_config": dict(config["model"]),
                "lora_config": dict(config["lora"]),
            },
            output / f"model_round_{round_index}.pt",
        )


@dataclass
class LoRAFitResult:
    model: ESM2LoRAClassifier
    best_validation: dict[str, Any]
    history: list[dict[str, Any]]


def _loader(frame: pd.DataFrame, num_classes: int, training_config: Mapping[str, Any], lora_config: Mapping[str, Any], seed: int, device: torch.device) -> DataLoader:
    dataset = SequenceDataset(frame)
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "batch_size": int(lora_config["train_batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "collate_fn": _collate_sequences,
        "generator": generator,
        "pin_memory": device.type == "cuda",
    }
    strategy = str(training_config["sampling_strategy"])
    if strategy == "none":
        return DataLoader(dataset, shuffle=True, **kwargs)
    counts = np.bincount(dataset.labels, minlength=num_classes)
    exponent = 1.0 if strategy == "class_balanced" else 0.5
    probabilities = np.maximum(counts[dataset.labels], 1).astype(np.float64) ** (-exponent)
    sampler = WeightedRandomSampler(
        torch.as_tensor(probabilities, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, sampler=sampler, **kwargs)


def evaluate_lora_classifier(
    model: ESM2LoRAClassifier,
    frame: pd.DataFrame,
    num_classes: int,
    inference_batch_size: int,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    progress_description: str | None = None,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities, _ = model.predict_sequences(
        frame["sequence"].astype(str).tolist(), inference_batch_size, progress_description
    )
    predictions = probabilities.argmax(dim=1).numpy()
    loss = nn.functional.nll_loss(
        torch.log(probabilities.clamp_min(1e-12)), torch.as_tensor(labels, dtype=torch.long)
    ).item()
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


def fit_lora_classifier(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    num_classes: int,
    model_config: Mapping[str, Any],
    lora_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    seed: int,
    history_path: str | Path | None = None,
    initial_model: ESM2LoRAClassifier | None = None,
) -> LoRAFitResult:
    """Train only LoRA attention adapters plus the classification head.

    ``initial_model`` is used only by an explicit resume path. Normal TACS
    rounds still create a new base ESM-2 model exactly as before.
    """
    set_reproducible_seed(seed)
    model = initial_model or ESM2LoRAClassifier(model_config, lora_config, num_classes)
    indexed_train_frame = train_frame.reset_index(drop=True).copy()
    indexed_train_frame["_row_id"] = np.arange(len(indexed_train_frame), dtype=np.int64)
    background_sampling = dict(training_config.get("background_sampling", {"strategy": "all"}))
    rotating_background = background_sampling.get("strategy", "all") == "rotating_hard_negative"
    base_loader = (
        None
        if rotating_background
        else _loader(
            indexed_train_frame,
            num_classes,
            training_config,
            lora_config,
            seed,
            model.device,
        )
    )
    hardness_sum = np.zeros(len(indexed_train_frame), dtype=np.float64)
    hardness_count = np.zeros(len(indexed_train_frame), dtype=np.int64)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA or classifier parameters are trainable.")
    optimizer = AdamW(
        trainable,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation = int(lora_config["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("lora.gradient_accumulation_steps must be at least one.")
    max_grad_norm = float(lora_config.get("max_grad_norm", 1.0))
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_state = model.trainable_state()
    best_validation: dict[str, Any] = {"macro_f1": float("-inf"), "loss": float("inf")}
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    minimum_epochs = int(training_config.get("minimum_epochs_before_early_stopping", 1))
    history_handle = None
    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_handle = history_path.open("w", encoding="utf-8")

    try:
        for epoch in range(1, int(training_config["epochs"]) + 1):
            sampling_metadata: dict[str, Any] = {}
            if rotating_background:
                epoch_frame, sampling_metadata = rotating_background_epoch(
                    indexed_train_frame,
                    epoch,
                    background_sampling,
                    seed,
                    hardness_sum,
                    hardness_count,
                )
                epoch_training_config = dict(training_config)
                epoch_training_config["sampling_strategy"] = "none"
                loader = _loader(
                    epoch_frame,
                    num_classes,
                    epoch_training_config,
                    lora_config,
                    seed + epoch,
                    model.device,
                )
                print(
                    json.dumps(
                        {"phase": "background_sampling", "epoch": epoch, **sampling_metadata}
                    ),
                    flush=True,
                )
            else:
                if base_loader is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("The standard LoRA data loader was not initialized.")
                loader = base_loader
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            progress = tqdm(
                loader, total=len(loader), desc=f"LoRA epoch {epoch}", unit="batch",
                dynamic_ncols=True, mininterval=2.0, leave=True,
            )
            for step, (sequences, labels, weights, row_ids) in enumerate(progress, start=1):
                device_labels = labels.to(model.device)
                with model.autocast():
                    logits, _ = model.forward_sequences(sequences)
                    float_logits = logits.float()
                    per_sample = criterion(float_logits, device_labels)
                    objective = (per_sample * weights.to(model.device)).mean()
                    loss = objective / accumulation
                if rotating_background:
                    background_label = int(background_sampling["background_label"])
                    background_mask = device_labels.eq(background_label)
                    if background_mask.any():
                        background_hardness = (
                            1.0
                            - torch.softmax(float_logits.detach(), dim=1)[
                                background_mask, background_label
                            ]
                        ).cpu().numpy()
                        background_row_ids = row_ids[labels.eq(background_label)].numpy()
                        np.add.at(hardness_sum, background_row_ids, background_hardness)
                        np.add.at(hardness_count, background_row_ids, 1)
                loss.backward()
                if step % accumulation == 0 or step == len(loader):
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(objective.detach().cpu()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}")
            progress.close()
            validation = evaluate_lora_classifier(
                model,
                validation_frame,
                num_classes,
                int(lora_config["inference_batch_size"]),
                progress_description=f"Validation epoch {epoch}",
            )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **sampling_metadata,
                **{f"validation_{key}": value for key, value in validation.items() if key != "confusion_matrix"},
            }
            history.append(row)
            if history_handle is not None:
                history_handle.write(json.dumps(row) + "\n")
                history_handle.flush()
            if validation["macro_f1"] > best_validation["macro_f1"] + 1e-12:
                best_state = model.trainable_state()
                best_validation = validation
                stale_epochs = 0
            else:
                if epoch > minimum_epochs:
                    stale_epochs += 1
                else:
                    stale_epochs = 0
                if stale_epochs >= int(training_config["early_stopping_patience"]):
                    print(
                        json.dumps(
                            {
                                "phase": "early_stopping",
                                "epoch": epoch,
                                "stale_epochs": stale_epochs,
                                "minimum_epochs": minimum_epochs,
                            }
                        ),
                        flush=True,
                    )
                    break
    finally:
        if history_handle is not None:
            history_handle.close()
    model.load_trainable_state(best_state)
    return LoRAFitResult(model=model, best_validation=best_validation, history=history)
