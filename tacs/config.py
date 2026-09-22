"""Configuration loading and validation for TACS experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "tacs",
        "seed": 42,
        "output_dir": "outputs/tacs",
        "ablation": "full_tacs",
        "retrain_from_scratch": True,
    },
    "data": {
        "task": "biological_process",
        "swiss_csv": "datasets/Swissprot/biological_process_trainable.csv",
        "trembl_source_csv": (
            "datasets/TREMBL/"
            "uniprotkb_taxonomy_id_2_AND_unreviewed_tr_2025_08_20_processed.csv"
        ),
        "trembl_candidates": "outputs/tacs/data/biological_process_trembl_candidates.parquet",
        "sequence_column": "Sequence",
        "id_column": "Entry",
        "label_column": "label",
        "keyword_column": None,
        "cluster_column": None,
        "conflict_policy": "specific_over_background",
        "background_label": 0,
        "drop_swiss_trembl_sequence_overlap": True,
        "candidate_read_batch_size": 10_000,
    },
    "split": {
        "method": "random",
        "seed_fraction": 0.20,
        "candidate_fraction": 0.60,
        "validation_fraction": 0.10,
        "test_fraction": 0.10,
        "min_validation_per_class": 0,
        "min_test_per_class": 0,
        "rare_class_max_size": None,
        "rare_class_seed_fraction": 0.50,
    },
    "model": {
        "esm_model_name": "facebook/esm2_t33_650M_UR50D",
        "encoder_mode": "frozen",
        "max_length": 1024,
        "embedding_batch_size": 4,
        "cache_shard_size": 5_000,
        "classifier": "mlp",
        "hidden_dim": 512,
        "dropout": 0.20,
        "device": "cuda",
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["query", "value"],
        "train_batch_size": 16,
        "inference_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "candidate_scan_per_round": 250_000,
        "log_every_steps": 50,
        "scan_log_every": 25_000,
    },
    "anchor": {
        "scope": "active_swiss",
        "include_high_confidence_trembl": False,
        "metric": "cosine",
        "k": 10,
        "distance_reduction": "mean",
    },
    "selection": {
        "strategy": "threshold_top_k",
        "confidence_threshold": 0.95,
        "knn_agreement_threshold": 0.90,
        "distance_threshold": None,
        "distance_temperature": 0.25,
        "require_model_knn_agreement": True,
        "require_trembl_model_agreement": True,
        "require_trembl_knn_agreement": False,
        "max_trembl_per_round": 10_000,
        "balance_by_class": True,
        "max_trembl_per_class_per_round": None,
        "excluded_trembl_labels": [],
        "percentile_per_round": None,
        "trembl_training_label": "annotation",
        "curriculum": [],
    },
    "training": {
        "rounds": 3,
        "swiss_candidate_fraction_per_round": None,
        "batch_size": 256,
        "epochs": 30,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "swiss_weight": 1.0,
        "trembl_base_weight": 0.30,
        "num_workers": 0,
        "sampling_strategy": "sqrt_inverse_frequency",
        "max_background_to_other_ratio": None,
        "minimum_epochs_before_early_stopping": 1,
        "background_sampling": {
            "strategy": "all",
            "background_label": 0,
            "background_to_non_background_ratio": 2.0,
            "coverage_epochs": 10,
            "hard_negative_fraction": 0.50,
        },
        "early_stopping_patience": 6,
    },
    "evaluation": {
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
    },
}


VALID_ABLATIONS = {
    "seed_only",
    "all_swiss",
    "all_trembl",
    "confidence_only",
    "confidence_neighborhood",
    "confidence_neighborhood_annotation",
    "full_tacs",
}


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge an experiment config into defaults."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_fractions(config: Mapping[str, Any]) -> None:
    split = config["split"]
    names = ("seed_fraction", "candidate_fraction", "validation_fraction", "test_fraction")
    total = sum(float(split[name]) for name in names)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Swiss split fractions must sum to 1.0, got {total}.")
    if any(float(split[name]) < 0 for name in names):
        raise ValueError("Swiss split fractions must all be non-negative.")
    if int(split.get("min_validation_per_class", 0)) < 0 or int(split.get("min_test_per_class", 0)) < 0:
        raise ValueError("Minimum per-class holdouts must be non-negative.")
    rare_max = split.get("rare_class_max_size")
    if rare_max is not None and int(rare_max) < 1:
        raise ValueError("split.rare_class_max_size must be positive or null.")
    rare_seed_fraction = float(split.get("rare_class_seed_fraction", 0.5))
    if not 0 < rare_seed_fraction < 1:
        raise ValueError("split.rare_class_seed_fraction must be between zero and one.")


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early for configurations that would violate TACS invariants."""
    required = {"experiment", "data", "split", "model", "anchor", "selection", "training"}
    if "evaluation" not in config:
        raise ValueError("Missing configuration section: evaluation")
    if int(config["evaluation"].get("bootstrap_samples", 0)) < 0:
        raise ValueError("evaluation.bootstrap_samples must be non-negative.")
    confidence_level = float(config["evaluation"].get("confidence_level", 0.95))
    if not 0 < confidence_level < 1:
        raise ValueError("evaluation.confidence_level must be between zero and one.")
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    _validate_fractions(config)
    encoder_mode = str(config["model"]["encoder_mode"])
    if encoder_mode not in {"frozen", "lora"}:
        raise ValueError("model.encoder_mode must be frozen or lora.")
    lora = config.get("lora", {})
    if int(lora.get("r", 0)) < 1 or int(lora.get("alpha", 0)) < 1:
        raise ValueError("lora.r and lora.alpha must be positive.")
    if int(lora.get("train_batch_size", 0)) < 1 or int(lora.get("inference_batch_size", 0)) < 1:
        raise ValueError("LoRA sequence batch sizes must be positive.")
    if int(lora.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("lora.gradient_accumulation_steps must be positive.")
    if str(lora.get("mixed_precision", "bf16")) not in {"bf16", "fp16", "none"}:
        raise ValueError("lora.mixed_precision must be bf16, fp16, or none.")
    scan_limit = lora.get("candidate_scan_per_round")
    if scan_limit is not None and int(scan_limit) < 1:
        raise ValueError("lora.candidate_scan_per_round must be positive or null.")
    if config["anchor"]["metric"] not in {"cosine", "euclidean"}:
        raise ValueError("anchor.metric must be 'cosine' or 'euclidean'.")
    if config["anchor"]["distance_reduction"] not in {"mean", "nearest"}:
        raise ValueError("anchor.distance_reduction must be 'mean' or 'nearest'.")
    if int(config["anchor"]["k"]) < 1:
        raise ValueError("anchor.k must be at least one.")
    if config["data"]["conflict_policy"] not in {"drop", "majority", "specific_over_background"}:
        raise ValueError("data.conflict_policy must be drop, majority, or specific_over_background.")
    if config["selection"]["strategy"] not in {"threshold_top_k", "top_percentile"}:
        raise ValueError("selection.strategy must be threshold_top_k or top_percentile.")
    if config["selection"]["trembl_training_label"] not in {"annotation", "prediction"}:
        raise ValueError("selection.trembl_training_label must be annotation or prediction.")
    if config["training"]["sampling_strategy"] not in {"none", "sqrt_inverse_frequency", "class_balanced"}:
        raise ValueError("training.sampling_strategy must be none, sqrt_inverse_frequency, or class_balanced.")
    background_ratio = config["training"].get("max_background_to_other_ratio")
    if background_ratio is not None and float(background_ratio) <= 0:
        raise ValueError("training.max_background_to_other_ratio must be positive or null.")
    epochs = int(config["training"]["epochs"])
    minimum_epochs = int(config["training"].get("minimum_epochs_before_early_stopping", 1))
    if not 1 <= minimum_epochs <= epochs:
        raise ValueError(
            "training.minimum_epochs_before_early_stopping must be between 1 and training.epochs."
        )
    background_sampling = config["training"].get("background_sampling", {})
    background_strategy = str(background_sampling.get("strategy", "all"))
    if background_strategy not in {"all", "rotating_hard_negative"}:
        raise ValueError(
            "training.background_sampling.strategy must be all or rotating_hard_negative."
        )
    sampling_background_label = background_sampling.get(
        "background_label", config["data"]["background_label"]
    )
    if not isinstance(sampling_background_label, int) or sampling_background_label < 0:
        raise ValueError("training.background_sampling.background_label must be non-negative.")
    sampling_ratio = float(background_sampling.get("background_to_non_background_ratio", 2.0))
    if sampling_ratio <= 0:
        raise ValueError(
            "training.background_sampling.background_to_non_background_ratio must be positive."
        )
    coverage_epochs = int(background_sampling.get("coverage_epochs", 1))
    if coverage_epochs < 1:
        raise ValueError("training.background_sampling.coverage_epochs must be positive.")
    hard_fraction = float(background_sampling.get("hard_negative_fraction", 0.5))
    if not 0 <= hard_fraction <= 1:
        raise ValueError(
            "training.background_sampling.hard_negative_fraction must be between zero and one."
        )
    if background_strategy == "rotating_hard_negative":
        if coverage_epochs > epochs:
            raise ValueError(
                "Rotating background coverage_epochs cannot exceed training.epochs."
            )
        if sampling_background_label != int(config["data"]["background_label"]):
            raise ValueError(
                "training.background_sampling.background_label must match data.background_label."
            )
        if minimum_epochs < coverage_epochs:
            raise ValueError(
                "Rotating background sampling requires minimum_epochs_before_early_stopping "
                "to cover all background shards."
            )
    per_class_cap = config["selection"].get("max_trembl_per_class_per_round")
    if per_class_cap is not None and int(per_class_cap) < 1:
        raise ValueError("selection.max_trembl_per_class_per_round must be positive or null.")
    excluded_trembl = config["selection"].get("excluded_trembl_labels", [])
    if not isinstance(excluded_trembl, list):
        raise ValueError("selection.excluded_trembl_labels must be a list of integer labels.")
    if any(not isinstance(label, int) or label < 0 for label in excluded_trembl):
        raise ValueError("selection.excluded_trembl_labels must contain non-negative integers.")
    if len(set(excluded_trembl)) != len(excluded_trembl):
        raise ValueError("selection.excluded_trembl_labels cannot contain duplicates.")
    if config["experiment"]["ablation"] not in VALID_ABLATIONS:
        choices = ", ".join(sorted(VALID_ABLATIONS))
        raise ValueError(f"Unknown experiment.ablation. Choose one of: {choices}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config, merge defaults, and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TACS config does not exist: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("PyYAML is required to read YAML TACS configs.") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("A TACS config must contain a mapping at its top level.")
    config = _merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a resolved experiment config as YAML when available, otherwise JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    except ImportError:
        path.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
