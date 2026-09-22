"""End-to-end Trusted-Anchor Curriculum Self-Training experiment runner."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .config import dump_config
from .data import iter_candidate_frames, load_swiss_dataset, make_swiss_splits, write_split_manifest
from .embeddings import ESM2MeanPoolEncoder, ShardedEmbeddingCache, dataframe_frames
from .neighbors import build_swiss_anchor
from .balancing import select_curriculum_candidates
from .reliability import score_predictions, thresholds_for_round
from .trainer import evaluate_classifier, fit_classifier, save_evaluation


TREMBl_ABLATIONS = {
    "all_trembl",
    "confidence_only",
    "confidence_neighborhood",
    "confidence_neighborhood_annotation",
    "full_tacs",
}


def _cache_weak_label_array(cache: ShardedEmbeddingCache) -> np.ndarray:
    """Read only weak labels once; embeddings remain streamed during scoring."""
    weak_labels = np.empty(cache.num_records, dtype=np.int64)
    for batch in cache.iter_batches(batch_size=4096):
        weak_labels[batch.indices] = batch.metadata["label"].to_numpy(dtype=np.int64)
    return weak_labels


def _ensure_contiguous_labels(labels: tuple[int, ...]) -> None:
    expected = tuple(range(len(labels)))
    if labels != expected:
        raise ValueError(
            f"TACS currently expects contiguous labels {expected}, but Swiss-Prot has {labels}. "
            "Remap the source labels before training."
        )


def _cache_or_build(
    directory: Path,
    frames,
    encoder: ESM2MeanPoolEncoder | None,
    config: Mapping[str, Any],
    source_name: str,
) -> tuple[ShardedEmbeddingCache, ESM2MeanPoolEncoder | None]:
    if ShardedEmbeddingCache.exists(directory):
        return ShardedEmbeddingCache(directory), encoder
    if encoder is None:
        encoder = ESM2MeanPoolEncoder(config["model"])
    cache = ShardedEmbeddingCache.create(
        directory,
        frames,
        encoder,
        int(config["model"]["cache_shard_size"]),
        {
            "source_name": source_name,
            "model_name": config["model"]["esm_model_name"],
            "max_length": config["model"]["max_length"],
            "encoder_mode": config["model"]["encoder_mode"],
        },
    )
    return cache, encoder


def _write_rows(path: Path, frame: pd.DataFrame, header: bool) -> None:
    frame.to_csv(path, index=False, mode="w" if header else "a", header=header)


def _round_swiss_indices(
    metadata: pd.DataFrame,
    round_index: int,
    rounds: int,
    ablation: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reveal each trusted Swiss class evenly over the curriculum rounds."""
    if rounds < 1:
        raise ValueError("training.rounds must be at least one.")
    seed_indices = np.flatnonzero(metadata["split"].eq("swiss_seed").to_numpy())
    candidates = np.flatnonzero(metadata["split"].eq("swiss_candidate").to_numpy())
    if ablation == "seed_only":
        return seed_indices, np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate_labels = metadata.iloc[candidates]["label"].to_numpy(dtype=np.int64)
    batches: list[list[np.ndarray]] = [[] for _ in range(rounds)]
    for label in np.unique(candidate_labels):
        class_indices = candidates[candidate_labels == label].copy()
        rng.shuffle(class_indices)
        for batch, class_batch in zip(batches, np.array_split(class_indices, rounds), strict=True):
            batch.append(class_batch)
    balanced_batches = []
    for batch in batches:
        indices = np.concatenate(batch).astype(np.int64) if batch else np.empty(0, dtype=np.int64)
        rng.shuffle(indices)
        balanced_batches.append(indices)
    active = np.concatenate([seed_indices, *balanced_batches[:round_index]]) if round_index else seed_indices
    new = balanced_batches[round_index - 1] if round_index else np.empty(0, dtype=np.int64)
    return active.astype(np.int64), new.astype(np.int64)


def _build_anchor(
    swiss_embeddings: torch.Tensor,
    swiss_metadata: pd.DataFrame,
    active_swiss_indices: np.ndarray,
    all_swiss_train_indices: np.ndarray,
    selected_trembl: pd.DataFrame,
    trembl_cache: ShardedEmbeddingCache | None,
    config: Mapping[str, Any],
):
    anchor_config = config["anchor"]
    if anchor_config["scope"] == "active_swiss":
        anchor_indices = active_swiss_indices
    elif anchor_config["scope"] == "all_swiss_train":
        anchor_indices = all_swiss_train_indices
    else:
        raise ValueError("anchor.scope must be active_swiss or all_swiss_train.")
    embeddings = swiss_embeddings[torch.as_tensor(anchor_indices)]
    labels = swiss_metadata.iloc[anchor_indices]["label"].to_numpy(dtype=np.int64)
    if anchor_config.get("include_high_confidence_trembl", False) and not selected_trembl.empty:
        if trembl_cache is None:
            raise RuntimeError("Cannot include TrEMBL anchors without a TrEMBL cache.")
        pseudo = trembl_cache.select(selected_trembl["cache_index"].to_numpy(dtype=np.int64))
        embeddings = torch.cat([embeddings, pseudo.embeddings], dim=0)
        labels = np.concatenate([labels, selected_trembl["training_label"].to_numpy(dtype=np.int64)])
    return build_swiss_anchor(embeddings, labels, anchor_config)


def _score_unused_trembl(
    model,
    trembl_cache: ShardedEmbeddingCache,
    anchor,
    selected_indices: set[int],
    labels: tuple[int, ...],
    round_config: Mapping[str, Any],
    ablation: str,
    anchor_config: Mapping[str, Any],
    device: torch.device,
    output_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Score every unused cached candidate and persist every reliability value."""
    score_values = np.full(trembl_cache.num_records, np.nan, dtype=np.float32)
    accepted = np.zeros(trembl_cache.num_records, dtype=bool)
    predictions = np.full(trembl_cache.num_records, -1, dtype=np.int64)
    stats = Counter()
    sums = Counter()
    first = True
    for batch in trembl_cache.iter_batches(batch_size=4096):
        available = np.fromiter((int(index) not in selected_indices for index in batch.indices), dtype=bool, count=len(batch.indices))
        if not available.any():
            continue
        cache_indices = batch.indices[available]
        metadata = batch.metadata.iloc[np.flatnonzero(available)].reset_index(drop=True)
        embeddings = batch.embeddings[torch.as_tensor(np.flatnonzero(available))]
        probabilities = model.predict_proba(embeddings, device=device).numpy()
        neighbors = anchor.query(embeddings, k=int(anchor_config["k"]))
        scored = score_predictions(
            probabilities,
            metadata["label"].to_numpy(dtype=np.int64),
            neighbors,
            len(labels),
            round_config,
            ablation,
            str(anchor_config["distance_reduction"]),
        )
        score_values[cache_indices] = scored["reliability"].to_numpy(dtype=np.float32)
        accepted[cache_indices] = scored["accepted"].to_numpy(dtype=bool)
        predictions[cache_indices] = scored["model_prediction"].to_numpy(dtype=np.int64)
        log_frame = pd.concat(
            [
                pd.DataFrame({"cache_index": cache_indices}),
                metadata.drop(columns=["sequence"], errors="ignore"),
                scored,
            ],
            axis=1,
        )
        _write_rows(output_csv, log_frame, header=first)
        first = False
        stats["scored"] += len(scored)
        stats["accepted_before_budget"] += int(scored["accepted"].sum())
        for column in ("confidence", "knn_agreement", "swiss_distance", "reliability"):
            sums[column] += float(scored[column].sum())
        for column in ("model_trembl_agreement", "model_knn_agreement", "triple_agreement"):
            sums[column] += float(scored[column].sum())
    if first:
        # Preserve a valid, inspectable artifact even if every candidate was used.
        pd.DataFrame(columns=["cache_index", "reliability", "accepted"]).to_csv(output_csv, index=False)
    denominator = max(stats["scored"], 1)
    aggregate = {
        "scored_trembl": float(stats["scored"]),
        "accepted_before_budget": float(stats["accepted_before_budget"]),
        "mean_model_confidence": sums["confidence"] / denominator,
        "mean_knn_agreement": sums["knn_agreement"] / denominator,
        "mean_swiss_distance": sums["swiss_distance"] / denominator,
        "mean_reliability": sums["reliability"] / denominator,
        "model_trembl_agreement_rate": sums["model_trembl_agreement"] / denominator,
        "model_knn_agreement_rate": sums["model_knn_agreement"] / denominator,
        "triple_agreement_rate": sums["triple_agreement"] / denominator,
    }
    return score_values, accepted, predictions, aggregate


def _assemble_training_data(
    swiss_embeddings: torch.Tensor,
    swiss_metadata: pd.DataFrame,
    active_swiss_indices: np.ndarray,
    selected_trembl: pd.DataFrame,
    trembl_cache: ShardedEmbeddingCache | None,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    swiss_positions = torch.as_tensor(active_swiss_indices, dtype=torch.long)
    embeddings = [swiss_embeddings[swiss_positions]]
    labels = [swiss_metadata.iloc[active_swiss_indices]["label"].to_numpy(dtype=np.int64)]
    weights = [np.full(len(active_swiss_indices), float(config["training"]["swiss_weight"]), dtype=np.float32)]
    if not selected_trembl.empty:
        if trembl_cache is None:
            raise RuntimeError("Selected TrEMBL rows require a TrEMBL embedding cache.")
        weak = trembl_cache.select(selected_trembl["cache_index"].to_numpy(dtype=np.int64))
        embeddings.append(weak.embeddings)
        labels.append(selected_trembl["training_label"].to_numpy(dtype=np.int64))
        if config["experiment"]["ablation"] == "full_tacs":
            reliability = selected_trembl["reliability"].to_numpy(dtype=np.float32)
            weights.append(float(config["training"]["trembl_base_weight"]) * np.clip(reliability, 0.0, 1.0))
        else:
            weights.append(
                np.full(len(selected_trembl), float(config["training"]["trembl_base_weight"]), dtype=np.float32)
            )
    return torch.cat(embeddings, dim=0), np.concatenate(labels), np.concatenate(weights)


def _save_checkpoint(model, output: Path, round_index: int, config: Mapping[str, Any], labels: tuple[int, ...]) -> None:
    torch.save(
        {
            "round": round_index,
            "state_dict": model.state_dict(),
            "labels": list(labels),
            "model_config": dict(config["model"]),
        },
        output / f"model_round_{round_index}.pt",
    )


def run_tacs(config: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Run one TACS experiment; the untouched Swiss test split is evaluated last."""
    if config["model"]["encoder_mode"] == "lora":
        from .lora_runner import run_lora_tacs

        return run_lora_tacs(config, dry_run=dry_run)
    experiment = config["experiment"]
    data_config = config["data"]
    task = str(data_config["task"])
    ablation = str(experiment["ablation"])
    output = Path(experiment["output_dir"]) / f"{experiment['name']}_{task}_{ablation}_seed{experiment['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    dump_config(config, output / "resolved_config.yaml")

    swiss, label_space = load_swiss_dataset(config)
    _ensure_contiguous_labels(label_space.labels)
    splits = make_swiss_splits(swiss, config)
    for split_name, frame in splits.items():
        frame.to_csv(output / f"{split_name}.csv", index=False)
    write_split_manifest(splits, label_space, output / "swiss_split_manifest.json")
    if dry_run:
        result = {
            "output": str(output),
            "task": task,
            "ablation": ablation,
            "swiss_rows": {name: len(frame) for name, frame in splits.items()},
            "trembl_candidates_exists": Path(data_config["trembl_candidates"]).is_file(),
        }
        (output / "dry_run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    swiss_frames = []
    for split_name, frame in splits.items():
        cached = frame.copy()
        cached["split"] = split_name
        swiss_frames.append(cached)
    all_swiss = pd.concat(swiss_frames, ignore_index=True)
    cache_root = output / "embedding_cache"
    encoder: ESM2MeanPoolEncoder | None = None
    swiss_cache, encoder = _cache_or_build(
        cache_root / "swiss",
        dataframe_frames(all_swiss, int(config["model"]["cache_shard_size"])),
        encoder,
        config,
        "swiss",
    )
    trembl_cache = None
    trembl_weak_labels: np.ndarray | None = None
    if ablation in TREMBl_ABLATIONS:
        candidate_path = Path(data_config["trembl_candidates"])
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"TrEMBL candidate pool is missing: {candidate_path}. Run scripts/prepare_trembl_candidates.py first."
            )
        trembl_cache, encoder = _cache_or_build(
            cache_root / "trembl",
            iter_candidate_frames(candidate_path, int(data_config["candidate_read_batch_size"])),
            encoder,
            config,
            "trembl",
        )

        trembl_weak_labels = _cache_weak_label_array(trembl_cache)

    swiss_cached = swiss_cache.load_all()
    swiss_metadata = swiss_cached.metadata
    validation_indices = np.flatnonzero(swiss_metadata["split"].eq("swiss_validation").to_numpy())
    test_indices = np.flatnonzero(swiss_metadata["split"].eq("swiss_test").to_numpy())
    all_swiss_train_indices = np.flatnonzero(
        swiss_metadata["split"].isin(["swiss_seed", "swiss_candidate"]).to_numpy()
    )
    validation_embeddings = swiss_cached.embeddings[torch.as_tensor(validation_indices)]
    validation_labels = swiss_metadata.iloc[validation_indices]["label"].to_numpy(dtype=np.int64)
    test_embeddings = swiss_cached.embeddings[torch.as_tensor(test_indices)]
    test_labels = swiss_metadata.iloc[test_indices]["label"].to_numpy(dtype=np.int64)
    labels = label_space.labels
    selected_trembl = pd.DataFrame(columns=["cache_index", "training_label", "weak_label", "reliability"])
    selected_indices: set[int] = set()
    rounds = int(config["training"]["rounds"])
    round_rows: list[dict[str, Any]] = []

    # Stage 0: only trusted Swiss-Prot seed supervision.
    active_swiss, _ = _round_swiss_indices(swiss_metadata, 0, rounds, ablation, int(experiment["seed"]))
    train_embeddings, train_labels, train_weights = _assemble_training_data(
        swiss_cached.embeddings,
        swiss_metadata,
        active_swiss,
        selected_trembl,
        trembl_cache,
        config,
    )
    fit = fit_classifier(
        train_embeddings,
        train_labels,
        train_weights,
        validation_embeddings,
        validation_labels,
        len(labels),
        config["model"],
        config["training"],
        int(experiment["seed"]),
        output / "round_0_epochs.jsonl",
    )
    model = fit.model
    _save_checkpoint(model, output, 0, config, labels)
    row0 = {
        "round": 0,
        "total_swiss_samples": int(len(active_swiss)),
        "new_swiss_samples": int(len(active_swiss)),
        "total_accepted_trembl": 0,
        "new_accepted_trembl": 0,
        "training_class_counts": {str(label): int(count) for label, count in zip(*np.unique(train_labels, return_counts=True), strict=True)},
        **{f"validation_{key}": value for key, value in fit.best_validation.items() if key != "confusion_matrix"},
    }
    round_rows.append(row0)

    if ablation != "seed_only":
        for round_index in range(1, rounds + 1):
            active_swiss, new_swiss = _round_swiss_indices(swiss_metadata, round_index, rounds, ablation, int(experiment["seed"]))
            score_summary: dict[str, float] = {}
            new_selected = pd.DataFrame(columns=selected_trembl.columns)
            if trembl_cache is not None:
                round_selection = thresholds_for_round(config["selection"], round_index)
                if ablation == "all_trembl":
                    round_selection["max_trembl_per_round"] = None
                    round_selection["balance_by_class"] = False
                anchor = _build_anchor(
                    swiss_cached.embeddings,
                    swiss_metadata,
                    active_swiss,
                    all_swiss_train_indices,
                    selected_trembl,
                    trembl_cache,
                    config,
                )
                score_values, accepted, predictions, score_summary = _score_unused_trembl(
                    model,
                    trembl_cache,
                    anchor,
                    selected_indices,
                    labels,
                    round_selection,
                    ablation,
                    config["anchor"],
                    next(iter(model.parameters())).device,
                    output / f"trembl_scores_round_{round_index}.csv",
                )
                selection_labels = trembl_weak_labels if round_selection["trembl_training_label"] == "annotation" else predictions
                if selection_labels is None:
                    raise RuntimeError("TrEMBL selection labels were not initialized.")
                chosen_indices = select_curriculum_candidates(
                    np.arange(trembl_cache.num_records, dtype=np.int64),
                    score_values,
                    accepted,
                    selection_labels,
                    len(labels),
                    round_selection,
                )
                if len(chosen_indices):
                    chosen = trembl_cache.select(chosen_indices)
                    weak_labels = chosen.metadata["label"].to_numpy(dtype=np.int64)
                    selected_labels = weak_labels if round_selection["trembl_training_label"] == "annotation" else predictions[chosen_indices]
                    new_selected = pd.DataFrame(
                        {
                            "cache_index": chosen_indices,
                            "training_label": selected_labels,
                            "weak_label": weak_labels,
                            "reliability": score_values[chosen_indices],
                        }
                    )
                    if selected_trembl.empty:
                        selected_trembl = new_selected.copy()
                    else:
                        selected_trembl = pd.concat([selected_trembl, new_selected], ignore_index=True)
                    selected_indices.update(int(index) for index in chosen_indices)
            train_embeddings, train_labels, train_weights = _assemble_training_data(
                swiss_cached.embeddings,
                swiss_metadata,
                active_swiss,
                selected_trembl,
                trembl_cache,
                config,
            )
            fit = fit_classifier(
                train_embeddings,
                train_labels,
                train_weights,
                validation_embeddings,
                validation_labels,
                len(labels),
                config["model"],
                config["training"],
                int(experiment["seed"]) + round_index,
                output / f"round_{round_index}_epochs.jsonl",
            )
            model = fit.model
            _save_checkpoint(model, output, round_index, config, labels)
            class_counts = {
                str(label): int(count)
                for label, count in selected_trembl["training_label"].value_counts().sort_index().items()
            }
            round_row = {
                "round": round_index,
                "total_swiss_samples": int(len(active_swiss)),
                "new_swiss_samples": int(len(new_swiss)),
                "total_accepted_trembl": int(len(selected_trembl)),
                "new_accepted_trembl": int(len(new_selected)),
                "accepted_trembl_by_class": class_counts,
                "new_accepted_trembl_by_class": {str(label): int(count) for label, count in new_selected["training_label"].value_counts().sort_index().items()},
                "training_class_counts": {str(label): int(count) for label, count in zip(*np.unique(train_labels, return_counts=True), strict=True)},
                **score_summary,
                **{f"validation_{key}": value for key, value in fit.best_validation.items() if key != "confusion_matrix"},
            }
            round_rows.append(round_row)
            selected_trembl.to_csv(output / "accepted_trembl.csv", index=False)

    evaluation = evaluate_classifier(
        model,
        test_embeddings,
        test_labels,
        len(labels),
        next(iter(model.parameters())).device,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        confidence_level=float(config["evaluation"]["confidence_level"]),
        bootstrap_seed=int(experiment["seed"]),
    )
    save_evaluation(evaluation, output, "swiss_test", list(labels))
    with (output / "round_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in round_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "output": str(output),
        "ablation": ablation,
        "swiss_test": evaluation,
        "rounds": round_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
