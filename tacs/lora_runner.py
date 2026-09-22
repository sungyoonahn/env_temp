"""Trusted-anchor curriculum self-training with sequence-level ESM-2 LoRA."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .balancing import select_curriculum_candidates
from .config import dump_config
from .data import iter_candidate_frames, load_swiss_dataset, make_swiss_splits, write_split_manifest
from .lora import ESM2LoRAClassifier, evaluate_lora_classifier, fit_lora_classifier
from .neighbors import build_swiss_anchor
from .reliability import score_predictions, thresholds_for_round
from .sampling import cap_background_indices
from .trainer import save_evaluation


TREMBl_ABLATIONS = {
    "all_trembl",
    "confidence_only",
    "confidence_neighborhood",
    "confidence_neighborhood_annotation",
    "full_tacs",
}


def _ensure_contiguous_labels(labels: tuple[int, ...]) -> None:
    expected = tuple(range(len(labels)))
    if labels != expected:
        raise ValueError(
            f"TACS currently expects contiguous labels {expected}, but Swiss-Prot has {labels}. "
            "Remap the source labels before training."
        )


def _round_swiss_indices(
    metadata: pd.DataFrame,
    round_index: int,
    rounds: int,
    ablation: str,
    seed: int,
    background_label: int,
    max_background_to_other_ratio: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reveal trusted Swiss candidate data evenly per class across rounds."""
    if rounds < 1:
        raise ValueError("training.rounds must be at least one.")
    seed_indices = np.flatnonzero(metadata["split"].eq("swiss_seed").to_numpy())
    candidates = np.flatnonzero(metadata["split"].eq("swiss_candidate").to_numpy())
    if ablation == "seed_only":
        active = cap_background_indices(
            metadata,
            seed_indices,
            background_label,
            max_background_to_other_ratio,
            seed + 10_000,
        )
        return active.astype(np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate_labels = metadata.iloc[candidates]["label"].to_numpy(dtype=np.int64)
    batches: list[list[np.ndarray]] = [[] for _ in range(rounds)]
    for label in np.unique(candidate_labels):
        class_indices = candidates[candidate_labels == label].copy()
        rng.shuffle(class_indices)
        for batch, class_batch in zip(batches, np.array_split(class_indices, rounds), strict=True):
            batch.append(class_batch)
    balanced = []
    for batch in batches:
        indices = np.concatenate(batch).astype(np.int64) if batch else np.empty(0, dtype=np.int64)
        rng.shuffle(indices)
        balanced.append(indices)
    active = np.concatenate([seed_indices, *balanced[:round_index]]) if round_index else seed_indices
    new = balanced[round_index - 1] if round_index else np.empty(0, dtype=np.int64)
    active = cap_background_indices(
        metadata,
        active,
        background_label,
        max_background_to_other_ratio,
        seed + 10_000 + round_index,
    )
    return active.astype(np.int64), new.astype(np.int64)


def _candidate_count(path: Path) -> int | None:
    if path.suffix.lower() != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Reading a parquet TrEMBL candidate pool requires pyarrow.") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def _hash_sample_mask(sequence_hashes: pd.Series, seed: int, round_index: int, total: int | None, scan_limit: int | None) -> np.ndarray:
    """Deterministically draw an approximately uniform scan budget from TrEMBL."""
    if scan_limit is None or total is None or scan_limit >= total:
        return np.ones(len(sequence_hashes), dtype=bool)
    if scan_limit < 1:
        raise ValueError("lora.candidate_scan_per_round must be positive or null.")
    threshold = int((int(scan_limit) / int(total)) * (1 << 64))
    prefix = f"{seed}:{round_index}:".encode("ascii")
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(prefix + value.encode("ascii"), digest_size=8).digest(), "big") < threshold
            for value in sequence_hashes.astype(str)
        ),
        dtype=bool,
        count=len(sequence_hashes),
    )


def _write_scores(path: Path, frame: pd.DataFrame, first: bool) -> None:
    frame.to_csv(path, index=False, mode="w" if first else "a", header=first)


def _anchor_for_round(
    model: ESM2LoRAClassifier,
    swiss_metadata: pd.DataFrame,
    active_indices: np.ndarray,
    all_train_indices: np.ndarray,
    selected_trembl: pd.DataFrame,
    config: Mapping[str, Any],
):
    anchor_config = config["anchor"]
    if anchor_config["scope"] == "active_swiss":
        positions = active_indices
    elif anchor_config["scope"] == "all_swiss_train":
        positions = cap_background_indices(
            swiss_metadata,
            all_train_indices,
            int(config["data"]["background_label"]),
            config["training"].get("max_background_to_other_ratio"),
            int(config["experiment"]["seed"]) + 20_000,
        )
    else:
        raise ValueError("anchor.scope must be active_swiss or all_swiss_train.")
    frame = swiss_metadata.iloc[positions]
    _, embeddings = model.predict_sequences(
        frame["sequence"].astype(str).tolist(),
        int(config["lora"]["inference_batch_size"]),
        progress_description=f"Swiss anchors R{len(active_indices)}",
    )
    anchor_labels = frame["label"].to_numpy(dtype=np.int64)
    if anchor_config.get("include_high_confidence_trembl", False) and not selected_trembl.empty:
        _, pseudo_embeddings = model.predict_sequences(
            selected_trembl["sequence"].astype(str).tolist(), int(config["lora"]["inference_batch_size"])
        )
        embeddings = torch.cat([embeddings, pseudo_embeddings], dim=0)
        anchor_labels = np.concatenate([anchor_labels, selected_trembl["training_label"].to_numpy(dtype=np.int64)])
    return build_swiss_anchor(embeddings, anchor_labels, anchor_config)


def _score_candidate_round(
    model: ESM2LoRAClassifier,
    candidate_path: Path,
    selected_indices: set[int],
    anchor,
    labels: tuple[int, ...],
    config: Mapping[str, Any],
    round_index: int,
    output_csv: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Stream a deterministic candidate subset, retaining only eligible records."""
    total = _candidate_count(candidate_path)
    scan_limit = config["lora"].get("candidate_scan_per_round")
    selection = thresholds_for_round(config["selection"], round_index)
    ablation = str(config["experiment"]["ablation"])
    if ablation == "all_trembl":
        selection["max_trembl_per_round"] = None
        selection["balance_by_class"] = False
    selected_records: list[pd.DataFrame] = []
    stats = Counter()
    sums = Counter()
    first = True
    offset = 0
    progress = tqdm(
        total=min(int(scan_limit), int(total)) if scan_limit is not None and total is not None else None,
        desc=f"TrEMBL scan R{round_index}", unit="protein", dynamic_ncols=True, mininterval=2.0, leave=True,
    )
    reported = 0
    scan_log_every = int(config["lora"].get("scan_log_every", 25_000))
    if scan_log_every < 1:
        raise ValueError("lora.scan_log_every must be at least one.")
    for frame in iter_candidate_frames(candidate_path, int(config["data"]["candidate_read_batch_size"])):
        identifiers = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        sample = _hash_sample_mask(
            frame["sequence_hash"], int(config["experiment"]["seed"]), round_index, total, scan_limit
        )
        if selected_indices:
            sample &= np.fromiter((int(identifier) not in selected_indices for identifier in identifiers), dtype=bool, count=len(identifiers))
        if not sample.any():
            continue
        candidate_ids = identifiers[sample]
        batch = frame.iloc[np.flatnonzero(sample)].reset_index(drop=True)
        probabilities, embeddings = model.predict_sequences(
            batch["sequence"].astype(str).tolist(), int(config["lora"]["inference_batch_size"])
        )
        neighbors = anchor.query(embeddings, k=int(config["anchor"]["k"]))
        scored = score_predictions(
            probabilities.numpy(),
            batch["label"].to_numpy(dtype=np.int64),
            neighbors,
            len(labels),
            selection,
            ablation,
            str(config["anchor"]["distance_reduction"]),
        )
        excluded_labels = np.asarray(selection.get("excluded_trembl_labels", []), dtype=np.int64)
        if len(excluded_labels):
            prospective_labels = (
                batch["label"].to_numpy(dtype=np.int64)
                if selection["trembl_training_label"] == "annotation"
                else scored["model_prediction"].to_numpy(dtype=np.int64)
            )
            scored.loc[np.isin(prospective_labels, excluded_labels), "accepted"] = False
        log_frame = pd.concat(
            [
                pd.DataFrame({"candidate_index": candidate_ids}),
                batch.drop(columns=["sequence"], errors="ignore"),
                scored,
            ],
            axis=1,
        )
        _write_scores(output_csv, log_frame, first)
        first = False
        stats["scored"] += len(scored)
        progress.update(len(scored))
        progress.set_postfix(accepted=int(stats["accepted_before_budget"] + int(scored["accepted"].sum())))
        stats["accepted_before_budget"] += int(scored["accepted"].sum())
        if stats["scored"] - reported >= scan_log_every:
            print(json.dumps({"phase": "lora_candidate_scan", "round": round_index, "scored": int(stats["scored"]), "accepted": int(stats["accepted_before_budget"])}), flush=True)
            reported = int(stats["scored"])
        for column in ("confidence", "knn_agreement", "swiss_distance", "reliability"):
            sums[column] += float(scored[column].sum())
        for column in ("model_trembl_agreement", "model_knn_agreement", "triple_agreement"):
            sums[column] += float(scored[column].sum())
        take = scored["accepted"].to_numpy(dtype=bool)
        if take.any():
            records = batch.iloc[np.flatnonzero(take)].copy().reset_index(drop=True)
            accepted_scores = scored.iloc[np.flatnonzero(take)].reset_index(drop=True)
            records = records.rename(columns={"label": "weak_label"})
            records.insert(0, "candidate_index", candidate_ids[take])
            training_label = (
                records["weak_label"].to_numpy(dtype=np.int64)
                if selection["trembl_training_label"] == "annotation"
                else accepted_scores["model_prediction"].to_numpy(dtype=np.int64)
            )
            records["training_label"] = training_label
            records["reliability"] = accepted_scores["reliability"].to_numpy(dtype=np.float32)
            selected_records.append(records)
    progress.close()
    if first:
        pd.DataFrame(columns=["candidate_index", "reliability", "accepted"]).to_csv(output_csv, index=False)
    denominator = max(stats["scored"], 1)
    summary = {
        "candidate_population": float(total) if total is not None else float("nan"),
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
    return (
        pd.concat(selected_records, ignore_index=True)
        if selected_records
        else pd.DataFrame(columns=["candidate_index", "sequence", "weak_label", "training_label", "reliability"]),
        summary,
    )


def _choose_candidates(eligible: pd.DataFrame, num_classes: int, selection: Mapping[str, Any]) -> pd.DataFrame:
    if eligible.empty:
        return eligible.copy()
    chosen_ids = select_curriculum_candidates(
        eligible["candidate_index"].to_numpy(dtype=np.int64),
        eligible["reliability"].to_numpy(dtype=np.float32),
        np.ones(len(eligible), dtype=bool),
        eligible["training_label"].to_numpy(dtype=np.int64),
        num_classes,
        selection,
    )
    return eligible.set_index("candidate_index", drop=False).loc[chosen_ids].reset_index(drop=True)


def _training_frame(swiss_metadata: pd.DataFrame, active_indices: np.ndarray, selected_trembl: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    trusted = swiss_metadata.iloc[active_indices][["sequence", "label"]].copy()
    trusted["weight"] = float(config["training"]["swiss_weight"])
    if selected_trembl.empty:
        return trusted.reset_index(drop=True)
    excluded_labels = set(int(label) for label in config["selection"].get("excluded_trembl_labels", []))
    present_excluded = excluded_labels.intersection(
        selected_trembl["training_label"].to_numpy(dtype=np.int64).tolist()
    )
    if present_excluded:
        raise RuntimeError(
            f"Excluded TrEMBL labels reached the training frame: {sorted(present_excluded)}"
        )
    weak = selected_trembl[["sequence", "training_label", "reliability"]].copy()
    weak = weak.rename(columns={"training_label": "label"})
    if config["experiment"]["ablation"] == "full_tacs":
        weak["weight"] = float(config["training"]["trembl_base_weight"]) * weak["reliability"].clip(0.0, 1.0)
    else:
        weak["weight"] = float(config["training"]["trembl_base_weight"])
    return pd.concat([trusted, weak[["sequence", "label", "weight"]]], ignore_index=True)


def run_lora_tacs(config: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Run TACS with trainable ESM-2 LoRA adapters and streamed TrEMBL scoring."""
    experiment = config["experiment"]
    task = str(config["data"]["task"])
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
    candidate_path = Path(config["data"]["trembl_candidates"])
    if dry_run:
        result = {
            "output": str(output),
            "task": task,
            "ablation": ablation,
            "encoder_mode": "lora",
            "swiss_rows": {name: len(frame) for name, frame in splits.items()},
            "trembl_candidates_exists": candidate_path.is_file(),
            "candidate_scan_per_round": config["lora"].get("candidate_scan_per_round"),
        }
        (output / "dry_run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    if ablation in TREMBl_ABLATIONS and not candidate_path.is_file():
        raise FileNotFoundError(f"TrEMBL candidate pool is missing: {candidate_path}")

    swiss_metadata = pd.concat(
        [frame.assign(split=name) for name, frame in splits.items()], ignore_index=True
    )
    labels = label_space.labels
    validation_frame = swiss_metadata[swiss_metadata["split"].eq("swiss_validation")][["sequence", "label"]].reset_index(drop=True)
    test_frame = swiss_metadata[swiss_metadata["split"].eq("swiss_test")][["sequence", "label"]].reset_index(drop=True)
    all_train_indices = np.flatnonzero(swiss_metadata["split"].isin(["swiss_seed", "swiss_candidate"]).to_numpy())
    rounds = int(config["training"]["rounds"])
    selected_trembl = pd.DataFrame(columns=["candidate_index", "sequence", "weak_label", "training_label", "reliability"])
    selected_indices: set[int] = set()
    round_rows: list[dict[str, Any]] = []

    active_swiss, _ = _round_swiss_indices(
        swiss_metadata, 0, rounds, ablation, int(experiment["seed"]),
        int(config["data"]["background_label"]),
        config["training"].get("max_background_to_other_ratio"),
    )
    train_frame = _training_frame(swiss_metadata, active_swiss, selected_trembl, config)
    fit = fit_lora_classifier(
        train_frame,
        validation_frame,
        len(labels),
        config["model"],
        config["lora"],
        config["training"],
        int(experiment["seed"]),
        output / "round_0_epochs.jsonl",
    )
    model = fit.model
    model.save_adapter(output, 0, config, labels)
    round_rows.append(
        {
            "round": 0,
            "total_swiss_samples": int(len(active_swiss)),
            "new_swiss_samples": int(len(active_swiss)),
            "total_accepted_trembl": 0,
            "new_accepted_trembl": 0,
            "training_class_counts": {str(label): int(count) for label, count in zip(*np.unique(train_frame["label"], return_counts=True), strict=True)},
            **{f"validation_{key}": value for key, value in fit.best_validation.items() if key != "confusion_matrix"},
        }
    )

    if ablation != "seed_only":
        for round_index in range(1, rounds + 1):
            active_swiss, new_swiss = _round_swiss_indices(
                swiss_metadata, round_index, rounds, ablation, int(experiment["seed"]),
                int(config["data"]["background_label"]),
                config["training"].get("max_background_to_other_ratio"),
            )
            score_summary: dict[str, float] = {}
            new_selected = pd.DataFrame(columns=selected_trembl.columns)
            if ablation in TREMBl_ABLATIONS:
                anchor = _anchor_for_round(model, swiss_metadata, active_swiss, all_train_indices, selected_trembl, config)
                eligible, score_summary = _score_candidate_round(
                    model,
                    candidate_path,
                    selected_indices,
                    anchor,
                    labels,
                    config,
                    round_index,
                    output / f"trembl_scores_round_{round_index}.csv",
                )
                selection = thresholds_for_round(config["selection"], round_index)
                if ablation == "all_trembl":
                    selection["max_trembl_per_round"] = None
                    selection["balance_by_class"] = False
                new_selected = _choose_candidates(eligible, len(labels), selection)
                if not new_selected.empty:
                    selected_trembl = pd.concat([selected_trembl, new_selected], ignore_index=True)
                    selected_indices.update(new_selected["candidate_index"].to_numpy(dtype=np.int64).tolist())
            # A new LoRA model is intentionally initialized from the same pretrained
            # ESM-2 weights each round; deleting the scorer avoids dual 650M models.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_frame = _training_frame(swiss_metadata, active_swiss, selected_trembl, config)
            fit = fit_lora_classifier(
                train_frame,
                validation_frame,
                len(labels),
                config["model"],
                config["lora"],
                config["training"],
                int(experiment["seed"]) + round_index,
                output / f"round_{round_index}_epochs.jsonl",
            )
            model = fit.model
            model.save_adapter(output, round_index, config, labels)
            selected_trembl.drop(columns=["sequence"], errors="ignore").to_csv(output / "accepted_trembl.csv", index=False)
            class_counts = {str(label): int(count) for label, count in selected_trembl["training_label"].value_counts().sort_index().items()}
            round_rows.append(
                {
                    "round": round_index,
                    "total_swiss_samples": int(len(active_swiss)),
                    "new_swiss_samples": int(len(new_swiss)),
                    "total_accepted_trembl": int(len(selected_trembl)),
                    "new_accepted_trembl": int(len(new_selected)),
                    "accepted_trembl_by_class": class_counts,
                    "new_accepted_trembl_by_class": {str(label): int(count) for label, count in new_selected["training_label"].value_counts().sort_index().items()},
                    "training_class_counts": {str(label): int(count) for label, count in zip(*np.unique(train_frame["label"], return_counts=True), strict=True)},
                    **score_summary,
                    **{f"validation_{key}": value for key, value in fit.best_validation.items() if key != "confusion_matrix"},
                }
            )

    evaluation = evaluate_lora_classifier(
        model,
        test_frame,
        len(labels),
        int(config["lora"]["inference_batch_size"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        confidence_level=float(config["evaluation"]["confidence_level"]),
        bootstrap_seed=int(experiment["seed"]),
        progress_description="Swiss test",
    )
    save_evaluation(evaluation, output, "swiss_test", list(labels))
    with (output / "round_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in round_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {"output": str(output), "ablation": ablation, "encoder_mode": "lora", "swiss_test": evaluation, "rounds": round_rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
