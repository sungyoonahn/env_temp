#!/usr/bin/env python3
"""Resume only the unfinished third round of the preserved ESM-2 LoRA run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacs.config import dump_config, load_config
from tacs.data import iter_candidate_frames, load_swiss_dataset, make_swiss_splits, write_split_manifest
from tacs.lora import ESM2LoRAClassifier, evaluate_lora_classifier, fit_lora_classifier
from tacs.lora_runner import (
    TREMBl_ABLATIONS,
    _anchor_for_round,
    _candidate_count,
    _choose_candidates,
    _ensure_contiguous_labels,
    _round_swiss_indices,
    _score_candidate_round,
    _training_frame,
)
from tacs.reliability import thresholds_for_round
from tacs.trainer import save_evaluation

RESUME_ROOT = Path("outputs/tacs/bp_full_ceiling8_resumed_round3")
SOURCE_DEFAULT = RESUME_ROOT / "source_rounds_0_to_2"


def output_for(config):
    del config
    return RESUME_ROOT / "completed_round_3"

def recover_sequences(candidate_path, accepted_path, batch_size):
    accepted = pd.read_csv(accepted_path)
    needed = {"candidate_index", "weak_label", "training_label", "reliability"}
    if missing := needed.difference(accepted.columns):
        raise ValueError(f"accepted_trembl.csv lacks {sorted(missing)}")
    if accepted["candidate_index"].duplicated().any():
        raise ValueError("accepted_trembl.csv contains duplicate candidate indices")
    accepted["candidate_index"] = accepted["candidate_index"].astype(np.int64)
    wanted = set(accepted["candidate_index"].tolist())
    total = _candidate_count(candidate_path)
    found, parts, offset = 0, [], 0
    progress = tqdm(total=total, desc="Recovering prior TrEMBL sequences", unit="protein",
                    dynamic_ncols=True, mininterval=2.0, leave=True)
    for frame in iter_candidate_frames(candidate_path, batch_size):
        indices = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        mask = np.fromiter((int(index) in wanted for index in indices),
                           dtype=bool, count=len(indices))
        if mask.any():
            rows = frame.iloc[np.flatnonzero(mask)]
            parts.append(pd.DataFrame({
                "candidate_index": indices[mask],
                "sequence": rows["sequence"].astype(str).to_numpy(),
                "candidate_label": rows["label"].to_numpy(dtype=np.int64),
            }))
            found += int(mask.sum())
        progress.update(len(frame))
        progress.set_postfix(found=found, expected=len(accepted))
    progress.close()
    if found != len(accepted):
        raise RuntimeError(
            f"Candidate pool mismatch: recovered {found} of {len(accepted)} previously selected rows."
        )
    recovered = pd.concat(parts, ignore_index=True).set_index("candidate_index", verify_integrity=True)
    restored = accepted.set_index("candidate_index", verify_integrity=True).join(
        recovered, how="left", validate="one_to_one"
    ).reset_index()
    if restored["sequence"].isna().any():
        raise RuntimeError("A recovered old pseudo-label has no sequence.")
    if not np.array_equal(restored["weak_label"].to_numpy(dtype=np.int64),
                          restored["candidate_label"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Candidate labels do not match saved round-2 pseudo-label metadata.")
    return restored.drop(columns=["candidate_label"])


def label_counts(frame, column):
    if frame.empty:
        return {}
    return {str(label): int(count) for label, count in frame[column].value_counts().sort_index().items()}


def run(config, source, validate_only=False):
    if config["model"]["encoder_mode"] != "lora":
        raise ValueError("This resume command requires model.encoder_mode=lora.")
    if int(config["training"]["rounds"]) != 3:
        raise ValueError("This command resumes exactly the unfinished round 3.")
    if config["training"].get("max_background_to_other_ratio") is not None:
        raise ValueError("Use max_background_to_other_ratio: null for exact pre-cap continuation.")
    if config["experiment"]["ablation"] not in TREMBl_ABLATIONS:
        raise ValueError("The preserved experiment needs a TrEMBL-scoring ablation.")

    adapter = source / "lora_adapter_round_2"
    checkpoint = source / "model_round_2.pt"
    accepted_path = source / "accepted_trembl.csv"
    missing = [str(path) for path in (adapter, checkpoint, accepted_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing preserved artifacts: " + ", ".join(missing))
    candidate_path = Path(config["data"]["trembl_candidates"])
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate pool missing: {candidate_path}")

    swiss, labels = load_swiss_dataset(config)
    _ensure_contiguous_labels(labels.labels)
    splits = make_swiss_splits(swiss, config)
    output = output_for(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite resume output: {output}")
    print(json.dumps({
        "phase": "resume_setup", "source_run": str(source),
        "completed_round": 2, "running_round": 3, "output": str(output),
        "background_cap": None,
    }), flush=True)
    if validate_only:
        return {"valid": True, "output": str(output),
                "split_rows": {name: len(frame) for name, frame in splits.items()}}

    output.mkdir(parents=True, exist_ok=False)
    dump_config(config, output / "resolved_config.yaml")
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_split_manifest(splits, labels, output / "swiss_split_manifest.json")

    selected = recover_sequences(candidate_path, accepted_path, int(config["data"]["candidate_read_batch_size"]))
    old_selected_count = len(selected)
    selected_indices = set(selected["candidate_index"].to_numpy(dtype=np.int64).tolist())
    swiss_meta = pd.concat([frame.assign(split=name) for name, frame in splits.items()],
                           ignore_index=True)
    validation = swiss_meta.loc[swiss_meta["split"].eq("swiss_validation"),
                                ["sequence", "label"]].reset_index(drop=True)
    test = swiss_meta.loc[swiss_meta["split"].eq("swiss_test"),
                          ["sequence", "label"]].reset_index(drop=True)
    all_train = np.flatnonzero(swiss_meta["split"].isin(["swiss_seed", "swiss_candidate"]).to_numpy())
    round_index = 3
    active, new_swiss = _round_swiss_indices(
        swiss_meta, round_index, int(config["training"]["rounds"]),
        str(config["experiment"]["ablation"]), int(config["experiment"]["seed"]),
        int(config["data"]["background_label"]), None,
    )
    print(json.dumps({
        "phase": "resume_restore", "prior_selected_trembl": old_selected_count,
        "round_3_swiss_samples": int(len(active)),
        "round_3_new_swiss_samples": int(len(new_swiss)),
    }), flush=True)

    model = ESM2LoRAClassifier(config["model"], config["lora"], len(labels.labels))
    model.load_saved_adapter(adapter, checkpoint, labels.labels)
    print(json.dumps({"phase": "round_3_anchor", "k": int(config["anchor"]["k"])}), flush=True)
    anchor = _anchor_for_round(model, swiss_meta, active, all_train, selected, config)
    eligible, score_summary = _score_candidate_round(
        model, candidate_path, selected_indices, anchor, labels.labels, config,
        round_index, output / "trembl_scores_round_3.csv",
    )
    new_selected = _choose_candidates(
        eligible, len(labels.labels), thresholds_for_round(config["selection"], round_index)
    )
    if not new_selected.empty:
        selected = pd.concat([selected, new_selected], ignore_index=True)

    train_frame = _training_frame(swiss_meta, active, selected, config)
    print(json.dumps({
        "phase": "round_3_training", "train_samples": int(len(train_frame)),
        "new_selected_trembl": int(len(new_selected)),
        "total_selected_trembl": int(len(selected)),
        "training_class_counts": label_counts(train_frame, "label"),
    }), flush=True)
    fit = fit_lora_classifier(
        train_frame, validation, len(labels.labels), config["model"], config["lora"],
        config["training"], int(config["experiment"]["seed"]) + round_index,
        output / "round_3_epochs.jsonl", initial_model=model,
    )
    model = fit.model
    model.save_adapter(output, round_index, config, labels.labels)
    selected.drop(columns=["sequence"], errors="ignore").to_csv(output / "accepted_trembl.csv", index=False)

    print(json.dumps({"phase": "final_swiss_test"}), flush=True)
    evaluation = evaluate_lora_classifier(
        model, test, len(labels.labels), int(config["lora"]["inference_batch_size"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        confidence_level=float(config["evaluation"]["confidence_level"]),
        bootstrap_seed=int(config["experiment"]["seed"]),
        progress_description="Swiss test",
    )
    save_evaluation(evaluation, output, "swiss_test", list(labels.labels))
    row = {
        "round": round_index, "resumed_from_completed_round": 2,
        "total_swiss_samples": int(len(active)), "new_swiss_samples": int(len(new_swiss)),
        "total_accepted_trembl": int(len(selected)), "new_accepted_trembl": int(len(new_selected)),
        "accepted_trembl_by_class": label_counts(selected, "training_label"),
        "new_accepted_trembl_by_class": label_counts(new_selected, "training_label"),
        "training_class_counts": label_counts(train_frame, "label"),
        **score_summary,
        **{f"validation_{key}": value for key, value in fit.best_validation.items()
           if key != "confusion_matrix"},
    }
    (output / "round_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "source_run": str(source), "resumed_from_completed_round": 2,
        "executed_round": 3, "source_adapter": str(adapter),
        "source_checkpoint": str(checkpoint), "source_accepted_trembl": str(accepted_path),
        "preserved_selected_trembl": int(old_selected_count), "background_cap": None,
    }
    (output / "resume_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                   encoding="utf-8")
    summary = {"output": str(output), "source_run": str(source),
               "resumed_from_completed_round": 2, "round": row, "swiss_test": evaluation}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tacs_go_lora_resume_round3.yaml")
    parser.add_argument("--source-run", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir)
    print(json.dumps(run(config, args.source_run, args.validate_only), indent=2), flush=True)


if __name__ == "__main__":
    main()
