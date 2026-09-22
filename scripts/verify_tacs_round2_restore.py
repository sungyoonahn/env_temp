#!/usr/bin/env python3
"""Verify that a restored round-2 LoRA checkpoint reproduces its saved validation score."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacs.config import load_config
from tacs.data import load_swiss_dataset, make_swiss_splits
from tacs.lora import ESM2LoRAClassifier, evaluate_lora_classifier
from tacs.lora_runner import _ensure_contiguous_labels

SOURCE = Path("outputs/tacs/bp_full_ceiling8_resumed_round3/source_rounds_0_to_2")
EXPECTED_MACRO_F1 = 0.8406900186062577


def main() -> None:
    config = load_config("configs/tacs_go_lora_resume_round3.yaml")
    swiss, label_space = load_swiss_dataset(config)
    _ensure_contiguous_labels(label_space.labels)
    splits = make_swiss_splits(swiss, config)
    validation = splits["swiss_validation"][["sequence", "label"]].reset_index(drop=True)
    model = ESM2LoRAClassifier(config["model"], config["lora"], len(label_space.labels))
    model.load_saved_adapter(
        SOURCE / "lora_adapter_round_2",
        SOURCE / "model_round_2.pt",
        label_space.labels,
    )
    metrics = evaluate_lora_classifier(
        model,
        validation,
        len(label_space.labels),
        int(config["lora"]["inference_batch_size"]),
        progress_description="Round-2 restoration parity",
    )
    result = {
        "expected_macro_f1": EXPECTED_MACRO_F1,
        "restored_macro_f1": metrics["macro_f1"],
        "absolute_difference": abs(metrics["macro_f1"] - EXPECTED_MACRO_F1),
        "accuracy": metrics["accuracy"],
        "per_class_f1": metrics["per_class_f1"],
    }
    print(json.dumps(result, indent=2), flush=True)
    if result["absolute_difference"] > 1e-12:
        raise RuntimeError("Restored adapter does not reproduce the archived round-2 validation result.")


if __name__ == "__main__":
    main()
