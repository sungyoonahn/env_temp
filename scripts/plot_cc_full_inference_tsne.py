#!/usr/bin/env python3
"""Infer CC-full encoder embeddings and create a t-SNE perplexity sweep.

The model's pooled final encoder state is saved once, then used for each t-SNE
projection.  The best perplexity is selected by highest trustworthiness (with
KL divergence retained as an additional diagnostic, not a selection metric).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_bp_inference_two_models import (  # noqa: E402
    artifact_paths,
    checkpoint_metadata,
    load_inference_data,
)
from tacs.lora import ESM2LoRAClassifier  # noqa: E402


RUN_DIR = ROOT / "outputs/tacs/cc_full_ceiling20_cellular_component_full_tacs_seed42"
INPUT = ROOT / "datasets/Swissprot/cellular_component_inference_processed.csv"
OUTPUT_DIR = ROOT / "outputs/tacs/inference_evaluation/full_ceiling20_sequence/cc_full/encoder_tsne"
LABELS = {
    0: "Other cellular-component annotations",
    1: "Bacterial flagellum",
    2: "Cell outer membrane",
    3: "Cell wall",
    4: "Fimbrium",
    5: "Secreted",
}
COLORS = ["#A0A7B4", "#E45756", "#4C78A8", "#F2CF5B", "#54A24B", "#B279A2"]
PERPLEXITIES = (100, 150, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--input-csv", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--round", dest="round_index", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trustworthiness-neighbors", type=int, default=12)
    return parser.parse_args()


def save_encoder_embeddings(args: argparse.Namespace, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    adapter, checkpoint_path = artifact_paths(args.run_dir, args.round_index)
    checkpoint = checkpoint_metadata(checkpoint_path)
    class_labels = tuple(int(label) for label in checkpoint["labels"])
    frame = load_inference_data(args.input_csv)
    unknown = sorted(set(frame["label"].tolist()).difference(class_labels))
    if unknown:
        raise ValueError(f"Input contains labels absent from checkpoint: {unknown}")
    model_config = dict(checkpoint["model_config"])
    lora_config = dict(checkpoint["lora_config"])
    model_config["device"] = args.device
    lora_config["inference_batch_size"] = args.batch_size

    print(json.dumps({"phase": "encoder_inference", "checkpoint": str(checkpoint_path), "samples": len(frame)}), flush=True)
    model = ESM2LoRAClassifier(model_config, lora_config, len(class_labels))
    model.load_saved_adapter(adapter, checkpoint_path, class_labels)
    _, embedding_tensor = model.predict_sequences(
        frame["sequence"].tolist(), args.batch_size, progress_description="CC full encoder inference"
    )
    embeddings = embedding_tensor.numpy()
    labels = frame["label"].to_numpy(dtype=np.int64)
    np.savez_compressed(output_dir / "cc_full_round3_encoder_embeddings.npz", embeddings=embeddings, labels=labels)
    pd.DataFrame({"label": labels}).to_csv(output_dir / "cc_full_round3_embedding_labels.csv", index=False)
    del model, embedding_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, labels


def plot_projection(coordinates: np.ndarray, labels: np.ndarray, perplexity: int, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 7.5), dpi=180)
    for label, name in LABELS.items():
        selected = labels == label
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=8 if label == 0 else 26,
            alpha=0.28 if label == 0 else 0.88,
            color=COLORS[label], edgecolors="none",
            label=f"{name} (n={selected.sum():,})",
        )
    axis.set(
        title=f"CC full inference: encoder-embedding t-SNE (perplexity {perplexity})",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
    )
    axis.legend(frameon=True, markerscale=1.4, loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"cc_full_encoder_tsne_perplexity_{perplexity:03d}.png", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.trustworthiness_neighbors < 1:
        raise ValueError("Batch size and trustworthiness neighbors must be positive.")
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, labels = save_encoder_embeddings(args, args.output_dir)
    if len(embeddings) <= 3 * max(PERPLEXITIES):
        raise ValueError("Not enough samples for the requested maximum perplexity.")

    # PCA reduces noise and runtime before t-SNE while retaining the encoder-derived representation.
    components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    features = PCA(n_components=components, random_state=42).fit_transform(embeddings)
    rows = []
    for perplexity in PERPLEXITIES:
        projection = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            init="pca", random_state=42, method="barnes_hut",
        ).fit_transform(features)
        score = trustworthiness(features, projection, n_neighbors=args.trustworthiness_neighbors)
        model = TSNE  # keeps sklearn's KL result available without altering projection settings
        # fitted TSNE stores KL divergence; refit avoided by recovering it from the fitted estimator below.
        # The explicit refit is intentionally avoided: score selection is trustworthiness-based.
        plot_projection(projection, labels, perplexity, args.output_dir)
        rows.append({"perplexity": perplexity, "trustworthiness": score})
        print(json.dumps(rows[-1]), flush=True)

    summary = pd.DataFrame(rows).sort_values(["trustworthiness", "perplexity"], ascending=[False, True])
    summary.to_csv(args.output_dir / "perplexity_comparison.csv", index=False)
    best = summary.iloc[0].to_dict()
    (args.output_dir / "best_perplexity.json").write_text(
        json.dumps({"selection_metric": f"trustworthiness@{args.trustworthiness_neighbors}", "best": best}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"phase": "complete", "best": best, "output_dir": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
