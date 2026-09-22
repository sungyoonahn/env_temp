import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

OUTPUT_DIR = Path(__file__).resolve().parent / "confusion_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "Molecular Function": ["Normal", "Toxin", "BT(Bacteriotoxin)"],
    "Biological Process": ["Normal", "Antibiotic resistance", "Metal resistance", "Toxin-antitoxin", "Virulence"],
    "Cellular Component": ["Normal", "Flagellum", "Cell outer membrane", "Cell wall", "Fimbrium", "Secreted"],
}
INFERENCE_LABELS = {
    "CC": [
        "CC normal",
        "CC Flagellum|Bacterial flagellum",
        "CC Cell outer membrane",
        "CC Cell wall",
        "CC Fimbrium",
        "CC Secreted",
    ],
    "BP": [
        "BP normal",
        "BP Antibiotic resistance|Trimethoprim resistance",
        "BP Metal Resistance",
        "BP Toxin-antitoxin system",
        "BP Virulence",
    ],
    "MF": ["MF normal", "MF Toxin", "MF Bacteriotoxin"],
}
REPORT_LABELS = {
    "CC": ["Normal", "Flagellum", "Outer membrane", "Cell wall", "Fimbrium", "Secreted"],
    "BP": ["Normal", "Antibiotic resistance", "Metal Resistance", "Toxin-antitoxin system", "Virulence"],
    "MF": ["Normal", "Toxin", "Bacteriotoxin"],
}
DATA = {
    "ProtBERT": {
        "Molecular Function": [[12433, 90, 99], [11, 125, 2], [3, 1, 34]],
        "Biological Process": [[10136, 41, 14, 101, 209], [128, 149, 1, 1, 3], [6, 0, 34, 0, 3], [25, 0, 0, 54, 13], [16, 0, 0, 3, 174]],
        "Cellular Component": [[8027, 47, 65, 4, 5, 131], [0, 46, 0, 0, 0, 1], [8, 8, 237, 1, 0, 22], [0, 0, 0, 2, 0, 6], [2, 0, 1, 0, 106, 1], [5, 6, 0, 3, 13, 130]],
    },
    "ESM-2 650M": {
        "Molecular Function": [[12471, 56, 95], [5, 128, 5], [1, 1, 36]],
        "Biological Process": [[10045, 120, 39, 48, 249], [131, 133, 10, 2, 6], [8, 0, 34, 0, 1], [20, 0, 1, 68, 3], [16, 0, 0, 1, 176]],
        "Cellular Component": [[7936, 38, 54, 9, 6, 236], [0, 39, 8, 0, 0, 0], [10, 0, 236, 2, 0, 28], [0, 0, 0, 3, 0, 5], [2, 0, 1, 0, 106, 1], [7, 7, 0, 4, 13, 126]],
    },
    "ESM-2 3B": {
        "Molecular Function": [[12449, 83, 90], [5, 128, 5], [3, 1, 34]],
        "Biological Process": [[10127, 60, 24, 38, 252], [156, 114, 5, 2, 5], [6, 0, 36, 0, 1], [16, 0, 0, 66, 10], [16, 0, 0, 0, 177]],
        "Cellular Component": [[7928, 45, 82, 9, 6, 209], [0, 46, 0, 0, 0, 1], [8, 8, 228, 2, 0, 30], [2, 0, 0, 3, 0, 3], [2, 0, 1, 0, 106, 1], [5, 6, 1, 4, 13, 128]],
    },
}
SLUGS = {
    "ProtBERT": "protbert", "ESM-2 650M": "esm2-650m", "ESM-2 3B": "esm2-3b",
    "Molecular Function": "molecular-function", "Biological Process": "biological-process",
    "Cellular Component": "cellular-component",
}
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def font(size, bold=False):
    names = ["seguisb.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def center_text(draw, box, text, text_font, fill, spacing=4, align="center"):
    left, top, right, bottom = box
    bb = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align=align)
    width, height = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text(((left + right - width) / 2, (top + bottom - height) / 2 - bb[1]), text,
                        font=text_font, fill=fill, spacing=spacing, align=align)


def wrapped(label):
    return {
        "BT(Bacteriotoxin)": "BT\n(Bacteriotoxin)",
        "Antibiotic resistance": "Antibiotic\nresistance",
        "Metal resistance": "Metal\nresistance",
        "Toxin-antitoxin": "Toxin-\nantitoxin",
        "Cell outer membrane": "Cell outer\nmembrane",
    }.get(label, label)


def blue_scale(percent):
    start, end = (239, 246, 255), (18, 92, 164)
    t = max(0.0, min(1.0, percent / 100.0)) ** 0.62
    return tuple(round(a + (b - a) * t) for a, b in zip(start, end))


def matrix_panel(model, category, title=None, cell=132):
    labels, values = LABELS[category], DATA[model][category]
    n, left, top, right, bottom = len(labels), 265, 160, 30, 105
    title_h = 74 if title else 16
    width, height = left + n * cell + right, title_h + top + n * cell + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    ink, muted = (17, 24, 39), (75, 85, 99)
    if title:
        center_text(draw, (15, 8, width - 15, title_h - 2), title, font(30, True), ink)
    grid_x, grid_y = left, title_h + top
    center_text(draw, (grid_x, title_h + 2, grid_x + n * cell, title_h + 46), "Predicted label", font(21, True), ink)
    center_text(draw, (8, grid_y, 50, grid_y + n * cell), "TRUE", font(17, True), muted)
    for j, label in enumerate(labels):
        center_text(draw, (grid_x + j * cell + 3, title_h + 44, grid_x + (j + 1) * cell - 3, grid_y - 5),
                    wrapped(label), font(17, True), ink, spacing=2)
    for i, label in enumerate(labels):
        center_text(draw, (52, grid_y + i * cell, grid_x - 12, grid_y + (i + 1) * cell),
                    wrapped(label), font(18, True), ink, spacing=2)
    for i, row in enumerate(values):
        row_total = sum(row)
        for j, value in enumerate(row):
            pct = 100 * value / row_total if row_total else 0
            x0, y0 = grid_x + j * cell, grid_y + i * cell
            x1, y1 = x0 + cell, y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=blue_scale(pct), outline="white", width=4)
            color = (255, 255, 255) if pct >= 48 else ink
            center_text(draw, (x0 + 3, y0 + 26, x1 - 3, y0 + 72), f"{value:,}", font(22, i == j), color)
            center_text(draw, (x0 + 3, y0 + 70, x1 - 3, y1 - 18), f"({pct:.1f}%)", font(17), color)
    center_text(draw, (15, grid_y + n * cell + 28, width - 15, height - 10),
                "Cell color: row-normalized (%)   |   Annotation: count (row %)", font(16), muted)
    return image


def save_png(image, filename):
    path = OUTPUT_DIR / filename
    image.save(path, "PNG", dpi=(300, 300), optimize=True)
    return path


def combine_horizontal(panels, heading, gap=30, top=110):
    width = sum(p.width for p in panels) + gap * (len(panels) - 1) + 40
    height = max(p.height for p in panels) + top + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    center_text(draw, (20, 10, width - 20, top - 10), heading, font(38, True), (17, 24, 39))
    x = 20
    for panel in panels:
        canvas.paste(panel, (x, top))
        x += panel.width + gap
    return canvas


def combine_grid(rows, heading, col_gap=24, row_gap=34, top=120):
    cols = max(len(row) for row in rows)
    col_widths = [max(row[c].width for row in rows) for c in range(cols)]
    row_heights = [max(p.height for p in row) for row in rows]
    width = sum(col_widths) + col_gap * (cols - 1) + 40
    height = sum(row_heights) + row_gap * (len(rows) - 1) + top + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    center_text(draw, (20, 10, width - 20, top - 10), heading, font(42, True), (17, 24, 39))
    y = top
    for r, row in enumerate(rows):
        x = 20
        for c, panel in enumerate(row):
            canvas.paste(panel, (x + (col_widths[c] - panel.width) // 2, y))
            x += col_widths[c] + col_gap
        y += row_heights[r] + row_gap
    return canvas


def save_inference_confusion_matrix(
    predictions_path: Path, task: str, output_path: Path | None = None
) -> Path:
    """Write a labeled confusion matrix from a main.py inference export."""
    frame = pd.read_csv(predictions_path)
    required = {"label", "predicted_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{predictions_path} is missing required column(s): {sorted(missing)}"
        )
    labels = INFERENCE_LABELS[task]
    indices = list(range(len(labels)))
    matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(
        index=indices, columns=indices, fill_value=0
    )
    matrix.index = [f"True: {label}" for label in labels]
    matrix.columns = [f"Pred: {label}" for label in labels]
    output_path = output_path or predictions_path.with_name("inference_confusion_matrix.csv")
    matrix.to_csv(output_path)
    print(f"Saved {output_path}")
    print(matrix.to_string())
    return output_path


def inference_report(frame: pd.DataFrame, task: str) -> str:
    """Create the requested Markdown metrics/matrix/per-class-accuracy report."""
    labels = INFERENCE_LABELS[task]
    indices = np.arange(len(labels))
    probability_columns = [f"probability_{index}" for index in indices]
    missing = set(probability_columns).difference(frame.columns)
    if missing:
        raise ValueError(
            "Prediction probabilities are required for AUROC/AUPRC; missing "
            f"{sorted(missing)}."
        )

    true = frame["label"].to_numpy(dtype=int)
    predicted = frame["predicted_label"].to_numpy(dtype=int)
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(
        index=indices, columns=indices, fill_value=0
    ).to_numpy(dtype=int)
    per_class_accuracy = np.divide(
        np.diag(matrix), matrix.sum(axis=1), out=np.zeros(len(indices), dtype=float), where=matrix.sum(axis=1) != 0
    )
    smoothed_accuracy = np.maximum(per_class_accuracy, 1e-3)
    harmonic_accuracy = len(indices) / np.sum(1.0 / smoothed_accuracy)
    geometric_accuracy = float(np.prod(smoothed_accuracy) ** (1.0 / len(indices)))
    one_hot = label_binarize(true, classes=indices)

    metrics = {
        # Project convention: "Accuracy" means balanced accuracy, i.e. the
        # arithmetic mean of per-class recalls.
        "Accuracy": float(per_class_accuracy.mean()),
        "F1 Score": float(f1_score(true, predicted, labels=indices, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(true, predicted)),
        "AUROC": float(roc_auc_score(true, probabilities, labels=indices, average="macro", multi_class="ovr")),
        "AUPRC": float(average_precision_score(one_hot, probabilities, average="macro")),
        "H-ACC": float(harmonic_accuracy),
        "G-ACC": float(geometric_accuracy),
    }
    headings = REPORT_LABELS[task]
    lines = [
        "| Accuracy | F1 Score | MCC | AUROC | AUPRC | H-ACC | G-ACC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| " + " | ".join(f"{value:.6f}" for value in metrics.values()) + " |",
        "",
        "| True \\ Pred | " + " | ".join(headings) + " |",
        "| --- | " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for heading, row in zip(headings, matrix, strict=True):
        lines.append("| " + heading + " | " + " | ".join(str(value) for value in row) + " |")
    lines.extend([
        "| Per Class ACC | " + " | ".join(f"{value:.6f}" for value in per_class_accuracy) + " |",
        "",
        "H-ACC and G-ACC use a 1e-3 floor for zero per-class accuracy.",
    ])
    return "\n".join(lines) + "\n"


def render_reference_matrices():
    paths = []
    for model in DATA:
        for category in LABELS:
            paths.append(save_png(matrix_panel(model, category, f"{model} — {category}"),
                                  f"{SLUGS[model]}_{SLUGS[category]}.png"))
    for model in DATA:
        panels = [matrix_panel(model, category, category, cell=112) for category in LABELS]
        paths.append(save_png(combine_horizontal(panels, f"{model} — Confusion Matrices"), f"{SLUGS[model]}_all-categories.png"))
    for category in LABELS:
        panels = [matrix_panel(model, category, model, cell=112) for model in DATA]
        paths.append(save_png(combine_horizontal(panels, f"{category} — Model Comparison"), f"all-models_{SLUGS[category]}.png"))
    rows = [[matrix_panel(model, category, f"{model} — {category}", cell=102) for category in LABELS] for model in DATA]
    paths.append(save_png(combine_grid(rows, "Protein Virulence Classification — Confusion Matrices"), "all-models_all-categories.png"))
    manifest = {
        "source": "User-provided confusion matrices from referenced conversation",
        "encoding": "Cell color is row-normalized percentage; annotation is raw count and row percentage.",
        "files": [p.name for p in paths],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(paths)} PNG files in {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="main.py inference_predictions.csv to convert into a confusion-matrix CSV.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(INFERENCE_LABELS),
        help="Task label mapping for --predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path (default: beside the predictions file).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.predictions:
        if not args.task:
            raise ValueError("--task is required with --predictions.")
        output_path = save_inference_confusion_matrix(args.predictions, args.task, args.output)
        frame = pd.read_csv(args.predictions)
        report = inference_report(frame, args.task)
        report_path = output_path.with_name("inference_report.md")
        report_path.write_text(report, encoding="utf-8")
        print(f"Saved {report_path}")
        print(report)
        return
    render_reference_matrices()


if __name__ == "__main__":
    main()
