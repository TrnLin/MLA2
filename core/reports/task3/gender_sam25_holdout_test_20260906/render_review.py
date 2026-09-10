"""Render the completed evaluation and matching high-resolution image examples."""

from fashion.task3_paths import resolve_task3_path

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train.metrics import classification_metrics
from fashion.train.task3_decisions import probability_columns

REPORT = Path(__file__).resolve().parent
CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]
METRICS = json.loads((REPORT / "evaluation_metrics.json").read_text())["metrics"]
SUMMARY = pd.read_csv(REPORT / "evaluation_summary.csv")
PRIMARY = SUMMARY.loc[SUMMARY.label_basis.eq("primary")]

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
x = np.arange(2)
for offset, column, label, color in (
    (-0.18, "accuracy", "Accuracy", "#337b8c"),
    (0.18, "macro_f1", "Macro-F1", "#b27742"),
):
    bars = axes[0, 0].bar(x + offset, PRIMARY[column] * 100, 0.36, label=label, color=color)
    axes[0, 0].bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
axes[0, 0].set(
    title="Primary scores against dataset labels",
    xticks=x,
    xticklabels=["Holdout (5,778)", "Teacher test (5,829)"],
    ylabel="Score (%)",
    ylim=(0, 105),
)
axes[0, 0].legend(frameon=False, loc="lower right")
for offset, scope, label, color in (
    (-0.18, "holdout", "Holdout", "#337b8c"),
    (0.18, "test", "Teacher test", "#b27742"),
):
    values = [r["f1"] * 100 for r in METRICS[f"{scope}/primary"]["per_class"]]
    axes[0, 1].bar(np.arange(5) + offset, values, 0.36, label=label, color=color)
axes[0, 1].set(
    title="Equal attention to all five classes",
    ylabel="Class F1 (%)",
    xticks=np.arange(5),
    xticklabels=CLASSES,
    ylim=(0, 105),
)
axes[0, 1].legend(frameon=False)
for ax, scope, label in (
    (axes[1, 0], "holdout", "Holdout"),
    (axes[1, 1], "test", "Teacher test"),
):
    matrix = np.asarray(METRICS[f"{scope}/primary"]["confusion_matrix"])
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for row in range(5):
        for col in range(5):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}\n{fractions[row, col]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if fractions[row, col] > 0.5 else "black",
            )
    ax.set(
        title=f"{label}: counts and share of each true class",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="Reference label",
    )
fig.suptitle("Fixed SAM25: five-model average on holdout and teacher test", fontsize=16)
fig.tight_layout()
fig.savefig(REPORT / "evaluation.png", dpi=160, bbox_inches="tight")
plt.close(fig)

test = pd.read_csv(REPORT / "test_predictions_and_labels.csv", keep_default_na=False)
manifest = pd.read_csv(REPORT / "test_image_manifest.csv").set_index("id")
sample = (
    test.sort_values("id")
    .groupby("reference_gender")
    .head(2)
    .assign(class_order=lambda t: t.reference_gender.map(dict(zip(CLASSES, range(5)))))
    .sort_values(["class_order", "id"])
)
sample_records = []
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for position, row in enumerate(sample.itertuples()):
    ax = axes[position % 2, position // 2]
    teacher_path, highres_path = resolve_task3_path(manifest.loc[row.id, "path"], root=ROOT), resolve_task3_path(row.highres_image_path, root=ROOT)
    canvas = Image.new("RGB", (320, 220), "#f1f3f4")
    sizes = []
    for index, path in enumerate((teacher_path, highres_path)):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        sizes.append(image.size)
        fitted = ImageOps.contain(image, (145, 200))
        canvas.paste(fitted, (index * 160 + (160 - fitted.width) // 2, (220 - fitted.height) // 2))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(
        f"ID {row.id} · {row.reference_gender}\nPrediction: {row.predicted_gender}", fontsize=10
    )
    ax.text(
        0.5,
        -0.02,
        "Teacher input       High-res match\n" + textwrap.fill(row.productDisplayName, 33),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    sample_records.append(
        {
            "id": row.id,
            "teacher_path": str(teacher_path.relative_to(ROOT)),
            "highres_path": row.highres_image_path,
            "teacher_size": list(sizes[0]),
            "highres_size": list(sizes[1]),
            "teacher_sha256": compute_sha256(teacher_path),
            "highres_sha256": compute_sha256(highres_path),
        }
    )
fig.suptitle(
    "Test image identity check — first two product IDs in each reference class", fontsize=15
)
fig.subplots_adjust(hspace=0.68, top=0.85, bottom=0.15, wspace=0.15)
fig.savefig(REPORT / "test_highres_image_matches.png", dpi=150, bbox_inches="tight")
plt.close(fig)
(REPORT / "image_match_sample.json").write_text(json.dumps(sample_records, indent=2) + "\n")

probabilities = pd.read_csv(
    REPORT / "gender_test_probabilities.csv", keep_default_na=False, float_precision="round_trip"
)
assert probabilities.id.tolist() == test.id.tolist()
diagnostics = {}
for name, selected in (
    ("name_overlap", test.normalized_name_in_development),
    ("no_name_overlap", ~test.normalized_name_in_development),
):
    frame = test.loc[selected]
    labels = frame.reference_gender.map(dict(zip(CLASSES, range(5)))).to_numpy()
    values = probabilities.loc[selected, probability_columns(CLASSES)].to_numpy()
    diagnostics[name] = classification_metrics(labels, values, CLASSES)
timing = pd.read_json(REPORT / "inference_timing.json")
folds = pd.read_csv(REPORT / "individual_fold_scores.csv")
diagnostics["inference_seconds"] = float(timing.seconds.sum())
diagnostics["mean_single_fold_scores"] = (
    folds.groupby("scope")[["accuracy", "macro_f1"]].mean().to_dict("index")
)
diagnostics["test_women_share"] = 4472 / 5829
diagnostics["test_name_overlap_share"] = 797 / 5829
(REPORT / "supplementary_analysis.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in diagnostics.items() if k not in {"name_overlap", "no_name_overlap"}},
        indent=2,
    )
)
for name in ("name_overlap", "no_name_overlap"):
    print(name, {k: diagnostics[name][k] for k in ("support", "accuracy", "macro_f1")})
