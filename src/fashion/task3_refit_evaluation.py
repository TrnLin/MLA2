"""Saved final-evaluation analysis for the selected Gender and Usage refits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.data.hashing import compute_sha256
from fashion.task3_development import verify_mixup_selection
from fashion.task3_final import verify_usage_final
from fashion.train.metrics import classification_metrics

GENDER_REPORT = Path("reports/task3/gender_mixup_refit_holdout_20260911")
USAGE_REPORT = Path("reports/task3/usage_e8_refit_holdout_20260907")


def load_selected_evaluation(root: Path) -> dict:
    """Verify identities and derive metrics from only each selected refit's predictions."""
    models = {
        "Gender MixUp": verify_mixup_selection(root),
        "Usage E8": verify_usage_final(root, saved_training_source=True),
    }
    reports = {"Gender MixUp": GENDER_REPORT, "Usage E8": USAGE_REPORT}
    result = {}
    for target, model in models.items():
        folder = root / reports[target]
        provenance = json.loads((folder / "evaluation_provenance.json").read_text())
        # E8's old lock also includes comparison tables. Only selected-model inputs are needed.
        required = [
            "evaluation.json",
            "holdout_predictions_and_labels.csv",
            "prediction_freeze.json",
        ]
        if target == "Gender MixUp":
            required += [
                "refit_holdout_probabilities.csv",
                "holdout_image_manifest.csv",
                "evaluation_plan.json",
            ]
            if (folder / "selected_refit_bootstrap.json").exists():
                required.append("selected_refit_bootstrap.json")
        for name in required:
            assert compute_sha256(folder / name) == provenance["outputs"][name], name
        evaluation = json.loads((folder / "evaluation.json").read_text())
        assert evaluation["run_id"] == model["run_id"]
        assert evaluation["checkpoint_sha256"] == model["checkpoint"]["sha256"]
        freeze = json.loads((folder / "prediction_freeze.json").read_text())
        probability_path = folder / "refit_holdout_probabilities.csv"
        expected = (
            freeze["predictions_sha256"]
            if target == "Gender MixUp"
            else freeze["outputs"][probability_path.name]
        )
        assert compute_sha256(probability_path) == expected
        predictions = pd.read_csv(probability_path, keep_default_na=False).set_index("id")
        labels = pd.read_csv(
            folder / "holdout_predictions_and_labels.csv", keep_default_na=False
        ).set_index("id")
        assert labels.index.is_unique and predictions.index.is_unique
        assert set(predictions.index) == set(labels.index) and len(labels) == 5778
        predictions = predictions.loc[labels.index]
        classes = model["class_names"]
        columns = (
            [f"probability_{i}_{name}" for i, name in enumerate(classes)]
            if target == "Gender MixUp"
            else [f"probability_{name}" for name in classes]
        )
        probabilities = predictions[columns].to_numpy(dtype=float)
        actual_column = "actual_gender" if target == "Gender MixUp" else "actual_usage"
        truth = (
            labels[actual_column].map(dict(zip(classes, range(len(classes))))).to_numpy(dtype=int)
        )
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
        metrics = classification_metrics(truth, probabilities, classes)
        recorded = (
            evaluation["metrics"]
            if target == "Gender MixUp"
            else evaluation["metrics"]["Single refit"]
        )
        for key in ["macro_f1", "accuracy", "nll", "brier", "ece_15"]:
            np.testing.assert_allclose(metrics[key], recorded[key], atol=1e-12, rtol=0)
        scored = labels[["product_family_group"]].copy()
        scored["true_index"] = truth
        scored["predicted_index"] = probabilities.argmax(1)
        scored["actual"] = np.asarray(classes)[truth]
        scored["predicted"] = np.asarray(classes)[probabilities.argmax(1)]
        scored["confidence"] = probabilities.max(1)
        if target == "Gender MixUp":
            scored["path"] = labels.path
            scored["articleType"] = labels.articleType
        result[target] = {
            "manifest": model,
            "metrics": metrics,
            "rows": scored.reset_index(),
            "probabilities": probabilities,
            "evaluation": evaluation,
            "freeze": freeze,
        }
    assert set(result["Gender MixUp"]["rows"].id) == set(result["Usage E8"]["rows"].id)
    return result


def family_score_interval(
    rows: pd.DataFrame, classes: list[str], *, draws: int = 10000, seed: int = 2753
) -> dict:
    """Single-model macro-F1 percentile interval from whole-family resampling."""
    if rows.empty or rows.id.duplicated().any() or draws < 2:
        raise ValueError("Need unique scored rows and at least two bootstrap draws")
    rows = rows.sort_values("id")
    families, inverse = np.unique(rows.product_family_group.astype(str), return_inverse=True)
    if len(families) < 2:
        raise ValueError("Need at least two independent family groups")
    k = len(classes)
    truth = rows.true_index.to_numpy(dtype=int)
    predicted = rows.predicted_index.to_numpy(dtype=int)
    one_hot = np.eye(k, dtype=np.int64)
    counts = np.concatenate(
        [one_hot[truth], one_hot[predicted], one_hot[truth] * (truth == predicted)[:, None]], axis=1
    )
    groups = np.zeros((len(families), 3 * k), dtype=np.int64)
    np.add.at(groups, inverse, counts)

    def score(total):
        support, predictions, correct = np.split(total, 3)
        denominator = support + predictions
        return float(
            np.divide(2 * correct, denominator, out=np.zeros(k), where=denominator > 0).mean()
        )

    rng = np.random.default_rng(seed)
    scores = np.asarray(
        [score(groups[rng.integers(0, len(groups), len(groups))].sum(0)) for _ in range(draws)]
    )
    return {
        "macro_f1": score(counts.sum(0)),
        "lower_95": float(np.quantile(scores, 0.025)),
        "upper_95": float(np.quantile(scores, 0.975)),
        "families": len(families),
        "draws": draws,
        "seed": seed,
        "method": "whole-family percentile; fixed class map",
    }


def class_table(evaluation: dict) -> pd.DataFrame:
    """Return per-class counts and precision/recall/F1 in readable units."""
    metrics = evaluation["metrics"]
    table = pd.DataFrame(metrics["per_class"]).set_index("class_name")
    table["correct"] = np.diag(np.asarray(metrics["confusion_matrix"]))
    table["missed"] = table.support - table.correct
    table[["precision", "recall", "f1"]] *= 100
    return table[["support", "predicted_count", "correct", "missed", "precision", "recall", "f1"]]


def reliability_table(evaluation: dict, bins: int = 15) -> pd.DataFrame:
    """Bin raw confidence; this does not fit a confidence correction."""
    rows = evaluation["rows"].copy()
    rows["correct"] = rows.actual.eq(rows.predicted)
    rows["bin"] = np.minimum((rows.confidence * bins).astype(int), bins - 1)
    return rows.groupby("bin").agg(
        images=("id", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean")
    )


def confusion_figure(evaluation: dict, title: str):
    """Draw counts and row-normalized shares together; absent rows stay explicit."""
    classes = evaluation["manifest"]["class_names"]
    counts = np.asarray(evaluation["metrics"]["confusion_matrix"])
    support = counts.sum(1, keepdims=True)
    shares = np.divide(counts, support, out=np.zeros_like(counts, dtype=float), where=support > 0)
    k = len(classes)
    fig, ax = plt.subplots(figsize=(max(7, k), max(5.8, k * 0.8)))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            label = f"{counts[i, j]}\n{shares[i, j]:.0%}" if support[i, 0] else "—"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8 if k > 5 else 10,
                color="white" if shares[i, j] > 0.55 else "#243340",
            )
    ax.set(
        xticks=range(k),
        yticks=range(k),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted class",
        ylabel="Original reference class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig
