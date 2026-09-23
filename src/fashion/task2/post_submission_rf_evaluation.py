"""Retrospective holdout evaluation of the post-submission full-development RF."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

from fashion.config import ROOT, SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.final_evaluation import (
    load_final_evaluation_spec,
    score_prediction_frame,
    validate_prediction_frame,
)
from fashion.task2.final_evaluation_runner import (
    FINAL_EVALUATION_DIR,
    PREDICTION_RECEIPT_PATH,
    _LabelFreeImageDataset,
    load_verified_final_evaluation,
)
from fashion.task2.post_submission_experiments import PRIMARY_SEED
from fashion.task2.post_submission_rf_refit import (
    VerifiedRFFit,
    _inside_root,
    _record,
    _verify_record,
    load_verified_rf_fit,
)
from fashion.task2.robustness import PerturbedTensorTransform, verify_image_frame
from fashion.train.artifacts import atomic_write_csv, atomic_write_json, verify_artifact
from fashion.train.metrics import SEASON_LABELS, paired_group_bootstrap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_EVIDENCE_DIR = ROOT / "results/evidence/task2/post_submission/full_refit"
DEFAULT_FIGURE_DIR = ROOT / "results/figures/task2/post_submission/full_refit"
PREDICTION_RECEIPT = DEFAULT_EVIDENCE_DIR / "prediction_receipt.json"
EVALUATION_MANIFEST = DEFAULT_EVIDENCE_DIR / "evaluation_manifest.json"
PROBABILITY_COLUMNS = [f"prob_{label}" for label in SEASON_LABELS]
EVIDENCE_NAMES = frozenset(
    {
        "scorecard.csv",
        "per_class.csv",
        "confusion_counts.csv",
        "paired_predictions.csv",
        "paired_bootstrap_intervals.csv",
        "paired_bootstrap_draws.csv",
        "robustness.csv",
        "cost.csv",
    }
)
FIGURE_NAMES = frozenset(
    {"holdout_scorecard.png", "holdout_bootstrap.png", "holdout_robustness.png"}
)
ExecutionMode = Literal["run", "load", "run_or_load"]


def predict_rf_image_frame(
    bundle: VerifiedRFFit,
    frame: pd.DataFrame,
    *,
    condition: Any,
    project_root: str | Path = ROOT,
    batch_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run image-only CNN embedding extraction followed by RF probabilities."""
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    root = Path(project_root).resolve()
    expected_ids = frame["id"].astype(int).tolist()
    transform = PerturbedTensorTransform(
        stats=bundle.encoder.transform.stats,
        condition=condition,
    )
    dataset = _LabelFreeImageDataset(frame, transform=transform, project_root=root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    model = bundle.encoder.model.eval()
    device = bundle.encoder.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    identifiers: list[int] = []
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            embedding = model.base_model.forward_embedding(image)
            if embedding.shape != (len(image), 256) or not bool(
                torch.isfinite(embedding).all().item()
            ):
                raise ValueError("RF inference received invalid 256-D embeddings")
            identifiers.extend(int(value) for value in batch["id"].tolist())
            pieces.append(embedding.float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    features = np.concatenate(pieces).astype(np.float32, copy=False)
    if identifiers != expected_ids or features.shape != (len(expected_ids), 256):
        raise ValueError("RF inference did not preserve the ordered holdout IDs")
    raw = bundle.forest.predict_proba(features)
    probabilities = np.zeros((len(features), len(SEASON_LABELS)), dtype=np.float64)
    for column, class_index in enumerate(bundle.forest.classes_):
        probabilities[:, int(class_index)] = raw[:, column]
    prediction = pd.DataFrame(
        {
            "id": identifiers,
            "y_pred": np.asarray(SEASON_LABELS, dtype=object)[probabilities.argmax(axis=1)],
        }
    )
    for column, label in enumerate(SEASON_LABELS):
        prediction[f"prob_{label}"] = probabilities[:, column]
    validate_prediction_frame(prediction, expected_ids=expected_ids)
    elapsed = time.perf_counter() - started
    runtime = {
        "condition": condition.condition,
        "rows": len(prediction),
        "device": str(device),
        "batch_size": batch_size,
        "runtime_seconds": elapsed,
        "images_per_second": len(prediction) / elapsed,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
    }
    return prediction, runtime


def _verified_holdout_images(root: Path) -> pd.DataFrame:
    """Read only protected image identities before producing RF predictions."""
    prior = json.loads(
        _inside_root(
            FINAL_EVALUATION_DIR / "evaluation_manifest.json", root
        ).read_text(encoding="utf-8")
    )
    verify_artifact(
        _inside_root(PREDICTION_RECEIPT_PATH, root),
        prior["prediction_receipt"]["sha256"],
    )
    receipt = json.loads(
        _inside_root(PREDICTION_RECEIPT_PATH, root).read_text(encoding="utf-8")
    )
    original_image_record = receipt["artifacts"]["holdout_image_manifest"]
    image_path = _inside_root(original_image_record["path"], root)
    if image_path != _inside_root(
        FINAL_EVALUATION_DIR / "holdout_image_manifest.csv", root
    ):
        raise ValueError("original holdout image manifest path changed")
    verify_artifact(image_path, original_image_record["sha256"])
    if image_path.stat().st_size != original_image_record["bytes"]:
        raise ValueError("original holdout image manifest size changed")
    frame = pd.read_csv(image_path).sort_values("id", kind="stable").reset_index(drop=True)
    split = load_splits(_inside_root(SPLITS_CSV, root))
    holdout = split.loc[split["partition"].eq("holdout")]
    if len(frame) != 5_778 or set(frame["id"].astype(int)) != set(
        holdout["id"].astype(int)
    ) or frame["id"].duplicated().any():
        raise ValueError("RF evaluation holdout image IDs changed")
    if holdout["season"].astype(str).str.strip().ne("").any():
        raise ValueError("RF prediction phase received protected labels")
    verify_image_frame(frame, project_root=root)
    return frame


def _save_predictions(
    bundle: VerifiedRFFit,
    *,
    root: Path,
    evidence: Path,
    mode: ExecutionMode,
) -> dict[str, Any]:
    receipt_path = _inside_root(PREDICTION_RECEIPT, root)
    clean_path = evidence / "holdout_rf_predictions.csv"
    robust_path = evidence / "holdout_rf_robustness_predictions.csv"
    runtime_path = evidence / "prediction_runtime.json"
    if receipt_path.exists():
        if mode == "run":
            raise FileExistsError("RF holdout predictions already exist")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _verify_rf_prediction_receipt(receipt, bundle=bundle, root=root, evidence=evidence)
        return receipt
    if mode == "load":
        raise FileNotFoundError(receipt_path)
    if any(path.exists() for path in (clean_path, robust_path, runtime_path)):
        raise FileExistsError("partial RF holdout predictions need audit before rerun")

    frame = _verified_holdout_images(root)
    spec = load_final_evaluation_spec(project_root=root)
    outputs = []
    runtimes = []
    clean: pd.DataFrame | None = None
    for condition in spec.conditions:
        prediction, runtime = predict_rf_image_frame(
            bundle, frame, condition=condition, project_root=root
        )
        prediction.insert(1, "condition", condition.condition)
        outputs.append(prediction)
        runtimes.append(runtime)
        if condition.condition == "clean":
            clean = prediction.drop(columns="condition")
    if clean is None:
        raise ValueError("RF evaluation has no clean condition")
    robust = pd.concat(outputs, ignore_index=True)
    if len(robust) != 5 * len(frame):
        raise ValueError("RF robustness output coverage changed")
    atomic_write_csv(clean_path, clean, float_format="%.17g")
    atomic_write_csv(robust_path, robust, float_format="%.17g")
    atomic_write_json(runtime_path, {"runs": runtimes})
    receipt = {
        "schema_version": "1.0.0",
        "phase": "post_submission_rf_prediction_before_this_comparison",
        "holdout_previously_opened": True,
        "labels_used_in_prediction": False,
        "model_manifest_sha256": compute_sha256(bundle.manifest_path),
        "image_manifest": _record(
            _inside_root(FINAL_EVALUATION_DIR / "holdout_image_manifest.csv", root), root
        ),
        "conditions": [condition.condition for condition in spec.conditions],
        "artifacts": {
            "clean": _record(clean_path, root),
            "robustness": _record(robust_path, root),
            "runtime": _record(runtime_path, root),
        },
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def _verify_rf_prediction_receipt(
    receipt: dict[str, Any],
    *,
    bundle: VerifiedRFFit,
    root: Path,
    evidence: Path,
) -> None:
    spec = load_final_evaluation_spec(project_root=root)
    expected_conditions = [condition.condition for condition in spec.conditions]
    if (
        receipt.get("schema_version") != "1.0.0"
        or receipt.get("phase") != "post_submission_rf_prediction_before_this_comparison"
        or receipt.get("holdout_previously_opened") is not True
        or receipt.get("labels_used_in_prediction") is not False
        or receipt.get("model_manifest_sha256") != compute_sha256(bundle.manifest_path)
        or receipt.get("conditions") != expected_conditions
        or set(receipt.get("artifacts", {})) != {"clean", "robustness", "runtime"}
    ):
        raise ValueError("RF prediction receipt contract changed")
    image_path = _inside_root(FINAL_EVALUATION_DIR / "holdout_image_manifest.csv", root)
    _verify_record(receipt["image_manifest"], root, image_path)
    for name, filename in (
        ("clean", "holdout_rf_predictions.csv"),
        ("robustness", "holdout_rf_robustness_predictions.csv"),
        ("runtime", "prediction_runtime.json"),
    ):
        _verify_record(receipt["artifacts"][name], root, evidence / filename)


def _comparison_tables(
    rf: pd.DataFrame,
    i2: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    scored: list[pd.DataFrame] = []
    for name, prediction in (("I2 frozen", i2), ("I2 embedding + RF", rf)):
        metrics, rows = score_prediction_frame(prediction, truth)
        scored.append(rows)
        summaries.append(
            {
                "model": name,
                **{
                    field: metrics[field]
                    for field in (
                        "n_samples", "macro_f1", "accuracy", "balanced_accuracy",
                        "weighted_f1", "nll", "brier", "ece",
                    )
                },
            }
        )
        for label in SEASON_LABELS:
            per_class.append({"model": name, "class": label, **metrics["per_class"][label]})
        for index, true_label in enumerate(SEASON_LABELS):
            for predicted_index, predicted_label in enumerate(SEASON_LABELS):
                confusion.append(
                    {
                        "model": name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": metrics["confusion_matrix"][index][predicted_index],
                    }
                )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(per_class),
        pd.DataFrame(confusion),
        scored[0].loc[:, ["id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "i2_prediction"}
        ).merge(
            scored[1].loc[:, ["id", "y_pred"]].rename(
                columns={"y_pred": "rf_prediction"}
            ),
            on="id",
            validate="one_to_one",
        ),
    )


def _bootstrap(
    paired: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = truth.loc[:, ["id", "product_family_group"]]
    aligned = paired.merge(groups, on="id", how="left", validate="one_to_one")
    if aligned["product_family_group"].isna().any():
        raise ValueError("family groups are missing from the paired holdout comparison")
    draws = paired_group_bootstrap(
        aligned["y_true"].astype(str).to_numpy(),
        aligned["product_family_group"].astype(str).to_numpy(),
        {
            "RF_minus_I2": (
                aligned["i2_prediction"].astype(str).to_numpy(),
                aligned["rf_prediction"].astype(str).to_numpy(),
            )
        },
        labels=SEASON_LABELS,
        replicates=10_000,
        random_seed=PRIMARY_SEED,
    )
    metrics = (
        ("macro_f1", "b_minus_a_macro_f1"),
        ("accuracy", "b_minus_a_accuracy"),
        ("spring_f1", "b_minus_a_f1_spring"),
    )
    intervals = pd.DataFrame(
        [
            {
                "comparison": "RF_minus_I2",
                "metric": name,
                "lower_95": float(draws[column].quantile(0.025)),
                "median": float(draws[column].median()),
                "upper_95": float(draws[column].quantile(0.975)),
                "probability_delta_above_zero": float((draws[column] > 0).mean()),
                "replicates": len(draws),
                "sampled_group_count": int(draws["sampled_group_count"].iloc[0]),
            }
            for name, column in metrics
        ]
    )
    return intervals, draws


def _robustness_table(
    rf: pd.DataFrame,
    i2: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    conditions = set(rf["condition"].astype(str))
    if conditions != set(i2["condition"].astype(str)):
        raise ValueError("RF and I2 robustness conditions differ")
    expected_ids = truth["id"].astype(int).tolist()
    for condition in sorted(conditions):
        for name, source in (("I2 frozen", i2), ("I2 embedding + RF", rf)):
            selected = source.loc[source["condition"].eq(condition)].drop(
                columns="condition"
            )
            validate_prediction_frame(selected, expected_ids=expected_ids)
            metrics, _ = score_prediction_frame(selected, truth)
            rows.append(
                {
                    "condition": condition,
                    "model": name,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "spring_recall": metrics["per_class"]["Spring"]["recall"],
                    "spring_f1": metrics["per_class"]["Spring"]["f1"],
                    "ece": metrics["ece"],
                    "rows": metrics["n_samples"],
                }
            )
    result = pd.DataFrame(rows)
    for metric in ("macro_f1", "accuracy", "spring_recall"):
        clean = result.loc[result["condition"].eq("clean")].set_index("model")[metric]
        result[f"delta_{metric}_from_clean"] = result[metric] - result["model"].map(
            clean
        )
    return result


def _plot_scorecard(scorecard: pd.DataFrame, per_class: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    colors = ("#457b9d", "#2a9d8f")
    positions = np.arange(2)
    for offset, metric in ((-0.18, "macro_f1"), (0.18, "balanced_accuracy")):
        axes[0].bar(
            positions + offset,
            scorecard[metric].to_numpy(),
            0.34,
            label=metric.replace("_", " "),
        )
    axes[0].set_xticks(positions, ["I2", "I2 + RF"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Same 5,778 holdout images")
    axes[0].legend()
    labels = list(SEASON_LABELS)
    for index, model in enumerate(("I2 frozen", "I2 embedding + RF")):
        selected = per_class.loc[per_class["model"].eq(model)].set_index("class")
        axes[1].bar(
            np.arange(4) + (-0.18 if index == 0 else 0.18),
            selected.loc[labels, "f1"],
            0.34,
            color=colors[index],
            label=model,
        )
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("F1 for each Season class")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bootstrap(draws: pd.DataFrame, intervals: pd.DataFrame, path: Path) -> None:
    values = draws["b_minus_a_macro_f1"].to_numpy(float)
    interval = intervals.loc[intervals["metric"].eq("macro_f1")].iloc[0]
    fig, axis = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    axis.hist(values, bins=60, density=True, color="#80b8b7", alpha=0.8)
    axis.axvspan(interval["lower_95"], interval["upper_95"], color="#f4a261", alpha=0.22)
    axis.axvline(0, color="#d62828", linestyle="--", label="no effect")
    axis.axvline(interval["median"], color="#264653", label="median")
    axis.set_title("RF minus I2: product-family bootstrap (10,000 draws)")
    axis.set_xlabel("Holdout macro-F1 difference")
    axis.set_ylabel("Density")
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(frame: pd.DataFrame, path: Path) -> None:
    order = (
        "clean", "jpeg_quality_85", "brightness_0_85", "brightness_1_15",
        "gaussian_blur_radius_1",
    )
    fig, axis = plt.subplots(figsize=(10, 4.3), constrained_layout=True)
    for model, color in (("I2 frozen", "#457b9d"), ("I2 embedding + RF", "#2a9d8f")):
        selected = frame.loc[frame["model"].eq(model)].set_index("condition").loc[list(order)]
        axis.plot(range(len(order)), selected["macro_f1"], marker="o", label=model, color=color)
    axis.set_xticks(range(len(order)), order, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Fixed holdout image changes; same 5,778 IDs")
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_verified_rf_holdout_evaluation(
    *, project_root: str | Path = ROOT
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = _inside_root(EVALUATION_MANIFEST, root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "1.0.0"
        or payload.get("role") != "retrospective_post_submission_holdout_comparison"
        or payload.get("status") != "complete"
        or payload.get("holdout_previously_opened") is not True
        or payload.get("used_for_training_or_tuning") is not False
    ):
        raise ValueError("RF holdout evaluation boundary changed")
    bundle = load_verified_rf_fit(project_root=root, device="cpu")
    if payload["rf_manifest_sha256"] != compute_sha256(bundle.manifest_path):
        raise ValueError("RF holdout evaluation model changed")
    original = load_verified_final_evaluation(project_root=root)
    if payload["original_evaluation_sha256"] != compute_sha256(
        _inside_root(FINAL_EVALUATION_DIR / "evaluation_manifest.json", root)
    ) or original["status"] != "complete":
        raise ValueError("original I2 evaluation changed")
    if (
        payload.get("rows") != 5_778
        or set(payload.get("artifacts", {})) != EVIDENCE_NAMES
        or set(payload.get("figures", {})) != FIGURE_NAMES
    ):
        raise ValueError("RF holdout evidence ledger is incomplete")
    receipt_path = _verify_record(
        payload["prediction_receipt"], root, PREDICTION_RECEIPT
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _verify_rf_prediction_receipt(
        receipt, bundle=bundle, root=root, evidence=_inside_root(DEFAULT_EVIDENCE_DIR, root)
    )
    if payload.get("prediction_phase") != receipt["phase"]:
        raise ValueError("RF holdout evaluation phase changed")
    for name, record in payload["artifacts"].items():
        _verify_record(record, root, DEFAULT_EVIDENCE_DIR / name)
    for name, record in payload["figures"].items():
        _verify_record(record, root, DEFAULT_FIGURE_DIR / name)
    scorecard = pd.read_csv(_inside_root(DEFAULT_EVIDENCE_DIR / "scorecard.csv", root))
    if (
        len(scorecard) != 2
        or set(scorecard["model"]) != {"I2 frozen", "I2 embedding + RF"}
        or set(scorecard["n_samples"]) != {5_778}
    ):
        raise ValueError("RF holdout scorecard coverage changed")
    measured = scorecard.set_index("model")["macro_f1"]
    expected_metrics = {
        "I2": float(measured["I2 frozen"]),
        "RF": float(measured["I2 embedding + RF"]),
        "RF_minus_I2": float(
            measured["I2 embedding + RF"] - measured["I2 frozen"]
        ),
    }
    if set(payload.get("macro_f1", {})) != set(expected_metrics) or any(
        not math.isclose(
            float(payload["macro_f1"][name]), value, rel_tol=0, abs_tol=1e-12
        )
        for name, value in expected_metrics.items()
    ):
        raise ValueError("RF holdout summary differs from its scorecard")
    return payload


def evaluate_full_development_rf(
    *,
    mode: ExecutionMode = "run_or_load",
    project_root: str | Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score fixed RF against I2 on the already-opened internal holdout."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    root = Path(project_root).resolve()
    evidence = _inside_root(DEFAULT_EVIDENCE_DIR, root)
    figures = _inside_root(DEFAULT_FIGURE_DIR, root)
    evaluation_path = _inside_root(EVALUATION_MANIFEST, root)
    if evaluation_path.exists():
        if mode == "run":
            raise FileExistsError("RF holdout evaluation already exists")
        return load_verified_rf_holdout_evaluation(project_root=root)
    if mode == "load":
        raise FileNotFoundError(evaluation_path)
    bundle = load_verified_rf_fit(project_root=root, device=device)
    evidence.mkdir(parents=True, exist_ok=True)
    prediction_receipt = _save_predictions(bundle, root=root, evidence=evidence, mode=mode)
    original_manifest = load_verified_final_evaluation(project_root=root)
    truth_all = pd.read_csv(
        _inside_root(FINAL_EVALUATION_DIR / "holdout_predictions_and_labels.csv", root)
    )
    truth = truth_all.loc[:, ["id", "actual_season", "product_family_group"]].rename(
        columns={"actual_season": "season"}
    )
    if len(truth) != 5_778 or truth["id"].duplicated().any():
        raise ValueError("scored holdout truth coverage changed")
    rf = pd.read_csv(evidence / "holdout_rf_predictions.csv")
    i2 = pd.read_csv(_inside_root(FINAL_EVALUATION_DIR / "holdout_predictions.csv", root))
    expected_ids = truth["id"].astype(int).tolist()
    validate_prediction_frame(rf, expected_ids=expected_ids)
    validate_prediction_frame(i2, expected_ids=expected_ids)
    scorecard, per_class, confusion, paired = _comparison_tables(rf, i2, truth)
    original_metrics = json.loads(
        _inside_root(FINAL_EVALUATION_DIR / "holdout_metrics.json", root).read_text(
            encoding="utf-8"
        )
    )
    measured_original = scorecard.loc[scorecard["model"].eq("I2 frozen")].iloc[0]
    for metric in ("macro_f1", "accuracy", "balanced_accuracy", "nll", "brier", "ece"):
        if not math.isclose(
            float(measured_original[metric]),
            float(original_metrics["I2_frozen_temperature"][metric]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"original I2 holdout {metric} no longer reproduces")
    intervals, draws = _bootstrap(paired, truth)
    rf_robust = pd.read_csv(evidence / "holdout_rf_robustness_predictions.csv")
    i2_robust = pd.read_csv(
        _inside_root(FINAL_EVALUATION_DIR / "holdout_robustness_predictions.csv", root)
    )
    robustness = _robustness_table(rf_robust, i2_robust, truth)
    runtime = json.loads((evidence / "prediction_runtime.json").read_text(encoding="utf-8"))
    old_runtime = json.loads(
        _inside_root(FINAL_EVALUATION_DIR / "blind_runtime.json", root).read_text(
            encoding="utf-8"
        )
    )
    old_clean = next(
        row for row in old_runtime["runs"]
        if row["condition"] == "clean" and row["dataset"] == "internal_holdout"
    )
    new_clean = next(row for row in runtime["runs"] if row["condition"] == "clean")
    cost = pd.DataFrame(
        [
            {
                "model": "I2 frozen",
                "device": old_clean["device"],
                "clean_runtime_seconds": old_clean["runtime_seconds"],
                "images_per_second": old_clean["images_per_second"],
                "bundle_bytes": bundle.encoder.bundle_path.stat().st_size,
            },
            {
                "model": "I2 embedding + RF",
                "device": new_clean["device"],
                "clean_runtime_seconds": new_clean["runtime_seconds"],
                "images_per_second": new_clean["images_per_second"],
                "bundle_bytes": (
                    bundle.encoder.bundle_path.stat().st_size + bundle.forest_path.stat().st_size
                ),
            },
        ]
    )
    outputs = {
        "scorecard.csv": scorecard,
        "per_class.csv": per_class,
        "confusion_counts.csv": confusion,
        "paired_predictions.csv": paired,
        "paired_bootstrap_intervals.csv": intervals,
        "paired_bootstrap_draws.csv": draws,
        "robustness.csv": robustness,
        "cost.csv": cost,
    }
    for filename, frame in outputs.items():
        atomic_write_csv(evidence / filename, frame, float_format="%.17g")
    figures.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "holdout_scorecard.png": figures / "holdout_scorecard.png",
        "holdout_bootstrap.png": figures / "holdout_bootstrap.png",
        "holdout_robustness.png": figures / "holdout_robustness.png",
    }
    _plot_scorecard(scorecard, per_class, figure_paths["holdout_scorecard.png"])
    _plot_bootstrap(draws, intervals, figure_paths["holdout_bootstrap.png"])
    _plot_robustness(robustness, figure_paths["holdout_robustness.png"])

    baseline = scorecard.loc[scorecard["model"].eq("I2 frozen")].iloc[0]
    challenger = scorecard.loc[scorecard["model"].eq("I2 embedding + RF")].iloc[0]
    manifest = {
        "schema_version": "1.0.0",
        "role": "retrospective_post_submission_holdout_comparison",
        "status": "complete",
        "holdout_previously_opened": True,
        "used_for_training_or_tuning": False,
        "claim_boundary": (
            "The internal holdout was already opened for the submitted I2 model. "
            "This is a retrospective post-submission comparison, not a fresh "
            "independent model-selection test or a new submission claim."
        ),
        "rf_manifest_sha256": compute_sha256(bundle.manifest_path),
        "original_evaluation_sha256": compute_sha256(
            _inside_root(FINAL_EVALUATION_DIR / "evaluation_manifest.json", root)
        ),
        "original_evaluation_id": original_manifest["evaluation_id"],
        "prediction_receipt": _record(
            _inside_root(PREDICTION_RECEIPT, root), root
        ),
        "prediction_phase": prediction_receipt["phase"],
        "rows": len(truth),
        "macro_f1": {
            "I2": float(baseline["macro_f1"]),
            "RF": float(challenger["macro_f1"]),
            "RF_minus_I2": float(challenger["macro_f1"] - baseline["macro_f1"]),
        },
        "artifacts": {name: _record(evidence / name, root) for name in outputs},
        "figures": {name: _record(path, root) for name, path in figure_paths.items()},
    }
    atomic_write_json(evaluation_path, manifest)
    return load_verified_rf_holdout_evaluation(project_root=root)
