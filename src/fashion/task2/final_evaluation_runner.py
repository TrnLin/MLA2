"""Two-phase execution for the frozen Task 2 Season final evaluation."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from fashion.config import (
    LABEL_MAPS_JSON,
    PREDICTION_MANIFEST_CSV,
    ROOT,
    SPLITS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_FIGURE_DIR,
    TASK2_MODEL_MANIFEST_JSON,
    TASK2_SELECTION_FREEZE_JSON,
    TEACHER_TRAIN_CSV,
    TEST_CSV,
)
from fashion.data.dataset import (
    get_samples,
    load_splits,
    load_splits_for_final_evaluation,
)
from fashion.data.hashing import compute_sha256
from fashion.task2.calibration import risk_coverage_curve, top_label_reliability_bins
from fashion.task2.final_evaluation import (
    PROBABILITY_COLUMNS,
    FinalEvaluationSpec,
    build_b0_prediction_frame,
    build_holdout_slice_assignments,
    build_official_predictions,
    fit_development_slice_reference,
    load_final_evaluation_spec,
    score_prediction_frame,
    summarise_grouped_bootstrap,
    validate_prediction_frame,
)
from fashion.task2.inference import SeasonBundle, load_season_bundle
from fashion.task2.robustness import PerturbedTensorTransform, RobustnessCondition
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics

FINAL_EVALUATION_DIR = TASK2_EVIDENCE_DIR / "final_evaluation"
FINAL_EVALUATION_FIGURE_DIR = TASK2_FIGURE_DIR / "final_evaluation"
FROZEN_REGISTRY_PATH = TASK2_EVIDENCE_DIR / "final_handoff/registry_snapshot.csv"
PREDICTION_RECEIPT_PATH = FINAL_EVALUATION_DIR / "prediction_receipt.json"
UNLOCK_RECEIPT_PATH = FINAL_EVALUATION_DIR / "unlock_receipt.json"
EVALUATION_MANIFEST_PATH = FINAL_EVALUATION_DIR / "evaluation_manifest.json"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _portable(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the project root: {resolved}") from error


def _artifact_record(path: str | Path, *, root: Path, rows: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evaluation artifact does not exist: {resolved}")
    record: dict[str, Any] = {
        "path": _portable(resolved, root),
        "sha256": compute_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "tracked_files_dirty": bool(tracked_status)}


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "executable": sys.executable,
    }


class _LabelFreeImageDataset(Dataset[dict[str, Any]]):
    """Ordered image manifest for inference without target columns."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        transform: PerturbedTensorTransform,
        project_root: str | Path,
    ) -> None:
        required = {"id", "path"}
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"prediction manifest is missing columns: {missing}")
        if frame.empty or frame["id"].duplicated().any():
            raise ValueError("prediction manifest must contain unique image IDs")
        self.frame = frame.loc[:, ["id", "path"]].reset_index(drop=True)
        self.transform = transform
        self.root = Path(project_root).resolve()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative = Path(str(row["path"]))
        if relative.is_absolute():
            raise ValueError(f"prediction image path must be project-relative: {relative}")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"prediction image path escapes project root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"prediction image does not exist: {path}")
        return {
            "id": int(row["id"]),
            "image": self.transform(path),
        }


def predict_image_frame(
    bundle: SeasonBundle,
    frame: pd.DataFrame,
    *,
    condition: RobustnessCondition,
    project_root: str | Path = ROOT,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict raw and frozen-temperature probabilities without reading labels."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("prediction batch size and worker count are invalid")
    root = Path(project_root).resolve()
    transform = PerturbedTensorTransform(stats=bundle.transform.stats, condition=condition)
    dataset = _LabelFreeImageDataset(frame, transform=transform, project_root=root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bundle.device.type == "cuda",
        persistent_workers=False,
    )
    model = bundle.model.to(bundle.device).eval()
    predictor = getattr(model, "predict_season_logits", None)
    if not callable(predictor):
        raise TypeError("frozen I2 model lacks image-only Season inference")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
        torch.cuda.synchronize(bundle.device)
    identifiers: list[int] = []
    raw_parts: list[np.ndarray] = []
    calibrated_parts: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(bundle.device, non_blocking=True)
            logits = predictor(images)
            if not isinstance(logits, torch.Tensor) or logits.shape != (
                len(images),
                len(bundle.labels),
            ):
                raise ValueError("frozen I2 model returned an invalid logits shape")
            raw = torch.softmax(logits.float(), dim=1)
            calibrated = torch.softmax(logits.float() / bundle.temperature, dim=1)
            if not bool(torch.isfinite(raw).all().item()) or not bool(
                torch.isfinite(calibrated).all().item()
            ):
                raise FloatingPointError("frozen I2 model returned non-finite probabilities")
            identifiers.extend(int(value) for value in batch["id"].tolist())
            raw_parts.append(raw.cpu().numpy())
            calibrated_parts.append(calibrated.cpu().numpy())
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    elapsed = time.perf_counter() - started
    raw_probabilities = np.concatenate(raw_parts).astype(np.float64, copy=False)
    probabilities = np.concatenate(calibrated_parts).astype(np.float64, copy=False)
    if not np.array_equal(raw_probabilities.argmax(axis=1), probabilities.argmax(axis=1)):
        raise RuntimeError("frozen temperature unexpectedly changed the predicted class")
    predictions = pd.DataFrame({"id": identifiers})
    predicted_indices = probabilities.argmax(axis=1)
    predictions["y_pred"] = [bundle.labels[index] for index in predicted_indices]
    predictions["raw_confidence"] = raw_probabilities.max(axis=1)
    predictions["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(bundle.labels):
        predictions[f"raw_prob_{label}"] = raw_probabilities[:, index]
        predictions[f"prob_{label}"] = probabilities[:, index]
    validate_prediction_frame(
        predictions,
        expected_ids=frame["id"].astype(int).tolist(),
        labels=bundle.labels,
    )
    peak_vram = None
    if bundle.device.type == "cuda":
        peak_vram = int(torch.cuda.max_memory_allocated(bundle.device))
    runtime = {
        "condition": condition.condition,
        "rows": len(predictions),
        "device": str(bundle.device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "runtime_seconds": elapsed,
        "images_per_second": len(predictions) / elapsed,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
        "peak_vram_bytes": peak_vram,
        "prediction_failures": 0,
    }
    return predictions, runtime


def _verify_image_manifest(frame: pd.DataFrame, *, root: Path) -> str:
    required = {"id", "path", "sha256"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"image manifest is missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for row in frame.loc[:, ["id", "path", "sha256"]].itertuples(index=False):
        path = (root / str(row.path)).resolve()
        try:
            portable = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"image path is outside project root: {path}") from error
        verify_artifact(path, str(row.sha256))
        records.append({"id": int(row.id), "path": portable, "sha256": str(row.sha256)})
    return canonical_sha256(records)


def _write_full_precision_csv(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_write_csv(path, frame, float_format="%.17g")


def _verified_bundle(spec: FinalEvaluationSpec, *, device: str, root: Path) -> SeasonBundle:
    bundle = load_season_bundle(
        TASK2_MODEL_MANIFEST_JSON,
        registry_path=FROZEN_REGISTRY_PATH,
        project_root=root,
        device=device,
    )
    if bundle.run_id != spec.run_id or bundle.bundle_sha256 != spec.bundle_sha256:
        raise ValueError("loaded Season bundle differs from the final evaluation contract")
    if tuple(bundle.labels) != spec.labels or not math.isclose(
        bundle.temperature,
        spec.temperature,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("loaded Season labels or temperature changed")
    return bundle


def build_blind_prediction_evidence(
    *,
    device: str = "cpu",
    batch_size: int = 128,
    num_workers: int = 0,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Freeze holdout and teacher-test predictions before protected labels are opened."""
    root = Path(project_root).resolve()
    spec = load_final_evaluation_spec(project_root=root)
    sealed = load_splits(SPLITS_CSV)
    development = get_samples(sealed, partition="development", target="season").copy()
    holdout = sealed.loc[sealed["partition"].eq("holdout")].copy()
    quarantine = sealed.loc[sealed["partition"].eq("quarantine")].copy()
    if len(holdout) != spec.expected_holdout_rows or len(quarantine) != (
        spec.expected_quarantine_rows
    ):
        raise ValueError("canonical protected partition counts changed")
    if holdout["season"].astype(str).str.strip().ne("").any():
        raise RuntimeError("blind prediction phase received visible holdout Season labels")
    if set(holdout["id"].astype(int)) & set(quarantine["id"].astype(int)):
        raise ValueError("holdout and quarantine IDs overlap")
    holdout = holdout.sort_values("id", kind="stable").reset_index(drop=True)

    test_template = pd.read_csv(TEST_CSV, usecols=["id"])
    test_manifest = pd.read_csv(PREDICTION_MANIFEST_CSV)
    if len(test_template) != spec.expected_test_rows or test_template["id"].duplicated().any():
        raise ValueError("teacher test template coverage changed")
    if test_manifest["id"].duplicated().any() or set(test_manifest["id"].astype(int)) != set(
        test_template["id"].astype(int)
    ):
        raise ValueError("teacher test image manifest differs from the template IDs")
    test_manifest = (
        test_template.merge(test_manifest, on="id", how="left", validate="one_to_one")
        .reset_index(drop=True)
    )

    bundle = _verified_bundle(spec, device=device, root=root)
    holdout_image_hash = _verify_image_manifest(holdout, root=root)
    test_image_hash = _verify_image_manifest(test_manifest, root=root)
    mappings, boundaries, slice_audit = fit_development_slice_reference(development)
    b0_predictions, b0_model = build_b0_prediction_frame(
        development,
        expected_ids=holdout["id"].astype(int).tolist(),
    )

    FINAL_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    holdout_manifest_path = FINAL_EVALUATION_DIR / "holdout_image_manifest.csv"
    test_manifest_path = FINAL_EVALUATION_DIR / "teacher_test_image_manifest.csv"
    mapping_path = FINAL_EVALUATION_DIR / "development_article_type_mapping.csv"
    boundaries_path = FINAL_EVALUATION_DIR / "development_file_size_boundaries.csv"
    b0_path = FINAL_EVALUATION_DIR / "holdout_b0_predictions.csv"
    clean_path = FINAL_EVALUATION_DIR / "holdout_predictions.csv"
    robustness_path = FINAL_EVALUATION_DIR / "holdout_robustness_predictions.csv"
    runtime_path = FINAL_EVALUATION_DIR / "blind_runtime.json"

    holdout_manifest_columns = [
        "id",
        "path",
        "sha256",
        "product_family_group",
        "year",
        "file_size_bytes",
        "mode",
    ]
    _write_full_precision_csv(holdout_manifest_path, holdout.loc[:, holdout_manifest_columns])
    _write_full_precision_csv(test_manifest_path, test_manifest.loc[:, ["id", "path", "sha256"]])
    _write_full_precision_csv(mapping_path, mappings)
    _write_full_precision_csv(boundaries_path, boundaries)
    _write_full_precision_csv(b0_path, b0_predictions)

    runtime_rows: list[dict[str, Any]] = []
    robustness_parts: list[pd.DataFrame] = []
    clean_predictions: pd.DataFrame | None = None
    for condition in spec.conditions:
        predictions, runtime = predict_image_frame(
            bundle,
            holdout,
            condition=condition,
            project_root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        runtime["dataset"] = "internal_holdout"
        runtime_rows.append(runtime)
        robust = predictions.loc[:, ["id", "y_pred", "confidence", *PROBABILITY_COLUMNS]].copy()
        robust.insert(1, "condition", condition.condition)
        robustness_parts.append(robust)
        if condition.condition == "clean":
            clean_predictions = predictions
    if clean_predictions is None:
        raise RuntimeError("final evaluation did not execute the clean holdout condition")
    robustness = pd.concat(robustness_parts, ignore_index=True)
    _write_full_precision_csv(clean_path, clean_predictions)
    _write_full_precision_csv(robustness_path, robustness)

    test_condition = spec.conditions[0]
    test_predictions, test_runtime = predict_image_frame(
        bundle,
        test_manifest,
        condition=test_condition,
        project_root=root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_runtime["dataset"] = "teacher_test"
    runtime_rows.append(test_runtime)
    official = build_official_predictions(
        test_predictions,
        expected_ids=test_template["id"].astype(int).tolist(),
    )
    _write_full_precision_csv(spec.official_output_path, official)

    runtime_payload = {
        "schema_version": "1.0.0",
        "phase": "blind_prediction_before_holdout_label_access",
        "environment": _runtime_environment(),
        "runs": runtime_rows,
    }
    atomic_write_json(runtime_path, runtime_payload)
    git_state = _git_state(root)
    if git_state["tracked_files_dirty"]:
        raise RuntimeError(
            "tracked source changed during blind prediction; receipt was not written"
        )
    artifacts = {
        "holdout_image_manifest": _artifact_record(
            holdout_manifest_path, root=root, rows=len(holdout)
        ),
        "teacher_test_image_manifest": _artifact_record(
            test_manifest_path, root=root, rows=len(test_manifest)
        ),
        "development_article_type_mapping": _artifact_record(
            mapping_path, root=root, rows=len(mappings)
        ),
        "development_file_size_boundaries": _artifact_record(
            boundaries_path, root=root, rows=len(boundaries)
        ),
        "holdout_b0_predictions": _artifact_record(b0_path, root=root, rows=len(b0_predictions)),
        "holdout_predictions": _artifact_record(clean_path, root=root, rows=len(clean_predictions)),
        "holdout_robustness_predictions": _artifact_record(
            robustness_path, root=root, rows=len(robustness)
        ),
        "season_test_predictions": _artifact_record(
            spec.official_output_path, root=root, rows=len(official)
        ),
        "runtime": _artifact_record(runtime_path, root=root),
    }
    receipt = {
        "schema_version": "1.0.0",
        "evaluation_id": spec.evaluation_id,
        "phase": "blind_prediction_before_holdout_label_access",
        "created_at_utc": _utc_now(),
        "authorization": "explicit_user_instruction_in_current_task",
        "labels_opened": False,
        "teacher_test_scored": False,
        "model_changed": False,
        "retuning_allowed": False,
        "git": git_state,
        "model": {
            "run_id": bundle.run_id,
            "bundle_sha256": bundle.bundle_sha256,
            "manifest_sha256": bundle.manifest_sha256,
            "temperature": bundle.temperature,
            "labels": list(bundle.labels),
            "scratch": True,
            "image_only_inference": True,
        },
        "inputs": {
            "config": _artifact_record(spec.config_path, root=root),
            "splits": _artifact_record(SPLITS_CSV, root=root),
            "label_maps": _artifact_record(LABEL_MAPS_JSON, root=root),
            "model_manifest": _artifact_record(TASK2_MODEL_MANIFEST_JSON, root=root),
            "registry_snapshot": _artifact_record(FROZEN_REGISTRY_PATH, root=root, rows=1),
            "holdout_image_set_sha256": holdout_image_hash,
            "teacher_test_image_set_sha256": test_image_hash,
        },
        "coverage": {
            "development_fit_rows": len(development),
            "holdout_rows": len(holdout),
            "quarantine_rows_excluded": len(quarantine),
            "teacher_test_rows": len(test_manifest),
            "holdout_conditions": [condition.condition for condition in spec.conditions],
        },
        "b0": {
            "fit_scope": "all_valid_development_rows_only",
            "majority_label": b0_model.majority_label,
            "class_counts": dict(zip(b0_model.labels, b0_model.class_counts, strict=True)),
            "class_probabilities": dict(
                zip(b0_model.labels, b0_model.class_probabilities, strict=True)
            ),
            "training_id_sha256": b0_model.training_id_sha256,
        },
        "slice_fit": slice_audit,
        "artifacts": artifacts,
    }
    atomic_write_json(PREDICTION_RECEIPT_PATH, receipt)
    return receipt


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _verify_prediction_receipt(*, root: Path, spec: FinalEvaluationSpec) -> dict[str, Any]:
    receipt = _read_json(PREDICTION_RECEIPT_PATH)
    if receipt.get("schema_version") != "1.0.0" or receipt.get("evaluation_id") != (
        spec.evaluation_id
    ):
        raise ValueError("blind prediction receipt identity changed")
    if receipt.get("labels_opened") is not False or receipt.get("model_changed") is not False:
        raise ValueError("blind prediction receipt does not preserve the evaluation boundary")
    model = receipt.get("model", {})
    if model.get("run_id") != spec.run_id or model.get("bundle_sha256") != spec.bundle_sha256:
        raise ValueError("blind prediction receipt references a different model")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("blind prediction receipt contains no artifact ledger")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"blind artifact record is invalid: {name}")
        path = (root / str(record["path"])).resolve()
        verify_artifact(path, str(record["sha256"]))
        if int(record["bytes"]) != path.stat().st_size:
            raise ValueError(f"blind artifact byte count changed: {name}")
    return receipt


def _raw_prediction_variant(clean: pd.DataFrame) -> pd.DataFrame:
    raw = clean.loc[
        :,
        ["id", "y_pred", *[f"raw_prob_{label}" for label in SEASON_LABELS]],
    ].copy()
    return raw.rename(
        columns={f"raw_prob_{label}": f"prob_{label}" for label in SEASON_LABELS}
    )


def _per_class_table(metrics_by_model: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, metrics in metrics_by_model.items():
        for label in SEASON_LABELS:
            values = metrics["per_class"][label]
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "support": int(values["support"]),
                    "precision": float(values["precision"]),
                    "recall": float(values["recall"]),
                    "f1": float(values["f1"]),
                }
            )
    return pd.DataFrame(rows)


def _confusion_tables(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_parts = []
    normalised_parts = []
    for model, metrics in metrics_by_model.items():
        counts = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        denominators = counts.sum(axis=1, keepdims=True)
        normalised = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=np.float64),
            where=denominators > 0,
        )
        count_frame = pd.DataFrame(counts, columns=SEASON_LABELS)
        count_frame.insert(0, "true_label", SEASON_LABELS)
        count_frame.insert(0, "model", model)
        normalised_frame = pd.DataFrame(normalised, columns=SEASON_LABELS)
        normalised_frame.insert(0, "true_label", SEASON_LABELS)
        normalised_frame.insert(0, "model", model)
        count_parts.append(count_frame)
        normalised_parts.append(normalised_frame)
    return pd.concat(count_parts, ignore_index=True), pd.concat(
        normalised_parts,
        ignore_index=True,
    )


def _calibration_tables(
    clean: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_by_id = truth.set_index("id")["season"]
    true = clean["id"].map(truth_by_id).astype(str).to_numpy()
    identifiers = clean["id"].astype(int).to_numpy()
    rows = []
    reliability_parts = []
    risk_parts = []
    for variant, columns in (
        ("raw", [f"raw_prob_{label}" for label in SEASON_LABELS]),
        ("frozen_temperature", list(PROBABILITY_COLUMNS)),
    ):
        probabilities = clean.loc[:, columns].to_numpy(dtype=np.float64)
        metrics = multiclass_metrics(true, probabilities=probabilities, labels=SEASON_LABELS)
        rows.append(
            {
                "variant": variant,
                "support": len(clean),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "mean_confidence": float(probabilities.max(axis=1).mean()),
            }
        )
        reliability = top_label_reliability_bins(
            true,
            probabilities,
            labels=SEASON_LABELS,
            ece_bins=15,
        )
        reliability.insert(0, "variant", variant)
        reliability_parts.append(reliability)
        risk = risk_coverage_curve(
            identifiers,
            true,
            probabilities,
            labels=SEASON_LABELS,
            coverage_start=0.1,
            coverage_stop=1.0,
            coverage_step=0.01,
        )
        risk.insert(0, "variant", variant)
        risk_parts.append(risk)
    summary = pd.DataFrame(rows)
    raw = summary.loc[summary["variant"].eq("raw")].iloc[0]
    frozen = summary.loc[summary["variant"].eq("frozen_temperature")].iloc[0]
    if not math.isclose(float(raw["accuracy"]), float(frozen["accuracy"]), abs_tol=0.0):
        raise RuntimeError("temperature scaling changed holdout accuracy")
    if not math.isclose(float(raw["macro_f1"]), float(frozen["macro_f1"]), abs_tol=0.0):
        raise RuntimeError("temperature scaling changed holdout macro-F1")
    for metric in ("nll", "brier", "ece", "mean_confidence"):
        summary[f"delta_{metric}_vs_raw"] = summary[metric] - float(raw[metric])
    risk = pd.concat(risk_parts, ignore_index=True)
    budgets = []
    for variant in ("raw", "frozen_temperature"):
        variant_rows = risk.loc[risk["variant"].eq(variant)]
        for review_budget in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            coverage = round(1.0 - review_budget, 10)
            row = variant_rows.loc[
                np.isclose(
                    variant_rows["requested_coverage"],
                    coverage,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(row) != 1:
                raise RuntimeError("review budget is missing from holdout risk curve")
            payload = row.iloc[0].to_dict()
            payload["review_budget"] = review_budget
            budgets.append(payload)
    return (
        summary,
        pd.concat(reliability_parts, ignore_index=True),
        risk,
        pd.DataFrame(budgets),
    )


def _slice_metrics(scored: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    frame = scored.merge(assignments, on="id", how="left", validate="one_to_one")
    families = (
        "article_type_shortcut",
        "acquisition_year",
        "file_size_quartile",
        "product_family_size",
        "image_mode",
    )
    rows = []
    for family in families:
        if frame[family].isna().any():
            raise ValueError(f"holdout slice assignment is incomplete: {family}")
        for name, group in frame.groupby(family, sort=True, observed=True):
            probabilities = group.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
            metrics = multiclass_metrics(
                group["y_true"].astype(str).to_numpy(),
                probabilities=probabilities,
                labels=SEASON_LABELS,
                y_pred=group["y_pred"].astype(str).to_numpy(),
            )
            rows.append(
                {
                    "slice_family": family,
                    "slice_name": str(name),
                    "support": len(group),
                    "low_support": len(group) < 100,
                    "labels_present": int(group["y_true"].nunique()),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "mean_confidence": float(probabilities.max(axis=1).mean()),
                    **{
                        f"recall_{label}": metrics["per_class"][label]["recall"]
                        for label in SEASON_LABELS
                    },
                }
            )
    return pd.DataFrame(rows)


def _robustness_metrics(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    expected_conditions = {
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    }
    if set(predictions["condition"].astype(str)) != expected_conditions:
        raise ValueError("holdout robustness prediction conditions are incomplete")
    truth_by_id = truth.set_index("id")["season"]
    clean = predictions.loc[predictions["condition"].eq("clean")].set_index("id")
    rows = []
    for condition, group in predictions.groupby("condition", sort=True, observed=True):
        group = group.copy()
        true = group["id"].map(truth_by_id).astype(str).to_numpy()
        probabilities = group.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
        metrics = multiclass_metrics(
            true,
            probabilities=probabilities,
            labels=SEASON_LABELS,
            y_pred=group["y_pred"].astype(str).to_numpy(),
        )
        clean_predictions = group["id"].map(clean["y_pred"]).astype(str).to_numpy()
        rows.append(
            {
                "condition": str(condition),
                "support": len(group),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "spring_recall": metrics["per_class"]["Spring"]["recall"],
                "spring_f1": metrics["per_class"]["Spring"]["f1"],
                "mean_confidence": float(probabilities.max(axis=1).mean()),
                "prediction_agreement_with_clean": float(
                    np.mean(group["y_pred"].astype(str).to_numpy() == clean_predictions)
                ),
            }
        )
    table = pd.DataFrame(rows)
    clean_row = table.loc[table["condition"].eq("clean")].iloc[0]
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece",
        "spring_recall",
        "spring_f1",
        "mean_confidence",
    ):
        table[f"delta_{metric}_vs_clean"] = table[metric] - float(clean_row[metric])
    return table


def _error_tables(
    scored: pd.DataFrame,
    holdout: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_columns = [
        "id",
        "path",
        "articleType",
        "year",
        "productDisplayName",
        "product_family_group",
    ]
    frame = scored.merge(
        holdout.loc[:, context_columns],
        on="id",
        how="left",
        validate="one_to_one",
    ).merge(assignments, on="id", how="left", validate="one_to_one")
    frame["correct"] = frame["y_true"].eq(frame["y_pred"])
    frame["true_probability"] = [
        float(frame.at[index, f"prob_{label}"])
        for index, label in zip(frame.index, frame["y_true"], strict=True)
    ]
    errors = frame.loc[~frame["correct"]].copy()
    routes = (
        errors.groupby(["y_true", "y_pred"], observed=True)
        .agg(
            count=("id", "size"),
            mean_confidence=("confidence", "mean"),
            high_confidence_count=("confidence", lambda values: int(values.ge(0.8).sum())),
        )
        .reset_index()
        .sort_values(["count", "y_true", "y_pred"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    examples = (
        errors.sort_values(["confidence", "id"], ascending=[False, True])
        .groupby("y_true", sort=True, group_keys=False, observed=True)
        .head(3)
        .reset_index(drop=True)
    )
    return frame, routes, examples


def _save_figure(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_scorecard(scorecard: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    primary = scorecard.loc[scorecard["model"].isin(["B0 majority", "I2 frozen"])]
    x = np.arange(len(primary))
    width = 0.35
    axes[0].bar(x - width / 2, primary["macro_f1"], width, label="Macro-F1")
    axes[0].bar(x + width / 2, primary["balanced_accuracy"], width, label="Balanced accuracy")
    axes[0].set_xticks(x, primary["model"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Independent holdout scorecard")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        primary["model"],
        primary["holdout_minus_development_macro_f1"],
        color=["#7f8c8d", "#2a9d8f"],
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Holdout minus development macro-F1")
    axes[1].set_ylabel("Difference")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_per_class_confusion(
    per_class: pd.DataFrame,
    confusion_normalised: pd.DataFrame,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    selected = per_class.loc[per_class["model"].eq("I2 frozen")]
    axes[0].bar(selected["label"], selected["f1"], color="#457b9d")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("F1")
    axes[0].set_title("I2 holdout per-class F1")
    axes[0].grid(axis="y", alpha=0.2)
    matrix = confusion_normalised.loc[
        confusion_normalised["model"].eq("I2 frozen"), SEASON_LABELS
    ].to_numpy(dtype=float)
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks(range(4), SEASON_LABELS)
    axes[1].set_yticks(range(4), SEASON_LABELS)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("I2 row-normalised confusion")
    for row in range(4):
        for column in range(4):
            axes[1].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_bootstrap(intervals: pd.DataFrame, path: Path) -> Path:
    names = ("i2_macro_f1", "i2_minus_b0_macro_f1", "i2_spring_f1")
    selected = intervals.set_index("metric").loc[list(names)].reset_index()
    y = np.arange(len(selected))
    lower = selected["median"] - selected["lower_95"]
    upper = selected["upper_95"] - selected["median"]
    figure, axis = plt.subplots(figsize=(8, 4.3))
    axis.errorbar(
        selected["median"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
        color="#264653",
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, selected["metric"])
    axis.set_xlabel("Bootstrap estimate with 95% percentile interval")
    axis.set_title("Product-family blocked uncertainty (10,000 draws)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_calibration_risk(
    reliability: pd.DataFrame,
    risk: pd.DataFrame,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant, rows in reliability.groupby("variant", sort=False, observed=True):
        visible = rows.loc[rows["count"].gt(0)]
        axes[0].plot(
            visible["mean_confidence"],
            visible["empirical_accuracy"],
            marker="o",
            label=str(variant),
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Mean confidence")
    axes[0].set_ylabel("Empirical accuracy")
    axes[0].set_title("Holdout reliability")
    axes[0].legend()
    for variant, rows in risk.groupby("variant", sort=False, observed=True):
        axes[1].plot(rows["coverage"], rows["selective_risk"], label=str(variant))
    axes[1].set_xlabel("Automatic coverage")
    axes[1].set_ylabel("Risk (1 - accuracy)")
    axes[1].set_title("Confidence-ranked risk-coverage")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_slices_robustness(
    slices: pd.DataFrame,
    robustness: pd.DataFrame,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    slice_rows = slices.copy()
    slice_rows["display"] = slice_rows["slice_family"] + ": " + slice_rows["slice_name"]
    axes[0].barh(slice_rows["display"], slice_rows["macro_f1"], color="#6d597a")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Fixed-label macro-F1")
    axes[0].set_title("Predeclared holdout slices")
    axes[0].grid(axis="x", alpha=0.2)
    ordered = robustness.set_index("condition").loc[
        [
            "clean",
            "jpeg_quality_85",
            "brightness_0_85",
            "brightness_1_15",
            "gaussian_blur_radius_1",
        ]
    ]
    x = np.arange(len(ordered))
    axes[1].bar(x - 0.18, ordered["macro_f1"], 0.36, label="Macro-F1")
    axes[1].bar(x + 0.18, ordered["spring_recall"], 0.36, label="Spring recall")
    axes[1].set_xticks(x, ordered.index, rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Frozen perturbation stress tests")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_error_examples(examples: pd.DataFrame, *, root: Path, path: Path) -> Path:
    columns = 3
    rows = max(1, math.ceil(len(examples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, rows * 3.2), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, examples.itertuples(index=False), strict=False):
        image_path = (root / str(row.path)).resolve()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            axis.imshow(image)
        axis.set_title(
            f"ID {row.id}: {row.y_true} → {row.y_pred}\np={row.confidence:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle("Deterministic highest-confidence holdout errors", fontsize=13)
    figure.tight_layout()
    return _save_figure(figure, path)


def _deployment_summary(root: Path, receipt: Mapping[str, Any]) -> pd.DataFrame:
    cost_path = root / "results/evidence/task2/robustness_cost/deployment_cost.csv"
    costs = pd.read_csv(cost_path)
    selected = costs.loc[costs["candidate"].eq("I2")].copy()
    if selected.empty:
        raise ValueError("development deployment-cost evidence lacks I2")
    runtime = _read_json(root / str(receipt["artifacts"]["runtime"]["path"]))
    clean_holdout = next(
        row
        for row in runtime["runs"]
        if row["dataset"] == "internal_holdout" and row["condition"] == "clean"
    )
    row = {
        "candidate": "I2",
        "parameter_count": 1_206_112,
        "bundle_bytes": int((root / "models/task2_season.pt").stat().st_size),
        "development_cpu_end_to_end_median_ms": float(
            selected.loc[selected["device"].eq("cpu"), "end_to_end_median_ms"].iloc[0]
        ),
        "development_cuda_end_to_end_median_ms": (
            float(selected.loc[selected["device"].eq("cuda"), "end_to_end_median_ms"].iloc[0])
            if selected["device"].eq("cuda").any()
            else np.nan
        ),
        "holdout_clean_device": clean_holdout["device"],
        "holdout_clean_runtime_seconds": clean_holdout["runtime_seconds"],
        "holdout_clean_images_per_second": clean_holdout["images_per_second"],
        "holdout_clean_prediction_failures": clean_holdout["prediction_failures"],
        "holdout_clean_peak_vram_bytes": clean_holdout["peak_vram_bytes"],
    }
    return pd.DataFrame([row])


def score_internal_holdout(
    *,
    evaluation_unlocked: bool = False,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Open protected Season labels once and score only pre-frozen predictions."""
    if not evaluation_unlocked:
        raise ValueError("internal holdout scoring requires evaluation_unlocked=True")
    root = Path(project_root).resolve()
    spec = load_final_evaluation_spec(project_root=root)
    receipt = _verify_prediction_receipt(root=root, spec=spec)

    unlocked = load_splits_for_final_evaluation(
        SPLITS_CSV,
        evaluation_unlocked=True,
        raw_teacher_csv=TEACHER_TRAIN_CSV,
    )
    holdout_all = unlocked.loc[unlocked["partition"].eq("holdout")].copy()
    quarantine = unlocked.loc[unlocked["partition"].eq("quarantine")].copy()
    valid_holdout = get_samples(holdout_all, target="season").copy()
    if len(holdout_all) != spec.expected_holdout_rows or len(quarantine) != (
        spec.expected_quarantine_rows
    ):
        raise ValueError("protected split counts changed during scoring")
    if valid_holdout["id"].duplicated().any():
        raise ValueError("valid holdout IDs are not unique")
    if set(valid_holdout["id"].astype(int)) & set(quarantine["id"].astype(int)):
        raise ValueError("quarantine IDs entered the scored holdout")
    valid_holdout = valid_holdout.sort_values("id", kind="stable").reset_index(drop=True)

    unlock = {
        "schema_version": "1.0.0",
        "evaluation_id": spec.evaluation_id,
        "opened_at_utc": _utc_now(),
        "authorization": "explicit_user_instruction_in_current_task",
        "prediction_receipt": _artifact_record(PREDICTION_RECEIPT_PATH, root=root),
        "raw_teacher_csv": _artifact_record(TEACHER_TRAIN_CSV, root=root),
        "holdout_opened": True,
        "holdout_rows": len(holdout_all),
        "valid_season_rows": len(valid_holdout),
        "blank_or_invalid_season_rows": len(holdout_all) - len(valid_holdout),
        "quarantine_rows_excluded": len(quarantine),
        "model_retrained": False,
        "model_retuned": False,
        "winner_changed": False,
        "temperature_refit": False,
    }
    atomic_write_json(UNLOCK_RECEIPT_PATH, unlock)

    artifact_paths = {
        name: root / str(record["path"])
        for name, record in receipt["artifacts"].items()
    }
    clean = pd.read_csv(artifact_paths["holdout_predictions"])
    b0 = pd.read_csv(artifact_paths["holdout_b0_predictions"])
    robustness_predictions = pd.read_csv(
        artifact_paths["holdout_robustness_predictions"]
    )
    valid_ids = valid_holdout["id"].astype(int).tolist()
    clean = clean.loc[clean["id"].isin(valid_ids)].copy()
    b0 = b0.loc[b0["id"].isin(valid_ids)].copy()
    robustness_predictions = robustness_predictions.loc[
        robustness_predictions["id"].isin(valid_ids)
    ].copy()
    truth = valid_holdout.loc[:, ["id", "season"]].copy()

    i2_metrics, i2_scored = score_prediction_frame(clean, truth)
    raw_metrics, _ = score_prediction_frame(_raw_prediction_variant(clean), truth)
    b0_metrics, b0_scored = score_prediction_frame(b0, truth)
    metrics_by_model = {"B0 majority": b0_metrics, "I2 frozen": i2_metrics}

    selection = _read_json(TASK2_SELECTION_FREEZE_JSON)
    b0_development = _read_json(
        root / "results/evidence/task2/b0_majority/pooled_metrics.json"
    )["macro_f1"]
    i2_development = selection["primary_development_evidence"]["pooled_oof_macro_f1"]
    scorecard_rows = []
    for model, role, metrics, development_metric in (
        ("B0 majority", "primary prior baseline", b0_metrics, b0_development),
        ("I2 frozen", "ultimate judgement", i2_metrics, i2_development),
    ):
        scorecard_rows.append(
            {
                "model": model,
                "role": role,
                "n_samples": metrics["n_samples"],
                **{
                    name: metrics[name]
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "weighted_f1",
                        "nll",
                        "brier",
                        "ece",
                    )
                },
                "development_macro_f1": development_metric,
                "holdout_minus_development_macro_f1": metrics["macro_f1"]
                - development_metric,
            }
        )
    scorecard = pd.DataFrame(scorecard_rows)
    per_class = _per_class_table(metrics_by_model)
    confusion_counts, confusion_normalised = _confusion_tables(metrics_by_model)

    b0_by_id = b0_scored.set_index("id")["y_pred"]
    groups = valid_holdout.set_index("id").loc[
        i2_scored["id"].astype(int), "product_family_group"
    ]
    bootstrap = summarise_grouped_bootstrap(
        i2_scored["y_true"].astype(str).to_numpy(),
        groups.astype(str).to_numpy(),
        i2_scored["id"].map(b0_by_id).astype(str).to_numpy(),
        i2_scored["y_pred"].astype(str).to_numpy(),
        replicates=spec.bootstrap_replicates,
        random_seed=spec.bootstrap_seed,
    )
    calibration_summary, reliability, risk, review_budgets = _calibration_tables(clean, truth)

    development = get_samples(unlocked, partition="development", target="season").copy()
    mappings, boundaries, _ = fit_development_slice_reference(development)
    saved_mappings = pd.read_csv(artifact_paths["development_article_type_mapping"])
    saved_boundaries = pd.read_csv(artifact_paths["development_file_size_boundaries"])
    pd.testing.assert_frame_equal(mappings, saved_mappings, check_dtype=False, check_exact=False)
    pd.testing.assert_frame_equal(
        boundaries,
        saved_boundaries,
        check_dtype=False,
        check_exact=False,
    )
    assignments, slice_audit = build_holdout_slice_assignments(development, valid_holdout)
    slices = _slice_metrics(i2_scored, assignments)
    robustness = _robustness_metrics(robustness_predictions, truth)
    full_predictions, error_routes, error_examples = _error_tables(
        i2_scored,
        valid_holdout,
        assignments,
    )
    deployment = _deployment_summary(root, receipt)

    output_paths = {
        "unlock_receipt": UNLOCK_RECEIPT_PATH,
        "holdout_predictions_and_labels": FINAL_EVALUATION_DIR
        / "holdout_predictions_and_labels.csv",
        "holdout_scorecard": FINAL_EVALUATION_DIR / "holdout_scorecard.csv",
        "holdout_metrics": FINAL_EVALUATION_DIR / "holdout_metrics.json",
        "holdout_per_class": FINAL_EVALUATION_DIR / "holdout_per_class.csv",
        "holdout_confusion_counts": FINAL_EVALUATION_DIR / "holdout_confusion_counts.csv",
        "holdout_confusion_row_normalised": FINAL_EVALUATION_DIR
        / "holdout_confusion_row_normalised.csv",
        "holdout_bootstrap_intervals": FINAL_EVALUATION_DIR
        / "holdout_bootstrap_intervals.csv",
        "holdout_calibration_summary": FINAL_EVALUATION_DIR
        / "holdout_calibration_summary.csv",
        "holdout_reliability_bins": FINAL_EVALUATION_DIR / "holdout_reliability_bins.csv",
        "holdout_risk_coverage": FINAL_EVALUATION_DIR / "holdout_risk_coverage.csv",
        "holdout_review_budgets": FINAL_EVALUATION_DIR / "holdout_review_budgets.csv",
        "holdout_slice_metrics": FINAL_EVALUATION_DIR / "holdout_slice_metrics.csv",
        "holdout_robustness_metrics": FINAL_EVALUATION_DIR
        / "holdout_robustness_metrics.csv",
        "holdout_error_routes": FINAL_EVALUATION_DIR / "holdout_error_routes.csv",
        "holdout_error_examples": FINAL_EVALUATION_DIR / "holdout_error_examples.csv",
        "deployment_summary": FINAL_EVALUATION_DIR / "deployment_summary.csv",
    }
    b0_predictions_by_id = b0.set_index("id")["y_pred"]
    scored_output = full_predictions.rename(
        columns={"y_true": "actual_season", "y_pred": "i2_prediction"}
    )
    scored_output.insert(
        scored_output.columns.get_loc("i2_prediction"),
        "b0_prediction",
        scored_output["id"].map(b0_predictions_by_id),
    )
    _write_full_precision_csv(output_paths["holdout_predictions_and_labels"], scored_output)
    _write_full_precision_csv(output_paths["holdout_scorecard"], scorecard)
    atomic_write_json(
        output_paths["holdout_metrics"],
        {"B0_majority": b0_metrics, "I2_raw": raw_metrics, "I2_frozen_temperature": i2_metrics},
    )
    _write_full_precision_csv(output_paths["holdout_per_class"], per_class)
    _write_full_precision_csv(output_paths["holdout_confusion_counts"], confusion_counts)
    _write_full_precision_csv(
        output_paths["holdout_confusion_row_normalised"],
        confusion_normalised,
    )
    _write_full_precision_csv(output_paths["holdout_bootstrap_intervals"], bootstrap)
    _write_full_precision_csv(output_paths["holdout_calibration_summary"], calibration_summary)
    _write_full_precision_csv(output_paths["holdout_reliability_bins"], reliability)
    _write_full_precision_csv(output_paths["holdout_risk_coverage"], risk)
    _write_full_precision_csv(output_paths["holdout_review_budgets"], review_budgets)
    _write_full_precision_csv(output_paths["holdout_slice_metrics"], slices)
    _write_full_precision_csv(output_paths["holdout_robustness_metrics"], robustness)
    _write_full_precision_csv(output_paths["holdout_error_routes"], error_routes)
    _write_full_precision_csv(output_paths["holdout_error_examples"], error_examples)
    _write_full_precision_csv(output_paths["deployment_summary"], deployment)

    figure_paths = {
        "holdout_scorecard": _plot_scorecard(
            scorecard,
            FINAL_EVALUATION_FIGURE_DIR / "holdout_scorecard.png",
        ),
        "per_class_confusion": _plot_per_class_confusion(
            per_class,
            confusion_normalised,
            FINAL_EVALUATION_FIGURE_DIR / "holdout_per_class_confusion.png",
        ),
        "bootstrap": _plot_bootstrap(
            bootstrap,
            FINAL_EVALUATION_FIGURE_DIR / "holdout_bootstrap_intervals.png",
        ),
        "calibration_risk": _plot_calibration_risk(
            reliability,
            risk,
            FINAL_EVALUATION_FIGURE_DIR / "holdout_calibration_risk.png",
        ),
        "slices_robustness": _plot_slices_robustness(
            slices,
            robustness,
            FINAL_EVALUATION_FIGURE_DIR / "holdout_slices_robustness.png",
        ),
        "error_examples": _plot_error_examples(
            error_examples,
            root=root,
            path=FINAL_EVALUATION_FIGURE_DIR / "holdout_error_examples.png",
        ),
    }

    i2_score = scorecard.loc[scorecard["model"].eq("I2 frozen")].iloc[0]
    delta_interval = bootstrap.loc[
        bootstrap["metric"].eq("i2_minus_b0_macro_f1")
    ].iloc[0]
    judgement = {
        "schema_version": "1.0.0",
        "evaluation_id": spec.evaluation_id,
        "ultimate_judgement": "conditionally_viable_for_catalogue_decision_support",
        "independent_holdout_macro_f1": float(i2_score["macro_f1"]),
        "independent_holdout_balanced_accuracy": float(i2_score["balanced_accuracy"]),
        "development_to_holdout_macro_f1_change": float(
            i2_score["holdout_minus_development_macro_f1"]
        ),
        "i2_minus_b0_macro_f1_interval_95": [
            float(delta_interval["lower_95"]),
            float(delta_interval["upper_95"]),
        ],
        "beats_b0_with_positive_grouped_interval": bool(delta_interval["lower_95"] > 0),
        "deployment_boundary": (
            "Use as decision support with human review; do not treat Season as an objective "
            "visual fact or force low-confidence/shifted cases into automation."
        ),
        "model_retrained_after_unlock": False,
        "temperature_refit_after_unlock": False,
        "winner_changed_after_unlock": False,
        "limitations": [
            "Season labels are contextual and can be ambiguous from pixels alone.",
            "The internal holdout comes from the same source as development data.",
            "Confidence ranking is diagnostic because no business-cost threshold was selected.",
            "Brightness and other image shifts can materially change performance.",
            (
                "Grad-CAM development evidence is non-causal and was not used to alter "
                "the final model."
            ),
        ],
    }
    judgement_path = FINAL_EVALUATION_DIR / "ultimate_judgement.json"
    atomic_write_json(judgement_path, judgement)
    output_paths["ultimate_judgement"] = judgement_path

    artifacts = {
        name: _artifact_record(
            path,
            root=root,
            rows=(len(pd.read_csv(path)) if Path(path).suffix == ".csv" else None),
        )
        for name, path in output_paths.items()
    }
    artifacts.update(
        {
            name: _artifact_record(path, root=root)
            for name, path in figure_paths.items()
        }
    )
    manifest = {
        "schema_version": "1.0.0",
        "evaluation_id": spec.evaluation_id,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "prediction_receipt": _artifact_record(PREDICTION_RECEIPT_PATH, root=root),
        "unlock_receipt": _artifact_record(UNLOCK_RECEIPT_PATH, root=root),
        "model": {
            "candidate": "I2",
            "run_id": spec.run_id,
            "bundle_sha256": spec.bundle_sha256,
            "scratch": True,
            "image_only_inference": True,
            "model_changed_after_unlock": False,
        },
        "coverage": {
            "holdout_rows": len(holdout_all),
            "valid_season_rows": len(valid_holdout),
            "invalid_season_rows": len(holdout_all) - len(valid_holdout),
            "quarantine_rows_excluded": len(quarantine),
            "product_family_groups": int(groups.nunique()),
            "teacher_test_rows": spec.expected_test_rows,
            "prediction_failures": 0,
        },
        "protocol": {
            "primary_metric": "macro_f1",
            "primary_baseline": "B0 development-prior majority",
            "b1_holdout_status": "not_run_no_prefrozen_predictions",
            "bootstrap_replicates": spec.bootstrap_replicates,
            "bootstrap_seed": spec.bootstrap_seed,
            "slice_fit": slice_audit,
            "temperature": spec.temperature,
            "threshold_selected": False,
        },
        "official_test": {
            "scored": False,
            "output": receipt["artifacts"]["season_test_predictions"],
        },
        "artifacts": artifacts,
    }
    atomic_write_json(EVALUATION_MANIFEST_PATH, manifest)
    return manifest


def load_verified_final_evaluation(
    *,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Verify the completed final-evaluation ledger for replay-only notebooks."""
    root = Path(project_root).resolve()
    spec = load_final_evaluation_spec(project_root=root)
    manifest = _read_json(EVALUATION_MANIFEST_PATH)
    if manifest.get("schema_version") != "1.0.0" or manifest.get("status") != "complete":
        raise ValueError("Task 2 final evaluation is not complete")
    if manifest.get("evaluation_id") != spec.evaluation_id:
        raise ValueError("Task 2 final evaluation identity changed")
    model = manifest.get("model", {})
    if model.get("run_id") != spec.run_id or model.get("bundle_sha256") != spec.bundle_sha256:
        raise ValueError("Task 2 final evaluation model identity changed")
    verify_artifact(PREDICTION_RECEIPT_PATH, manifest["prediction_receipt"]["sha256"])
    verify_artifact(UNLOCK_RECEIPT_PATH, manifest["unlock_receipt"]["sha256"])
    for name, record in manifest.get("artifacts", {}).items():
        path = root / str(record["path"])
        verify_artifact(path, str(record["sha256"]))
        if int(record["bytes"]) != path.stat().st_size:
            raise ValueError(f"final evaluation artifact size changed: {name}")
    official = manifest.get("official_test", {}).get("output", {})
    verify_artifact(root / str(official["path"]), str(official["sha256"]))
    return manifest


__all__ = [
    "EVALUATION_MANIFEST_PATH",
    "FINAL_EVALUATION_DIR",
    "FINAL_EVALUATION_FIGURE_DIR",
    "PREDICTION_RECEIPT_PATH",
    "UNLOCK_RECEIPT_PATH",
    "build_blind_prediction_evidence",
    "load_verified_final_evaluation",
    "predict_image_frame",
    "score_internal_holdout",
]
