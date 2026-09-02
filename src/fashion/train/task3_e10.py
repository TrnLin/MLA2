"""E10 audience-helper metrics, diagnostics, and zero-step evidence contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fashion.data import load_splits
from fashion.train.metrics import classification_metrics
from fashion.train.task3_e9 import gender_semantic_conflicts

GENDER_CLASSES = ("Boys", "Girls", "Men", "Unisex", "Women")
AUDIENCE_CLASSES = ("male audience", "female audience", "unisex audience")
GENDER_TO_AUDIENCE = {
    "Boys": "male audience",
    "Girls": "female audience",
    "Men": "male audience",
    "Unisex": "unisex audience",
    "Women": "female audience",
}
GENDER_INDEX_TO_AUDIENCE_INDEX = np.asarray([0, 1, 0, 2, 1], dtype=np.int64)

GENDER_E10_PARENT_CLEAN_ERRORS = 3_338
GENDER_E10_PARENT_AUDIENCE_OR_UNISEX_ERRORS = 2_959
GENDER_E10_MINIMUM_PARENT_FOCUS_SHARE = 0.85
GENDER_E10_MAXIMUM_FOCUS_ERRORS = 2_663


def _json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gender_audience_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Collapse fixed five-way Gender probabilities into three audience groups."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(GENDER_CLASSES):
        raise ValueError("Gender probabilities must have shape (rows, 5)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Gender probabilities must be finite and non-negative")
    audience = np.column_stack(
        (values[:, 0] + values[:, 2], values[:, 1] + values[:, 4], values[:, 3])
    )
    if not np.allclose(audience.sum(axis=1), values.sum(axis=1), atol=1e-8):
        raise RuntimeError("audience probability aggregation changed total probability")
    return audience


def gender_audience_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Evaluate the derived three-way audience prediction from the five-way head."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= len(GENDER_CLASSES)):
        raise ValueError("Gender labels must use the fixed five-class index order")
    audience_probabilities = gender_audience_probabilities(probabilities)
    audience_labels = GENDER_INDEX_TO_AUDIENCE_INDEX[labels]
    return classification_metrics(
        audience_labels,
        audience_probabilities,
        AUDIENCE_CLASSES,
    )


def gender_audience_error_diagnostics(
    splits: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, object]:
    """Measure clean audience/Unisex errors without changing official evaluation rows."""
    development = splits.loc[
        splits["partition"].eq("development") & splits["has_gender_label"]
    ].copy()
    annotated = gender_semantic_conflicts(development).merge(
        predictions.loc[:, ["id", "true_label", "predicted_label", "confidence"]],
        on="id",
        validate="one_to_one",
    )
    if len(annotated) != len(development):
        raise ValueError("Gender predictions do not cover every labelled development row")
    if not annotated["gender"].equals(annotated["true_label"]):
        raise ValueError("Gender prediction labels disagree with the saved split labels")

    clean = ~annotated["e9_semantic_conflict"].astype(bool)
    error = annotated["true_label"].ne(annotated["predicted_label"])
    same_age_wrong_audience_pairs = {
        ("Boys", "Girls"),
        ("Girls", "Boys"),
        ("Men", "Women"),
        ("Women", "Men"),
    }
    same_age_wrong_audience = pd.Series(
        [
            (truth, predicted) in same_age_wrong_audience_pairs
            for truth, predicted in zip(
                annotated["true_label"], annotated["predicted_label"], strict=True
            )
        ],
        index=annotated.index,
        dtype=bool,
    )
    involves_unisex = annotated["true_label"].eq("Unisex") | annotated["predicted_label"].eq(
        "Unisex"
    )
    focus_error = clean & error & (same_age_wrong_audience | involves_unisex)
    clean_error = clean & error

    annotated["e10_clean_evaluation_row"] = clean
    annotated["e10_clean_error"] = clean_error
    annotated["e10_same_age_wrong_audience"] = clean & error & same_age_wrong_audience
    annotated["e10_involves_unisex"] = clean & error & involves_unisex
    annotated["e10_audience_or_unisex_error"] = focus_error
    clean_errors = int(clean_error.sum())
    focus_errors = int(focus_error.sum())
    summary = {
        "clean_rows": int(clean.sum()),
        "clean_errors": clean_errors,
        "same_age_wrong_audience_errors": int((clean & error & same_age_wrong_audience).sum()),
        "involves_unisex_errors": int((clean & error & involves_unisex).sum()),
        "audience_or_unisex_errors": focus_errors,
        "audience_or_unisex_share_of_clean_errors": (
            float(focus_errors / clean_errors) if clean_errors else 0.0
        ),
        "semantic_conflict_rows_excluded_from_diagnostic_only": int((~clean).sum()),
        "official_validation_rows_unchanged": True,
    }
    return {"summary": summary, "annotated": annotated}


def audit_gender_e10_parent(
    *,
    splits_path: str | Path,
    parent_prediction_path: str | Path,
    parent_metrics_path: str | Path,
    parent_run_ids: Sequence[str],
) -> dict[str, object]:
    """Verify the exact E6 evidence and helper-label coverage before E10 training."""
    if len(tuple(parent_run_ids)) != 5 or len(set(parent_run_ids)) != 5:
        raise ValueError("Gender E10 requires five distinct E6 parent run IDs")
    metrics = json.loads(Path(parent_metrics_path).read_text(encoding="utf-8"))
    saved_parent_ids = tuple(str(value) for value in metrics.get("fold_run_ids", []))
    splits = load_splits(Path(splits_path))
    predictions = pd.read_csv(parent_prediction_path, keep_default_na=False)
    diagnostics = gender_audience_error_diagnostics(splits, predictions)

    development = splits.loc[
        splits["partition"].eq("development") & splits["has_gender_label"]
    ].copy()
    semantic = gender_semantic_conflicts(development)
    conflicts = semantic.loc[semantic["e9_semantic_conflict"]].copy()
    official_audience = conflicts["gender"].map(GENDER_TO_AUDIENCE)
    implied_audience = conflicts["e9_name_implied_gender"].map(GENDER_TO_AUDIENCE)
    consistent_conflicts = bool(
        official_audience.notna().all()
        and implied_audience.notna().all()
        and official_audience.equals(implied_audience)
    )

    development["e10_audience_label"] = development["gender"].map(GENDER_TO_AUDIENCE)
    fold_rows: list[dict[str, object]] = []
    for validation_fold in range(5):
        for split_name, selected in (
            ("training", development["cv_fold"].ne(validation_fold)),
            ("validation", development["cv_fold"].eq(validation_fold)),
        ):
            counts = (
                development.loc[selected, "e10_audience_label"]
                .value_counts()
                .reindex(AUDIENCE_CLASSES, fill_value=0)
            )
            fold_rows.append(
                {
                    "validation_fold": validation_fold,
                    "split": split_name,
                    **{name: int(counts[name]) for name in AUDIENCE_CLASSES},
                    "all_helper_labels_present": bool((counts > 0).all()),
                }
            )
    fold_coverage = pd.DataFrame(fold_rows)
    parent_summary = diagnostics["summary"]
    checks = {
        "exact_parent_run_ids": saved_parent_ids == tuple(parent_run_ids),
        "exact_clean_error_count": (
            parent_summary["clean_errors"] == GENDER_E10_PARENT_CLEAN_ERRORS
        ),
        "exact_audience_or_unisex_error_count": (
            parent_summary["audience_or_unisex_errors"]
            == GENDER_E10_PARENT_AUDIENCE_OR_UNISEX_ERRORS
        ),
        "focus_share_at_least_85_percent": (
            parent_summary["audience_or_unisex_share_of_clean_errors"]
            >= GENDER_E10_MINIMUM_PARENT_FOCUS_SHARE
        ),
        "semantic_conflict_count_is_305": len(conflicts) == 305,
        "all_semantic_conflicts_keep_the_same_helper_label": consistent_conflicts,
        "all_helper_labels_present_in_every_fold": bool(
            fold_coverage["all_helper_labels_present"].all()
        ),
    }
    summary = {
        **parent_summary,
        "parent_run_ids": list(parent_run_ids),
        "helper_mapping": GENDER_TO_AUDIENCE,
        "maximum_accepted_focus_errors": GENDER_E10_MAXIMUM_FOCUS_ERRORS,
        "checks": checks,
        "verified": all(checks.values()),
        "optimizer_steps": 0,
    }
    return {
        "summary": summary,
        "annotated": diagnostics["annotated"],
        "fold_coverage": fold_coverage,
    }


def write_task3_e10_prerun_evidence(
    *,
    splits_path: str | Path,
    parent_prediction_path: str | Path,
    parent_metrics_path: str | Path,
    parent_run_ids: Sequence[str],
    output_dir: str | Path,
) -> dict[str, object]:
    """Write the deterministic E10 audit without constructing an optimiser."""
    output_dir = Path(output_dir)
    audit = audit_gender_e10_parent(
        splits_path=splits_path,
        parent_prediction_path=parent_prediction_path,
        parent_metrics_path=parent_metrics_path,
        parent_run_ids=parent_run_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    diagnostic_path = output_dir / "parent_diagnostic_rows.csv"
    coverage_path = output_dir / "helper_label_fold_coverage.csv"
    _json_dump(audit["summary"], summary_path)
    audit["annotated"].sort_values("id").to_csv(diagnostic_path, index=False)
    audit["fold_coverage"].to_csv(coverage_path, index=False)
    return {
        **audit["summary"],
        "summary_path": str(summary_path),
        "diagnostic_path": str(diagnostic_path),
        "coverage_path": str(coverage_path),
    }
