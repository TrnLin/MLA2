"""Development-only shortcut slices and error tables for Task 2 finalists."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import TASK2_CONFIG_DIR
from fashion.data.dataset import get_cv_split, get_samples
from fashion.task2.multitask_evidence import build_article_type_shortcut_audit
from fashion.train.artifacts import atomic_write_bytes, canonical_sha256
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics

SLICE_ANALYSIS_CONFIG_PATH = TASK2_CONFIG_DIR / "g6_shortcut_error_slices.json"

EXPECTED_CANDIDATE_EXPERIMENTS = (
    ("C2", "g3-c2-t0-resnet18", 2753),
    ("C2", "g5-c2-t0-resnet18-s2026", 2026),
    ("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
    ("I2", "g5-i2-article-type-lambda-0-3-c1-s2026", 2026),
)

SLICE_GROUPS = {
    "article_type_shortcut": (
        "aligned",
        "conflict",
        "unseen_article_type",
        "missing_article_type",
    ),
    "acquisition_year": (
        "dominant_2011_2012",
        "other_years",
        "missing_year",
    ),
    "file_size_quartile": (
        "q1_smallest",
        "q2",
        "q3",
        "q4_largest",
    ),
    "product_family_size": ("singleton", "multirow"),
    "image_mode": ("rgb", "greyscale", "other_mode"),
}

SLICE_ASSIGNMENT_SCOPES = {
    "article_type_shortcut": "majority_mapping_fit_on_four_training_folds_only",
    "acquisition_year": "metadata_after_prediction_only",
    "file_size_quartile": "quartile_boundaries_fit_on_four_training_folds_only",
    "product_family_size": "canonical_product_family_group_after_prediction_only",
    "image_mode": "structural_audit_after_prediction_only",
}

SLICE_CONTRASTS = (
    ("article_type_shortcut", "aligned", "conflict", "conflict_minus_aligned"),
    (
        "acquisition_year",
        "dominant_2011_2012",
        "other_years",
        "other_minus_dominant_2011_2012",
    ),
    (
        "file_size_quartile",
        "q1_smallest",
        "q4_largest",
        "q4_largest_minus_q1_smallest",
    ),
    (
        "product_family_size",
        "singleton",
        "multirow",
        "multirow_minus_singleton",
    ),
    ("image_mode", "rgb", "greyscale", "greyscale_minus_rgb"),
)

PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in SEASON_LABELS)


@dataclass(frozen=True)
class CandidateExperiment:
    """One immutable candidate/seed OOF identity."""

    candidate: str
    experiment_id: str
    seed: int


@dataclass(frozen=True)
class SliceAnalysisSpec:
    """Validated G6 slice-analysis declaration."""

    analysis_id: str
    expected_row_count: int
    candidates: tuple[CandidateExperiment, ...]
    high_confidence_threshold: float
    maximum_ranked_confusions: int
    low_support_threshold: int
    quantiles: tuple[float, float, float]


@dataclass(frozen=True)
class CandidateOOFPack:
    """One complete five-fold OOF pack and its audited registry rows."""

    candidate: str
    experiment_id: str
    seed: int
    oof: pd.DataFrame
    registry: pd.DataFrame


@dataclass(frozen=True)
class SliceAssignmentBundle:
    """Canonical slice assignments and their fold-fitted audit tables."""

    assignments: pd.DataFrame
    file_size_boundaries: pd.DataFrame
    article_type_mappings: pd.DataFrame
    article_type_fold_audit: pd.DataFrame
    slice_support: pd.DataFrame
    assignment_sha256: str


@dataclass(frozen=True)
class SliceAnalysisTables:
    """All deterministic tables produced from candidate OOF packs."""

    slice_metrics: pd.DataFrame
    candidate_slice_deltas: pd.DataFrame
    slice_contrasts: pd.DataFrame
    spring_metrics: pd.DataFrame
    spring_destinations: pd.DataFrame
    error_confusions: pd.DataFrame
    error_examples: pd.DataFrame


def load_slice_analysis_spec(
    path: str | Path = SLICE_ANALYSIS_CONFIG_PATH,
) -> SliceAnalysisSpec:
    """Load and strictly validate the frozen G6 slice contract."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("slice analysis requires schema_version 1.0.0")
    if payload.get("analysis_id") != "g6-shortcut-error-slices":
        raise ValueError("slice analysis identity is not frozen G6")
    if payload.get("stage") != "g6_shortcut_error_analysis":
        raise ValueError("slice analysis stage is invalid")
    if payload.get("target") != "season":
        raise ValueError("slice analysis target must be season")
    if tuple(payload.get("labels", ())) != tuple(SEASON_LABELS):
        raise ValueError("slice analysis changed the canonical Season order")
    expected_row_count = int(payload.get("expected_row_count", -1))
    if expected_row_count != 32_753:
        raise ValueError("slice analysis expected row count must remain 32,753")

    candidates = tuple(
        CandidateExperiment(
            candidate=str(row.get("candidate", "")),
            experiment_id=str(row.get("experiment_id", "")),
            seed=int(row.get("seed", -1)),
        )
        for row in payload.get("candidate_experiments", ())
    )
    observed_candidates = tuple((row.candidate, row.experiment_id, row.seed) for row in candidates)
    if observed_candidates != EXPECTED_CANDIDATE_EXPERIMENTS:
        raise ValueError("slice analysis changed the frozen candidate/seed order")

    protocols = payload.get("slice_protocols", {})
    if set(protocols) != set(SLICE_GROUPS):
        raise ValueError("slice analysis changed the declared slice families")
    for family, groups in SLICE_GROUPS.items():
        protocol = protocols.get(family, {})
        if tuple(protocol.get("groups", ())) != groups:
            raise ValueError(f"slice analysis changed groups for {family}")
        if protocol.get("assignment_scope") != SLICE_ASSIGNMENT_SCOPES[family]:
            raise ValueError(f"slice analysis changed assignment scope for {family}")
    quantiles = tuple(float(value) for value in protocols["file_size_quartile"]["quantiles"])
    if quantiles != (0.25, 0.5, 0.75):
        raise ValueError("file-size quantiles must remain 0.25, 0.5, and 0.75")

    error_analysis = payload.get("error_analysis", {})
    high_confidence_threshold = float(error_analysis.get("high_confidence_threshold", np.nan))
    maximum_ranked_confusions = int(error_analysis.get("maximum_ranked_confusions", -1))
    if not 0.5 <= high_confidence_threshold < 1.0:
        raise ValueError("high-confidence threshold must be in [0.5, 1.0)")
    if maximum_ranked_confusions < 1:
        raise ValueError("maximum ranked confusions must be positive")

    warnings = payload.get("warnings", {})
    required_warnings = {
        "confidence_is_uncalibrated",
        "metadata_is_never_an_inference_feature",
        "slice_results_do_not_freeze_the_ultimate_winner",
    }
    if not all(warnings.get(name) is True for name in required_warnings):
        raise ValueError("slice analysis safety warnings must remain enabled")
    low_support_threshold = int(warnings.get("low_support_threshold", -1))
    if low_support_threshold < 1:
        raise ValueError("low-support threshold must be positive")

    required_metrics = {
        "support",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "per_class_recall",
        "mean_confidence",
        "mean_true_probability",
        "nll",
        "brier",
        "ece",
    }
    if set(payload.get("metrics", ())) != required_metrics:
        raise ValueError("slice analysis changed the frozen metric set")
    return SliceAnalysisSpec(
        analysis_id=str(payload["analysis_id"]),
        expected_row_count=expected_row_count,
        candidates=candidates,
        high_confidence_threshold=high_confidence_threshold,
        maximum_ranked_confusions=maximum_ranked_confusions,
        low_support_threshold=low_support_threshold,
        quantiles=quantiles,
    )


def fit_file_size_boundaries(
    training_frame: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> tuple[float, float, float]:
    """Fit strictly ordered file-size quartile boundaries on training rows only."""
    if "file_size_bytes" not in training_frame:
        raise ValueError("training frame is missing file_size_bytes")
    requested = tuple(float(value) for value in quantiles)
    if requested != (0.25, 0.5, 0.75):
        raise ValueError("file-size analysis requires the frozen quartiles")
    values = pd.to_numeric(training_frame["file_size_bytes"], errors="raise")
    if values.empty or values.isna().any() or values.le(0).any():
        raise ValueError("training file sizes must be positive and complete")
    boundaries = tuple(
        float(value) for value in values.quantile(list(requested), interpolation="linear").tolist()
    )
    if len(boundaries) != 3 or not (boundaries[0] < boundaries[1] < boundaries[2]):
        raise ValueError("training file-size quartiles are not strictly ordered")
    return boundaries


def assign_file_size_quartiles(
    values: pd.Series,
    boundaries: Sequence[float],
) -> pd.Series:
    """Apply training-fitted boundaries to validation file sizes."""
    cut_points = np.asarray(tuple(float(value) for value in boundaries), dtype=float)
    if cut_points.shape != (3,) or not np.all(np.diff(cut_points) > 0):
        raise ValueError("file-size boundaries must contain three increasing values")
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or numeric.le(0).any():
        raise ValueError("validation file sizes must be positive and complete")
    labels = np.asarray(SLICE_GROUPS["file_size_quartile"], dtype=object)
    assigned = labels[np.searchsorted(cut_points, numeric.to_numpy(), side="right")]
    return pd.Series(assigned, index=values.index, dtype="string")


def build_slice_assignments(
    splits: pd.DataFrame,
    spec: SliceAnalysisSpec,
) -> SliceAssignmentBundle:
    """Build one leakage-safe slice assignment for every valid development row."""
    required = {
        "id",
        "season",
        "has_season_label",
        "partition",
        "cv_fold",
        "year",
        "file_size_bytes",
        "product_family_group",
        "mode",
    }
    missing = sorted(required - set(splits.columns))
    if missing:
        raise ValueError(f"splits are missing slice columns: {missing}")
    development = get_samples(splits, partition="development")
    expected = get_samples(development, target="season").reset_index(drop=True)
    if len(expected) != spec.expected_row_count:
        raise ValueError("slice analysis development row count changed")
    if expected["id"].duplicated().any():
        raise ValueError("slice analysis contains duplicate development IDs")
    folds = pd.to_numeric(expected["cv_fold"], errors="raise").astype(int)
    if set(folds) != set(range(5)):
        raise ValueError("slice analysis requires all five canonical folds")

    mappings, shortcut_assignments, shortcut_fold_audit = build_article_type_shortcut_audit(splits)
    assignments = expected.loc[
        :,
        [
            "id",
            "cv_fold",
            "season",
            "year",
            "file_size_bytes",
            "product_family_group",
            "mode",
        ],
    ].copy()
    assignments = assignments.rename(columns={"cv_fold": "fold"})
    assignments["id"] = pd.to_numeric(assignments["id"], errors="raise").astype(int)
    assignments["fold"] = pd.to_numeric(assignments["fold"], errors="raise").astype(int)
    assignments = assignments.merge(
        shortcut_assignments.loc[:, ["id", "fold", "shortcut_slice"]].rename(
            columns={"fold": "shortcut_fold"}
        ),
        on="id",
        how="left",
        validate="one_to_one",
    )
    if assignments["shortcut_slice"].isna().any():
        raise ValueError("ArticleType shortcut assignments are incomplete")
    if not assignments["fold"].eq(assignments["shortcut_fold"]).all():
        raise ValueError("ArticleType shortcut assignments changed validation folds")
    assignments["article_type_shortcut"] = assignments.pop("shortcut_slice")
    assignments = assignments.drop(columns="shortcut_fold")

    boundary_rows: list[dict[str, Any]] = []
    file_size_parts: list[pd.DataFrame] = []
    for fold in range(5):
        training, validation = get_cv_split(splits, fold)
        training = get_samples(training, target="season").reset_index(drop=True)
        validation = get_samples(validation, target="season").reset_index(drop=True)
        boundaries = fit_file_size_boundaries(training, quantiles=spec.quantiles)
        training_ids = sorted(int(value) for value in training["id"])
        boundary_rows.append(
            {
                "fold": fold,
                "training_products": len(training),
                "training_id_sha256": canonical_sha256(training_ids),
                "q25_bytes": boundaries[0],
                "q50_bytes": boundaries[1],
                "q75_bytes": boundaries[2],
            }
        )
        file_size_parts.append(
            pd.DataFrame(
                {
                    "id": pd.to_numeric(validation["id"], errors="raise").astype(int),
                    "file_size_fold": fold,
                    "file_size_quartile": assign_file_size_quartiles(
                        validation["file_size_bytes"],
                        boundaries,
                    ),
                }
            )
        )
    file_size_assignments = pd.concat(file_size_parts, ignore_index=True)
    assignments = assignments.merge(
        file_size_assignments,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if assignments["file_size_quartile"].isna().any():
        raise ValueError("file-size slice assignments are incomplete")
    if not assignments["fold"].eq(assignments["file_size_fold"]).all():
        raise ValueError("file-size assignments changed validation folds")
    assignments = assignments.drop(columns="file_size_fold")

    years = pd.to_numeric(assignments["year"], errors="coerce")
    assignments["acquisition_year"] = np.select(
        [years.isna(), years.isin([2011, 2012])],
        ["missing_year", "dominant_2011_2012"],
        default="other_years",
    )
    family_sizes = expected.groupby("product_family_group", observed=True).size()
    assignments["family_size"] = assignments["product_family_group"].map(family_sizes).astype(int)
    assignments["product_family_size"] = np.where(
        assignments["family_size"].eq(1),
        "singleton",
        "multirow",
    )
    modes = assignments["mode"].astype("string").str.upper()
    assignments["image_mode"] = np.select(
        [modes.eq("RGB"), modes.eq("L")],
        ["rgb", "greyscale"],
        default="other_mode",
    )

    assignment_columns = list(SLICE_GROUPS)
    for family, groups in SLICE_GROUPS.items():
        unknown = set(assignments[family].astype(str)) - set(groups)
        if unknown:
            raise ValueError(f"slice assignment contains unknown {family}: {sorted(unknown)}")
    long_assignments = assignments.melt(
        id_vars=["id", "fold"],
        value_vars=assignment_columns,
        var_name="slice_family",
        value_name="slice_name",
    )
    pooled_support = (
        long_assignments.groupby(["slice_family", "slice_name"], observed=True)
        .size()
        .rename("support")
        .reset_index()
    )
    pooled_support.insert(0, "fold", -1)
    fold_support = (
        long_assignments.groupby(
            ["fold", "slice_family", "slice_name"],
            observed=True,
        )
        .size()
        .rename("support")
        .reset_index()
    )
    slice_support = pd.concat([pooled_support, fold_support], ignore_index=True).sort_values(
        ["slice_family", "slice_name", "fold"],
        kind="stable",
    )
    hash_columns = ["id", "fold", "season", *assignment_columns]
    hash_records = (
        assignments.loc[:, hash_columns].sort_values("id", kind="stable").to_dict(orient="records")
    )
    return SliceAssignmentBundle(
        assignments=assignments.sort_values("id", kind="stable").reset_index(drop=True),
        file_size_boundaries=pd.DataFrame(boundary_rows),
        article_type_mappings=mappings,
        article_type_fold_audit=shortcut_fold_audit,
        slice_support=slice_support.reset_index(drop=True),
        assignment_sha256=canonical_sha256(hash_records),
    )


def _enrich_oof(pack: CandidateOOFPack, assignments: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_id",
        "experiment_id",
        "id",
        "fold",
        "seed",
        "y_true",
        "y_pred",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(pack.oof.columns))
    if missing:
        raise ValueError(f"{pack.experiment_id} OOF is missing columns: {missing}")
    assignment_view = assignments.rename(
        columns={"fold": "assignment_fold", "season": "assignment_season"}
    )
    merged = pack.oof.merge(
        assignment_view,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if merged["assignment_season"].isna().any():
        raise ValueError(f"{pack.experiment_id} OOF has IDs without slice assignments")
    if (
        not pd.to_numeric(merged["fold"], errors="raise")
        .astype(int)
        .eq(pd.to_numeric(merged["assignment_fold"], errors="raise").astype(int))
        .all()
    ):
        raise ValueError(f"{pack.experiment_id} OOF fold differs from slice assignment")
    if not merged["y_true"].astype(str).eq(merged["assignment_season"].astype(str)).all():
        raise ValueError(f"{pack.experiment_id} OOF truth differs from slice assignment")
    probabilities = merged.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{pack.experiment_id} OOF probabilities do not sum to one")
    label_to_index = {label: index for index, label in enumerate(SEASON_LABELS)}
    true_indices = np.fromiter(
        (label_to_index[str(label)] for label in merged["y_true"]),
        dtype=np.int64,
        count=len(merged),
    )
    merged["confidence"] = probabilities.max(axis=1)
    merged["true_probability"] = probabilities[
        np.arange(len(merged)),
        true_indices,
    ]
    merged["correct"] = merged["y_true"].astype(str).eq(merged["y_pred"].astype(str))
    return merged


def _slice_metric_row(
    frame: pd.DataFrame,
    *,
    pack: CandidateOOFPack,
    slice_family: str,
    slice_name: str,
    low_support_threshold: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate": pack.candidate,
        "experiment_id": pack.experiment_id,
        "seed": pack.seed,
        "slice_family": slice_family,
        "slice_name": slice_name,
        "support": len(frame),
        "low_support": len(frame) < low_support_threshold,
    }
    if frame.empty:
        return {
            **base,
            "labels_present": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "mean_confidence": np.nan,
            "mean_true_probability": np.nan,
            "nll": np.nan,
            "brier": np.nan,
            "ece": np.nan,
            **{f"recall_{label}": np.nan for label in SEASON_LABELS},
        }
    metrics = multiclass_metrics(
        frame["y_true"].astype(str),
        probabilities=frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float),
        labels=SEASON_LABELS,
        y_pred=frame["y_pred"].astype(str),
    )
    return {
        **base,
        "labels_present": int(frame["y_true"].nunique()),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_confidence": float(frame["confidence"].mean()),
        "mean_true_probability": float(frame["true_probability"].mean()),
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        **{f"recall_{label}": metrics["per_class"][label]["recall"] for label in SEASON_LABELS},
    }


def _build_candidate_slice_deltas(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, family, name), group in slice_metrics.groupby(
        ["seed", "slice_family", "slice_name"],
        observed=True,
        sort=True,
    ):
        indexed = group.set_index("candidate")
        if set(indexed.index) != {"C2", "I2"}:
            raise ValueError("candidate slice comparison requires exactly C2 and I2")
        row: dict[str, Any] = {
            "seed": int(seed),
            "slice_family": family,
            "slice_name": name,
            "c2_support": int(indexed.loc["C2", "support"]),
            "i2_support": int(indexed.loc["I2", "support"]),
        }
        for metric in (
            "accuracy",
            "macro_f1",
            "mean_confidence",
            *tuple(f"recall_{label}" for label in SEASON_LABELS),
        ):
            row[f"c2_{metric}"] = float(indexed.loc["C2", metric])
            row[f"i2_{metric}"] = float(indexed.loc["I2", metric])
            row[f"i2_minus_c2_{metric}"] = row[f"i2_{metric}"] - row[f"c2_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["seed", "slice_family", "slice_name"],
        kind="stable",
    )


def _build_slice_contrasts(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in ("C2", "I2"):
        for seed in (2753, 2026):
            subset = slice_metrics.loc[
                slice_metrics["candidate"].eq(candidate) & slice_metrics["seed"].eq(seed)
            ]
            indexed = subset.set_index(["slice_family", "slice_name"])
            for family, reference, comparison, contrast in SLICE_CONTRASTS:
                left = indexed.loc[(family, reference)]
                right = indexed.loc[(family, comparison)]
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "seed": seed,
                    "slice_family": family,
                    "contrast": contrast,
                    "reference_slice": reference,
                    "comparison_slice": comparison,
                    "reference_support": int(left["support"]),
                    "comparison_support": int(right["support"]),
                }
                for metric in ("accuracy", "macro_f1", "mean_confidence"):
                    row[f"reference_{metric}"] = float(left[metric])
                    row[f"comparison_{metric}"] = float(right[metric])
                    row[f"comparison_minus_reference_{metric}"] = (
                        row[f"comparison_{metric}"] - row[f"reference_{metric}"]
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["candidate", "seed", "slice_family"],
        kind="stable",
    )


def _spring_tables(
    enriched: pd.DataFrame,
    pack: CandidateOOFPack,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = multiclass_metrics(
        enriched["y_true"].astype(str),
        probabilities=enriched.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float),
        labels=SEASON_LABELS,
        y_pred=enriched["y_pred"].astype(str),
    )
    spring_class = metrics["per_class"]["Spring"]
    spring = enriched.loc[enriched["y_true"].eq("Spring")].copy()
    errors = spring.loc[~spring["correct"]]
    error_counts = errors["y_pred"].value_counts()
    main_destination = str(error_counts.index[0]) if len(error_counts) else "none"
    main_count = int(error_counts.iloc[0]) if len(error_counts) else 0
    spring_row = {
        "candidate": pack.candidate,
        "experiment_id": pack.experiment_id,
        "seed": pack.seed,
        "support": int(spring_class["support"]),
        "precision": spring_class["precision"],
        "recall": spring_class["recall"],
        "f1": spring_class["f1"],
        "correct_count": int(spring["correct"].sum()),
        "error_count": int((~spring["correct"]).sum()),
        "mean_confidence": float(spring["confidence"].mean()),
        "mean_true_probability": float(spring["true_probability"].mean()),
        "correct_mean_confidence": float(spring.loc[spring["correct"], "confidence"].mean()),
        "error_mean_confidence": float(errors["confidence"].mean()),
        "main_error_destination": main_destination,
        "main_error_count": main_count,
        "main_error_rate_of_spring": main_count / len(spring),
    }
    destinations = []
    for label in SEASON_LABELS:
        destination = spring.loc[spring["y_pred"].eq(label)]
        destinations.append(
            {
                "candidate": pack.candidate,
                "experiment_id": pack.experiment_id,
                "seed": pack.seed,
                "predicted_label": label,
                "count": len(destination),
                "proportion_of_true_spring": len(destination) / len(spring),
                "mean_confidence": (
                    float(destination["confidence"].mean()) if len(destination) else np.nan
                ),
            }
        )
    return spring_row, destinations


def _error_tables(
    enriched: pd.DataFrame,
    pack: CandidateOOFPack,
    spec: SliceAnalysisSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = enriched.loc[~enriched["correct"]].copy()
    errors["high_confidence_error"] = errors["confidence"].ge(spec.high_confidence_threshold)
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    grouped = (
        errors.groupby(["y_true", "y_pred"], observed=True)
        .agg(
            count=("id", "size"),
            mean_confidence=("confidence", "mean"),
            high_confidence_count=("high_confidence_error", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["count", "y_true", "y_pred"],
        ascending=[False, True, True],
        kind="stable",
    ).head(spec.maximum_ranked_confusions)
    true_support = enriched["y_true"].value_counts()
    true_errors = errors["y_true"].value_counts()
    for rank, confusion in enumerate(grouped.itertuples(index=False), start=1):
        pair = errors.loc[
            errors["y_true"].eq(confusion.y_true) & errors["y_pred"].eq(confusion.y_pred)
        ].sort_values(["confidence", "id"], ascending=[False, True], kind="stable")
        rows.append(
            {
                "candidate": pack.candidate,
                "experiment_id": pack.experiment_id,
                "seed": pack.seed,
                "rank": rank,
                "true_label": confusion.y_true,
                "predicted_label": confusion.y_pred,
                "count": int(confusion.count),
                "share_of_all_errors": int(confusion.count) / len(errors),
                "share_of_true_label": int(confusion.count) / int(true_support[confusion.y_true]),
                "share_of_true_label_errors": int(confusion.count)
                / int(true_errors[confusion.y_true]),
                "mean_confidence": float(confusion.mean_confidence),
                "high_confidence_count": int(confusion.high_confidence_count),
                "high_confidence_rate": int(confusion.high_confidence_count) / int(confusion.count),
            }
        )
        example = pair.iloc[0]
        examples.append(
            {
                "candidate": pack.candidate,
                "experiment_id": pack.experiment_id,
                "seed": pack.seed,
                "confusion_rank": rank,
                "id": int(example["id"]),
                "fold": int(example["fold"]),
                "true_label": str(example["y_true"]),
                "predicted_label": str(example["y_pred"]),
                "confidence": float(example["confidence"]),
                "true_probability": float(example["true_probability"]),
                "article_type_shortcut": str(example["article_type_shortcut"]),
                "acquisition_year": str(example["acquisition_year"]),
                "file_size_quartile": str(example["file_size_quartile"]),
                "product_family_size": str(example["product_family_size"]),
                "image_mode": str(example["image_mode"]),
                "run_id": str(example["run_id"]),
            }
        )
    return rows, examples


def analyse_slice_packs(
    packs: Sequence[CandidateOOFPack],
    bundle: SliceAssignmentBundle,
    spec: SliceAnalysisSpec,
) -> SliceAnalysisTables:
    """Calculate all frozen shortcut, minority, and error tables."""
    observed = tuple((pack.candidate, pack.experiment_id, pack.seed) for pack in packs)
    expected = tuple(
        (candidate.candidate, candidate.experiment_id, candidate.seed)
        for candidate in spec.candidates
    )
    if observed != expected:
        raise ValueError("slice OOF packs changed the frozen candidate/seed order")
    metric_rows: list[dict[str, Any]] = []
    spring_rows: list[dict[str, Any]] = []
    destination_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for pack in packs:
        enriched = _enrich_oof(pack, bundle.assignments)
        for family, groups in SLICE_GROUPS.items():
            for name in groups:
                metric_rows.append(
                    _slice_metric_row(
                        enriched.loc[enriched[family].eq(name)],
                        pack=pack,
                        slice_family=family,
                        slice_name=name,
                        low_support_threshold=spec.low_support_threshold,
                    )
                )
        spring_row, destinations = _spring_tables(enriched, pack)
        spring_rows.append(spring_row)
        destination_rows.extend(destinations)
        confusions, examples = _error_tables(enriched, pack, spec)
        confusion_rows.extend(confusions)
        example_rows.extend(examples)
    slice_metrics = pd.DataFrame(metric_rows).sort_values(
        ["candidate", "seed", "slice_family", "slice_name"],
        kind="stable",
    )
    return SliceAnalysisTables(
        slice_metrics=slice_metrics.reset_index(drop=True),
        candidate_slice_deltas=_build_candidate_slice_deltas(slice_metrics).reset_index(drop=True),
        slice_contrasts=_build_slice_contrasts(slice_metrics).reset_index(drop=True),
        spring_metrics=pd.DataFrame(spring_rows)
        .sort_values(["candidate", "seed"], kind="stable")
        .reset_index(drop=True),
        spring_destinations=pd.DataFrame(destination_rows)
        .sort_values(["candidate", "seed", "predicted_label"], kind="stable")
        .reset_index(drop=True),
        error_confusions=pd.DataFrame(confusion_rows)
        .sort_values(["candidate", "seed", "rank"], kind="stable")
        .reset_index(drop=True),
        error_examples=pd.DataFrame(example_rows)
        .sort_values(["candidate", "seed", "confusion_rank"], kind="stable")
        .reset_index(drop=True),
    )


def plot_slice_macro_f1(
    slice_metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot macro-F1 for every non-empty declared slice across models and seeds."""
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    styles = {
        ("C2", 2753): ("#82A6EA", "o", "-"),
        ("C2", 2026): ("#2F66E8", "o", "--"),
        ("I2", 2753): ("#83C99B", "s", "-"),
        ("I2", 2026): ("#18A34A", "s", "--"),
    }
    for axis, family in zip(axes.flat, SLICE_GROUPS, strict=False):
        names = list(SLICE_GROUPS[family])
        positions = np.arange(len(names))
        for (candidate, seed), (color, marker, line_style) in styles.items():
            subset = slice_metrics.loc[
                slice_metrics["candidate"].eq(candidate)
                & slice_metrics["seed"].eq(seed)
                & slice_metrics["slice_family"].eq(family)
            ].set_index("slice_name")
            values = [float(subset.loc[name, "macro_f1"]) for name in names]
            axis.plot(
                positions,
                values,
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=2,
                label=f"{candidate} seed {seed}",
            )
        axis.set_xticks(positions, [name.replace("_", "\n") for name in names])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Fixed-label macro-F1")
        axis.set_title(family.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, fontsize=9)
    figure.suptitle(
        "Frozen C2/I2 OOF performance across declared shortcut slices",
        fontsize=16,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    path = Path(output_path)
    atomic_write_bytes(path, buffer.getvalue())
    return path


def plot_spring_destinations(
    spring_destinations: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot where true Spring products are predicted for each candidate and seed."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    colors = {
        "Fall": "#D97706",
        "Spring": "#16A34A",
        "Summer": "#2563EB",
        "Winter": "#7C3AED",
    }
    for axis, seed in zip(axes, (2753, 2026), strict=True):
        seed_rows = spring_destinations.loc[spring_destinations["seed"].eq(seed)]
        bottom = np.zeros(2, dtype=float)
        for label in SEASON_LABELS:
            values = []
            for candidate in ("C2", "I2"):
                row = seed_rows.loc[
                    seed_rows["candidate"].eq(candidate) & seed_rows["predicted_label"].eq(label)
                ]
                if len(row) != 1:
                    raise ValueError("Spring destination table is incomplete")
                values.append(float(row.iloc[0]["proportion_of_true_spring"]))
            axis.bar(
                [0, 1],
                values,
                bottom=bottom,
                color=colors[label],
                label=label,
            )
            bottom += np.asarray(values)
        axis.set_xticks([0, 1], ["C2", "I2"])
        axis.set_ylim(0, 1)
        axis.set_title(f"Seed {seed}")
        axis.set_ylabel("Share of true Spring products")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, fontsize=9)
    figure.suptitle(
        "True Spring prediction destinations",
        fontsize=15,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    path = Path(output_path)
    atomic_write_bytes(path, buffer.getvalue())
    return path


__all__ = [
    "CandidateExperiment",
    "CandidateOOFPack",
    "EXPECTED_CANDIDATE_EXPERIMENTS",
    "SLICE_ANALYSIS_CONFIG_PATH",
    "SLICE_GROUPS",
    "SliceAnalysisSpec",
    "SliceAnalysisTables",
    "SliceAssignmentBundle",
    "analyse_slice_packs",
    "assign_file_size_quartiles",
    "build_slice_assignments",
    "fit_file_size_boundaries",
    "load_slice_analysis_spec",
    "plot_slice_macro_f1",
    "plot_spring_destinations",
]
