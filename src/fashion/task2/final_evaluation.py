"""Frozen Task 2 Season evaluation contracts and evidence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fashion.config import ROOT, TASK2_CONFIG_DIR
from fashion.task2.baselines import MajorityBaselineModel, fit_training_fold_majority
from fashion.task2.robustness import RobustnessCondition
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics, paired_group_bootstrap

FINAL_EVALUATION_CONFIG_PATH = TASK2_CONFIG_DIR / "g9_final_evaluation.json"
PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in SEASON_LABELS)


@dataclass(frozen=True)
class FinalEvaluationSpec:
    """Strict identity and protocol for the one-shot Season evaluation."""

    evaluation_id: str
    labels: tuple[str, ...]
    freeze_id: str
    run_id: str
    bundle_path: Path
    bundle_sha256: str
    temperature: float
    expected_holdout_rows: int
    expected_quarantine_rows: int
    expected_test_rows: int
    bootstrap_replicates: int
    bootstrap_seed: int
    conditions: tuple[RobustnessCondition, ...]
    official_output_path: Path
    config_path: Path
    config_sha256: str


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _parse_robustness_condition(payload: Mapping[str, Any]) -> RobustnessCondition:
    name = str(payload.get("condition", ""))
    kind = str(payload.get("kind", ""))
    keys_by_kind = {
        "none": {"condition", "kind"},
        "jpeg_reencode": {"condition", "kind", "quality", "subsampling"},
        "brightness": {"condition", "kind", "factor"},
        "gaussian_blur": {"condition", "kind", "radius"},
    }
    if kind not in keys_by_kind:
        raise ValueError(f"unknown final-evaluation robustness kind: {kind}")
    _require_exact_keys(payload, keys_by_kind[kind], f"robustness condition {name}")
    return RobustnessCondition(
        condition=name,
        kind=kind,
        quality=int(payload["quality"]) if "quality" in payload else None,
        subsampling=int(payload["subsampling"]) if "subsampling" in payload else None,
        factor=float(payload["factor"]) if "factor" in payload else None,
        radius=float(payload["radius"]) if "radius" in payload else None,
    )


def load_final_evaluation_spec(
    path: str | Path = FINAL_EVALUATION_CONFIG_PATH,
    *,
    project_root: str | Path = ROOT,
) -> FinalEvaluationSpec:
    """Load the pre-holdout contract and fail closed if any fixed choice changed."""
    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "evaluation_id",
            "target",
            "labels",
            "frozen_selection",
            "internal_holdout",
            "baseline",
            "metrics",
            "bootstrap",
            "slices",
            "robustness",
            "official_test",
            "safety",
        },
        "final evaluation config",
    )
    if payload["schema_version"] != "1.0.0":
        raise ValueError("final evaluation requires schema_version 1.0.0")
    if payload["evaluation_id"] != "g9-task2-season-final-evaluation":
        raise ValueError("final evaluation identity changed")
    if payload["target"] != "season" or tuple(payload["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("final evaluation target or canonical label order changed")

    frozen = payload["frozen_selection"]
    _require_exact_keys(
        frozen,
        {
            "freeze_id",
            "candidate",
            "experiment_id",
            "run_id",
            "bundle_path",
            "bundle_sha256",
            "temperature",
        },
        "frozen selection",
    )
    expected_frozen = {
        "freeze_id": "task2-season-i2-development-v1",
        "candidate": "I2",
        "experiment_id": "g4-i2-article-type-lambda-0-3-c1",
        "run_id": "task2-season-i2-refit-fall-s2753-637dd6378be9",
        "bundle_path": "models/task2_season.pt",
        "bundle_sha256": (
            "5927eff73130acedc8015199e1df5a6c6edf64c0b45023ebd91c48d7ed40f93c"
        ),
        "temperature": 1.3650015953177774,
    }
    if frozen != expected_frozen:
        raise ValueError("final evaluation changed the frozen I2 identity")

    holdout = payload["internal_holdout"]
    expected_holdout = {
        "partition": "holdout",
        "expected_rows": 5778,
        "quarantine_partition": "quarantine",
        "expected_quarantine_rows": 61,
        "quarantine_is_excluded": True,
        "unlock_is_one_shot": True,
    }
    if holdout != expected_holdout:
        raise ValueError("final evaluation changed the protected split boundary")

    baseline = payload["baseline"]
    if baseline != {
        "id": "B0",
        "rule": "most_frequent_season_in_development_only",
        "primary_holdout_comparator": True,
        "b1_holdout_status": "development_only_no_prefrozen_holdout_predictions",
    }:
        raise ValueError("final evaluation changed the baseline boundary")

    bootstrap = payload["bootstrap"]
    if bootstrap != {
        "unit": "product_family_group",
        "replicates": 10000,
        "random_seed": 2753,
        "interval": "percentile_95",
        "comparisons": ["I2_minus_B0"],
    }:
        raise ValueError("final evaluation changed the grouped bootstrap protocol")

    condition_payloads = payload["robustness"]
    if not isinstance(condition_payloads, list):
        raise ValueError("final evaluation robustness conditions must be a list")
    conditions = tuple(_parse_robustness_condition(row) for row in condition_payloads)
    if tuple(condition.condition for condition in conditions) != (
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    ):
        raise ValueError("final evaluation changed the robustness condition order")

    official = payload["official_test"]
    if official != {
        "expected_rows": 5829,
        "input_manifest": "data/processed/prediction_manifest.csv",
        "output_path": "results/season_test_predictions.csv",
        "output_columns": ["id", "season"],
    }:
        raise ValueError("final evaluation changed the teacher-test output contract")

    safety = payload["safety"]
    required_safety = {
        "retraining_after_unlock_allowed": False,
        "retuning_after_unlock_allowed": False,
        "winner_change_after_unlock_allowed": False,
        "temperature_refit_after_unlock_allowed": False,
        "official_test_labels_are_unavailable": True,
        "raw_protected_labels_are_not_written_to_prediction_artifacts": True,
    }
    if safety != required_safety:
        raise ValueError("final evaluation safety boundary changed")

    bundle_relative = Path(str(frozen["bundle_path"]))
    output_relative = Path(str(official["output_path"]))
    if bundle_relative.is_absolute() or output_relative.is_absolute():
        raise ValueError("final evaluation artifact paths must be project-relative")
    return FinalEvaluationSpec(
        evaluation_id=str(payload["evaluation_id"]),
        labels=tuple(str(value) for value in payload["labels"]),
        freeze_id=str(frozen["freeze_id"]),
        run_id=str(frozen["run_id"]),
        bundle_path=(root / bundle_relative).resolve(),
        bundle_sha256=str(frozen["bundle_sha256"]),
        temperature=float(frozen["temperature"]),
        expected_holdout_rows=int(holdout["expected_rows"]),
        expected_quarantine_rows=int(holdout["expected_quarantine_rows"]),
        expected_test_rows=int(official["expected_rows"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["random_seed"]),
        conditions=conditions,
        official_output_path=(root / output_relative).resolve(),
        config_path=config_path,
        config_sha256=canonical_sha256(payload),
    )


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: Sequence[int],
    labels: Sequence[str] = SEASON_LABELS,
) -> dict[str, Any]:
    """Require exact ID coverage and full-precision fixed-label probabilities."""
    ordered_labels = tuple(str(value) for value in labels)
    probability_columns = tuple(f"prob_{label}" for label in ordered_labels)
    required = {"id", "y_pred", *probability_columns}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    relevant = frame.loc[:, ["id", "y_pred", *probability_columns]].copy()
    if relevant.isna().any().any():
        raise ValueError("prediction frame contains missing values")
    ids = pd.to_numeric(relevant["id"], errors="raise").astype(int)
    expected = [int(value) for value in expected_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected prediction IDs must be unique")
    if ids.duplicated().any() or len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("every expected prediction ID must appear exactly once")
    unknown = sorted(set(relevant["y_pred"].astype(str)) - set(ordered_labels))
    if unknown:
        raise ValueError(f"prediction frame contains unknown labels: {unknown}")
    probabilities = relevant.loc[:, probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("prediction probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("prediction probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("prediction probability rows must sum to one")
    argmax = np.asarray(ordered_labels, dtype=object)[probabilities.argmax(axis=1)]
    if not np.array_equal(argmax.astype(str), relevant["y_pred"].astype(str).to_numpy()):
        raise ValueError("recorded labels differ from probability argmax")
    return {
        "row_count": len(frame),
        "unique_id_count": int(ids.nunique()),
        "id_set_sha256": canonical_sha256(sorted(ids.tolist())),
        "labels": list(ordered_labels),
    }


def build_b0_prediction_frame(
    development_frame: pd.DataFrame,
    *,
    expected_ids: Sequence[int],
    labels: tuple[str, ...] = SEASON_LABELS,
) -> tuple[pd.DataFrame, MajorityBaselineModel]:
    """Fit B0 on development only and predict an ordered label-free ID list."""
    model = fit_training_fold_majority(development_frame, labels=labels, target="season")
    identifiers = [int(value) for value in expected_ids]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("B0 prediction IDs must be unique")
    probabilities = model.predict_proba(len(identifiers))
    frame = pd.DataFrame(
        {
            "id": identifiers,
            "y_pred": model.predict(len(identifiers)),
        }
    )
    for index, label in enumerate(model.labels):
        frame[f"prob_{label}"] = probabilities[:, index]
    validate_prediction_frame(frame, expected_ids=identifiers, labels=model.labels)
    return frame, model


def score_prediction_frame(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    labels: Sequence[str] = SEASON_LABELS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Join protected truth after unlock and compute fixed-label metrics."""
    if set(("id", "season")) - set(truth):
        raise ValueError("truth frame must contain id and season")
    if truth["id"].duplicated().any():
        raise ValueError("truth IDs must be unique")
    expected_ids = pd.to_numeric(truth["id"], errors="raise").astype(int).tolist()
    validate_prediction_frame(predictions, expected_ids=expected_ids, labels=labels)
    ordered = predictions.copy()
    ordered["id"] = pd.to_numeric(ordered["id"], errors="raise").astype(int)
    truth_values = truth.loc[:, ["id", "season"]].copy()
    truth_values["id"] = pd.to_numeric(truth_values["id"], errors="raise").astype(int)
    truth_values["season"] = truth_values["season"].astype(str)
    scored = ordered.merge(truth_values, on="id", how="left", validate="one_to_one")
    scored = scored.rename(columns={"season": "y_true"})
    allowed = set(str(value) for value in labels)
    unknown = sorted(set(scored["y_true"]) - allowed)
    if unknown:
        raise ValueError(f"truth frame contains unknown Season labels: {unknown}")
    probabilities = scored.loc[:, [f"prob_{label}" for label in labels]].to_numpy(
        dtype=np.float64
    )
    metrics = multiclass_metrics(
        scored["y_true"].astype(str).to_numpy(),
        probabilities=probabilities,
        labels=labels,
        y_pred=scored["y_pred"].astype(str).to_numpy(),
        ece_bins=15,
    )
    return metrics, scored


def fit_development_slice_reference(
    development: pd.DataFrame,
    *,
    labels: Sequence[str] = SEASON_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit every holdout slice reference without reading any protected targets."""
    required = {"id", "articleType", "season", "file_size_bytes"}
    missing = sorted(required - set(development))
    if missing:
        raise ValueError(f"development slice frame is missing columns: {missing}")
    if development["id"].duplicated().any():
        raise ValueError("development slice reference IDs must be unique")
    label_order = {str(label): index for index, label in enumerate(labels)}
    valid = development.loc[
        development["articleType"].astype(str).str.strip().ne("")
        & development["season"].astype(str).isin(label_order)
    ]
    mapping_rows: list[dict[str, Any]] = []
    for article_type, group in valid.groupby("articleType", sort=True, observed=True):
        counts = group["season"].astype(str).value_counts()
        maximum = int(counts.max())
        winners = [label for label, count in counts.items() if int(count) == maximum]
        majority = min(winners, key=label_order.__getitem__)
        mapping_rows.append(
            {
                "articleType": str(article_type),
                "shortcut_majority_season": majority,
                "majority_count": maximum,
                "training_labeled_count": len(group),
                "majority_share": maximum / len(group),
            }
        )
    mappings = pd.DataFrame(mapping_rows)

    development_sizes = pd.to_numeric(development["file_size_bytes"], errors="raise")
    if development_sizes.isna().any() or development_sizes.le(0).any():
        raise ValueError("development file sizes must be positive")
    boundaries = tuple(
        float(value)
        for value in development_sizes.quantile([0.25, 0.5, 0.75], interpolation="linear")
    )
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError("development file-size boundaries are not strictly increasing")
    boundary_frame = pd.DataFrame(
        [
            {
                "fit_scope": "all_valid_development_rows",
                "training_products": len(development),
                "q25_bytes": boundaries[0],
                "q50_bytes": boundaries[1],
                "q75_bytes": boundaries[2],
            }
        ]
    )
    audit = {
        "mapping_fit_rows": len(development),
        "file_size_fit_rows": len(development),
        "article_type_count": len(mappings),
        "development_id_sha256": canonical_sha256(
            sorted(pd.to_numeric(development["id"], errors="raise").astype(int).tolist())
        ),
    }
    return mappings, boundary_frame, audit


def build_holdout_slice_assignments(
    development_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    *,
    labels: Sequence[str] = SEASON_LABELS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit shortcut and size references on development, then describe holdout rows."""
    required_development = {"id", "articleType", "season", "file_size_bytes"}
    required_holdout = {
        "id",
        "articleType",
        "season",
        "year",
        "file_size_bytes",
        "product_family_group",
        "mode",
    }
    missing_development = sorted(required_development - set(development_frame))
    missing_holdout = sorted(required_holdout - set(holdout_frame))
    if missing_development or missing_holdout:
        raise ValueError(
            "slice inputs are incomplete; "
            f"development={missing_development}, holdout={missing_holdout}"
        )
    development = development_frame.copy()
    holdout = holdout_frame.copy().reset_index(drop=True)
    if development["id"].duplicated().any() or holdout["id"].duplicated().any():
        raise ValueError("slice input IDs must be unique")
    mappings, boundary_frame, reference_audit = fit_development_slice_reference(
        development,
        labels=labels,
    )
    mapping = dict(
        zip(
            mappings["articleType"].astype(str),
            mappings["shortcut_majority_season"].astype(str),
            strict=True,
        )
    )
    article_types = holdout["articleType"].astype(str)
    mapped = article_types.map(mapping)
    holdout_truth = holdout["season"].astype(str)
    holdout["article_type_shortcut"] = np.select(
        [
            article_types.str.strip().eq(""),
            mapped.isna(),
            mapped.eq(holdout_truth),
        ],
        ["missing_article_type", "unseen_article_type", "aligned"],
        default="conflict",
    )

    boundary_row = boundary_frame.iloc[0]
    boundaries = tuple(
        float(boundary_row[column])
        for column in ("q25_bytes", "q50_bytes", "q75_bytes")
    )
    holdout_sizes = pd.to_numeric(holdout["file_size_bytes"], errors="raise")
    quartiles = np.asarray(("q1_smallest", "q2", "q3", "q4_largest"), dtype=object)
    holdout["file_size_quartile"] = quartiles[
        np.searchsorted(np.asarray(boundaries), holdout_sizes.to_numpy(), side="right")
    ]

    years = pd.to_numeric(holdout["year"], errors="coerce")
    holdout["acquisition_year"] = np.select(
        [years.isna(), years.isin([2011, 2012])],
        ["missing_year", "dominant_2011_2012"],
        default="other_years",
    )
    family_sizes = holdout.groupby("product_family_group", observed=True).size()
    holdout["product_family_size"] = np.where(
        holdout["product_family_group"].map(family_sizes).eq(1),
        "singleton",
        "multirow",
    )
    modes = holdout["mode"].astype(str).str.upper()
    holdout["image_mode"] = np.select(
        [modes.eq("RGB"), modes.eq("L")],
        ["rgb", "greyscale"],
        default="other_mode",
    )
    assignments = holdout.loc[
        :,
        [
            "id",
            "article_type_shortcut",
            "acquisition_year",
            "file_size_quartile",
            "product_family_size",
            "image_mode",
        ],
    ].copy()
    audit = {
        **reference_audit,
        "article_type_mapping": mapping,
        "file_size_boundaries": {
            "q25_bytes": boundaries[0],
            "q50_bytes": boundaries[1],
            "q75_bytes": boundaries[2],
        },
    }
    return assignments, audit


def summarise_grouped_bootstrap(
    y_true: Sequence[str] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    b0_predictions: Sequence[str] | np.ndarray,
    i2_predictions: Sequence[str] | np.ndarray,
    *,
    labels: Sequence[str] = SEASON_LABELS,
    replicates: int = 10_000,
    random_seed: int = 2753,
) -> pd.DataFrame:
    """Return deterministic family-blocked percentile intervals for I2 versus B0."""
    draws = paired_group_bootstrap(
        y_true,
        groups,
        {"I2_minus_B0": (b0_predictions, i2_predictions)},
        labels=labels,
        replicates=replicates,
        random_seed=random_seed,
    )
    columns = {
        "model_b_macro_f1": "i2_macro_f1",
        "b_minus_a_macro_f1": "i2_minus_b0_macro_f1",
        "model_b_accuracy": "i2_accuracy",
        "b_minus_a_accuracy": "i2_minus_b0_accuracy",
    }
    for label in labels:
        slug = "_".join(str(label).lower().split())
        columns[f"model_b_f1_{slug}"] = f"i2_{slug}_f1"
        columns[f"b_minus_a_f1_{slug}"] = f"i2_minus_b0_{slug}_f1"
    rows = []
    for column, metric in columns.items():
        values = draws[column].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
                "replicates": int(replicates),
                "random_seed": int(random_seed),
                "sampled_group_count": int(draws["sampled_group_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_official_predictions(
    predictions: pd.DataFrame,
    *,
    expected_ids: Sequence[int],
    labels: Sequence[str] = SEASON_LABELS,
) -> pd.DataFrame:
    """Return the exact two-column teacher-test Season handoff in template order."""
    expected = [int(value) for value in expected_ids]
    validate_prediction_frame(predictions, expected_ids=expected, labels=labels)
    by_id = predictions.copy()
    by_id["id"] = pd.to_numeric(by_id["id"], errors="raise").astype(int)
    by_id = by_id.set_index("id")
    if not by_id.index.is_unique:
        raise ValueError("official Season prediction IDs must be unique")
    output = pd.DataFrame(
        {
            "id": expected,
            "season": [str(by_id.at[identifier, "y_pred"]) for identifier in expected],
        }
    )
    if output.columns.tolist() != ["id", "season"]:
        raise RuntimeError("official Season output schema changed")
    if output["season"].isna().any() or output["season"].astype(str).str.strip().eq("").any():
        raise ValueError("official Season predictions contain blank labels")
    return output


__all__ = [
    "FINAL_EVALUATION_CONFIG_PATH",
    "FinalEvaluationSpec",
    "PROBABILITY_COLUMNS",
    "build_b0_prediction_frame",
    "build_holdout_slice_assignments",
    "build_official_predictions",
    "fit_development_slice_reference",
    "load_final_evaluation_spec",
    "score_prediction_frame",
    "summarise_grouped_bootstrap",
    "validate_prediction_frame",
]
