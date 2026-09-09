"""Leakage-safe calibration and selective-risk analysis for Task 2 finalists."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import TASK2_CONFIG_DIR
from fashion.task2.slices import CandidateOOFPack
from fashion.train.artifacts import atomic_write_bytes, canonical_sha256
from fashion.train.metrics import (
    SEASON_LABELS,
    cross_fit_temperature,
    fit_temperature,
    multiclass_metrics,
)

CALIBRATION_CONFIG_PATH = TASK2_CONFIG_DIR / "g6_cross_fitted_calibration.json"
EXPECTED_CANDIDATES = (
    ("C2", "g3-c2-t0-resnet18", 2753),
    ("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
)
PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in SEASON_LABELS)
CALIBRATION_METHODS = ("uncalibrated", "cross_fitted_temperature")


@dataclass(frozen=True)
class CalibrationCandidate:
    """One primary-seed finalist identity."""

    candidate: str
    experiment_id: str
    seed: int


@dataclass(frozen=True)
class CalibrationSpec:
    """Strictly validated calibration and risk-coverage declaration."""

    analysis_id: str
    expected_row_count: int
    candidates: tuple[CalibrationCandidate, ...]
    folds: tuple[int, ...]
    probability_floor: float
    temperature_bounds: tuple[float, float]
    optimizer_tolerance: float
    ece_bins: int
    coverage_start: float
    coverage_stop: float
    coverage_step: float
    review_budgets: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationTables:
    """All deterministic calibration outputs, including temporary calibrated OOF rows."""

    calibration_summary: pd.DataFrame
    fold_temperatures: pd.DataFrame
    reliability_bins: pd.DataFrame
    risk_coverage: pd.DataFrame
    review_budget_summary: pd.DataFrame
    deployment_temperatures: pd.DataFrame
    calibrated_oof: pd.DataFrame


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _exact_int(value: Any, scope: str) -> int:
    """Parse a JSON integer without silently truncating booleans or fractions."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{scope} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{scope} must be an integer")
    return int(numeric)


def _int64_vector(values: Sequence[Any] | np.ndarray, scope: str) -> np.ndarray:
    """Return exact int64 values without routing large integers through float64."""
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{scope} must be a vector")
    parsed = [_exact_int(value, scope) for value in raw.tolist()]
    int64_info = np.iinfo(np.int64)
    if any(value < int64_info.min or value > int64_info.max for value in parsed):
        raise ValueError(f"{scope} must fit signed int64")
    return np.asarray(parsed, dtype=np.int64)


def _validated_probability_inputs(
    y_true: Sequence[str] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[str],
    *,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate and normalise probability rows before confidence ranking."""
    ordered_labels = tuple(str(label) for label in labels)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError(f"{scope} labels must contain at least two unique values")
    true = np.asarray(y_true, dtype=object)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if true.ndim != 1 or len(true) == 0:
        raise ValueError(f"{scope} targets must be a non-empty vector")
    if matrix.shape != (len(true), len(ordered_labels)):
        raise ValueError(f"{scope} probabilities have an incompatible shape")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{scope} probabilities must all be finite")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError(f"{scope} probabilities must be in [0, 1]")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"each {scope} probability row must sum to one")
    true = true.astype(str)
    unknown = sorted(set(true) - set(ordered_labels))
    if unknown:
        raise ValueError(f"{scope} targets contain unknown labels: {unknown}")
    return true, matrix / row_sums[:, np.newaxis], ordered_labels


def load_calibration_spec(path: str | Path = CALIBRATION_CONFIG_PATH) -> CalibrationSpec:
    """Load the frozen G6 calibration declaration and reject any protocol drift."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "stage",
            "target",
            "labels",
            "expected_row_count",
            "candidate_experiments",
            "cross_fitting",
            "calibration_metrics",
            "risk_coverage",
            "deployment_temperature",
            "warnings",
        },
        "calibration config",
    )
    identity = {
        "schema_version": "1.0.0",
        "analysis_id": "g6-cross-fitted-calibration",
        "stage": "g6_cross_fitted_calibration_and_risk_coverage",
        "target": "season",
        "expected_row_count": 32_753,
    }
    mismatches = [name for name, expected in identity.items() if payload.get(name) != expected]
    if mismatches:
        raise ValueError(f"calibration identity changed: {mismatches}")
    if tuple(payload["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("calibration changed the canonical Season order")
    candidate_rows = payload["candidate_experiments"]
    if not isinstance(candidate_rows, list):
        raise ValueError("candidate experiments must be a list")
    candidates_list = []
    for index, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate experiment {index} must be an object")
        _require_exact_keys(
            row,
            {"candidate", "experiment_id", "seed"},
            f"candidate experiment {index}",
        )
        candidates_list.append(
            CalibrationCandidate(
                candidate=str(row["candidate"]),
                experiment_id=str(row["experiment_id"]),
                seed=_exact_int(row["seed"], "candidate seed"),
            )
        )
    candidates = tuple(candidates_list)
    if tuple((row.candidate, row.experiment_id, row.seed) for row in candidates) != (
        EXPECTED_CANDIDATES
    ):
        raise ValueError("calibration changed the primary-seed candidate pair")

    cross_fitting = dict(payload["cross_fitting"])
    _require_exact_keys(
        cross_fitting,
        {
            "evaluation_rule",
            "fit_rule",
            "folds",
            "objective",
            "optimizer",
            "optimizer_tolerance",
            "probability_floor",
            "probability_source",
            "pseudo_logit_rule",
            "temperature_bounds",
        },
        "cross-fitting protocol",
    )
    required_cross_fitting = {
        "evaluation_rule": "apply_each_temperature_only_to_its_held_out_oof_fold",
        "fit_rule": "fit_on_the_other_four_oof_folds_of_the_same_candidate",
        "objective": "multiclass_negative_log_likelihood",
        "optimizer": "bounded_scalar_minimisation",
        "probability_source": "frozen_softmax_oof_probabilities",
        "pseudo_logit_rule": "natural_log_of_clipped_probability",
    }
    if any(cross_fitting.get(key) != value for key, value in required_cross_fitting.items()):
        raise ValueError("calibration cross-fitting semantics changed")
    folds = tuple(_exact_int(value, "cross-fitting fold") for value in cross_fitting["folds"])
    bounds = tuple(float(value) for value in cross_fitting["temperature_bounds"])
    probability_floor = float(cross_fitting["probability_floor"])
    optimizer_tolerance = float(cross_fitting["optimizer_tolerance"])
    if (
        folds != tuple(range(5))
        or bounds != (0.05, 10.0)
        or probability_floor != 1e-12
        or optimizer_tolerance != 1e-8
    ):
        raise ValueError("calibration numeric protocol changed")

    metrics = dict(payload["calibration_metrics"])
    _require_exact_keys(
        metrics,
        {"ece_bins", "metrics", "reliability_definition", "reliability_strategy"},
        "calibration metrics",
    )
    if (
        _exact_int(metrics["ece_bins"], "ECE bin count") != 15
        or tuple(metrics["metrics"])
        != ("accuracy", "macro_f1", "nll", "brier", "ece", "mean_confidence")
        or metrics["reliability_definition"] != "top_label_confidence_vs_empirical_accuracy"
        or metrics["reliability_strategy"] != "equal_width"
    ):
        raise ValueError("calibration metric protocol changed")

    risk = dict(payload["risk_coverage"])
    _require_exact_keys(
        risk,
        {
            "coverage_start",
            "coverage_stop",
            "coverage_step",
            "ranking_rule",
            "retained_count_rounding",
            "review_budgets",
            "risk_definition",
            "threshold_selection_rule",
        },
        "risk-coverage protocol",
    )
    review_budgets = tuple(float(value) for value in risk["review_budgets"])
    if (
        float(risk["coverage_start"]) != 0.1
        or float(risk["coverage_stop"]) != 1.0
        or float(risk["coverage_step"]) != 0.01
        or risk["ranking_rule"] != "descending_calibrated_top_label_confidence_then_ascending_id"
        or risk["retained_count_rounding"] != "ceiling"
        or review_budgets != (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        or risk["risk_definition"] != "one_minus_accuracy_among_automatically_accepted_rows"
        or risk["threshold_selection_rule"]
        != "no_threshold_is_frozen_without_a_business_error_cost"
    ):
        raise ValueError("risk-coverage protocol changed")

    deployment = dict(payload["deployment_temperature"])
    if deployment != {
        "evaluation_claim_allowed": False,
        "fit_rule": "fit_one_scalar_on_all_primary_seed_oof_rows_after_cross_fitted_evaluation",
        "purpose": "future_frozen_bundle_confidence_only",
    }:
        raise ValueError("deployment-temperature boundary changed")
    warnings = dict(payload["warnings"])
    required_warnings = {
        "calibration_cannot_reopen_g5_model_selection",
        "holdout_is_forbidden",
        "pseudo_logits_are_equivalent_only_up_to_probability_clipping",
        "temperature_scaling_preserves_class_ranking",
        "ultimate_winner_remains_unfrozen",
    }
    if set(warnings) != required_warnings or not all(value is True for value in warnings.values()):
        raise ValueError("calibration safety warnings changed")
    return CalibrationSpec(
        analysis_id=str(payload["analysis_id"]),
        expected_row_count=_exact_int(payload["expected_row_count"], "expected row count"),
        candidates=candidates,
        folds=folds,
        probability_floor=probability_floor,
        temperature_bounds=(bounds[0], bounds[1]),
        optimizer_tolerance=optimizer_tolerance,
        ece_bins=_exact_int(metrics["ece_bins"], "ECE bin count"),
        coverage_start=float(risk["coverage_start"]),
        coverage_stop=float(risk["coverage_stop"]),
        coverage_step=float(risk["coverage_step"]),
        review_budgets=review_budgets,
    )


def top_label_reliability_bins(
    y_true: Sequence[str] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    labels: Sequence[str] = SEASON_LABELS,
    ece_bins: int = 15,
) -> pd.DataFrame:
    """Return every fixed-width top-label reliability bin, including empty bins."""
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least two")
    true, matrix, ordered_labels = _validated_probability_inputs(
        y_true,
        probabilities,
        labels,
        scope="reliability",
    )
    confidence = matrix.max(axis=1)
    predicted = np.asarray(ordered_labels, dtype=object)[matrix.argmax(axis=1)].astype(str)
    correct = predicted == true
    indices = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    rows = []
    for index in range(ece_bins):
        mask = indices == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else np.nan
        empirical_accuracy = float(correct[mask].mean()) if count else np.nan
        rows.append(
            {
                "bin": index,
                "lower_bound": index / ece_bins,
                "upper_bound": (index + 1) / ece_bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "calibration_gap": (empirical_accuracy - mean_confidence if count else np.nan),
            }
        )
    return pd.DataFrame(rows)


def _coverage_points(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step))
    points = tuple(round(start + index * step, 10) for index in range(count + 1))
    if not points or points[0] != start or points[-1] != stop:
        raise ValueError("coverage grid does not end on its declared bounds")
    return points


def risk_coverage_curve(
    ids: Sequence[int] | np.ndarray,
    y_true: Sequence[str] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    labels: Sequence[str] = SEASON_LABELS,
    coverage_start: float = 0.1,
    coverage_stop: float = 1.0,
    coverage_step: float = 0.01,
) -> pd.DataFrame:
    """Rank confidence deterministically and measure risk among automatically accepted rows."""
    identifiers = _int64_vector(ids, "risk-coverage IDs")
    true, matrix, ordered_labels = _validated_probability_inputs(
        y_true,
        probabilities,
        labels,
        scope="risk-coverage",
    )
    if len(identifiers) != len(true):
        raise ValueError("risk-coverage IDs and targets must be equal-length vectors")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("risk-coverage IDs must be unique")
    if not 0 < coverage_start <= coverage_stop <= 1 or coverage_step <= 0:
        raise ValueError("risk-coverage bounds are invalid")
    confidence = matrix.max(axis=1)
    predicted = np.asarray(ordered_labels, dtype=object)[matrix.argmax(axis=1)].astype(str)
    correct = predicted == true
    order = np.lexsort((identifiers, -confidence))
    rows = []
    for requested in _coverage_points(coverage_start, coverage_stop, coverage_step):
        retained = min(len(true), int(math.ceil(requested * len(true))))
        selected = order[:retained]
        selected_correct = int(correct[selected].sum())
        metrics = multiclass_metrics(
            true[selected],
            probabilities=matrix[selected],
            labels=ordered_labels,
            y_pred=predicted[selected],
        )
        rows.append(
            {
                "requested_coverage": requested,
                "coverage": retained / len(true),
                "review_rate": 1.0 - retained / len(true),
                "retained_count": retained,
                "review_count": len(true) - retained,
                "confidence_threshold": float(confidence[selected[-1]]),
                "accepted_correct_count": selected_correct,
                "selective_accuracy": selected_correct / retained,
                "selective_risk": 1.0 - selected_correct / retained,
                "selective_macro_f1": float(metrics["macro_f1"]),
                "accepted_id_sha256": canonical_sha256(
                    sorted(int(value) for value in identifiers[selected])
                ),
            }
        )
    return pd.DataFrame(rows)


def analyse_candidate_calibration(
    pack: CandidateOOFPack,
    spec: CalibrationSpec,
) -> CalibrationTables:
    """Cross-fit one candidate and return calibrated evidence without touching holdout data."""
    identity = (pack.candidate, pack.experiment_id, pack.seed)
    if identity not in EXPECTED_CANDIDATES:
        raise ValueError(f"unexpected calibration candidate: {identity}")
    frame = pack.oof.copy()
    required = {"id", "fold", "y_true", "y_pred", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"calibration OOF lacks columns: {missing}")
    if len(frame) != spec.expected_row_count:
        raise ValueError("calibration OOF coverage changed")
    identifiers = _int64_vector(frame["id"].to_numpy(dtype=object), "calibration OOF IDs")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("calibration OOF coverage changed")
    fold_ids = _int64_vector(
        frame["fold"].to_numpy(dtype=object),
        "calibration OOF folds",
    )
    if set(fold_ids) != set(spec.folds):
        raise ValueError("calibration OOF fold set changed")
    frame["id"] = identifiers
    frame["fold"] = fold_ids
    true = frame["y_true"].astype(str).to_numpy()
    raw = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    raw_labels = np.asarray(SEASON_LABELS, dtype=object)[raw.argmax(axis=1)].astype(str)
    if not np.array_equal(raw_labels, frame["y_pred"].astype(str).to_numpy()):
        raise ValueError("calibration OOF labels differ from probability argmax")
    calibrated, fold_temperatures = cross_fit_temperature(
        raw,
        true,
        fold_ids,
        labels=SEASON_LABELS,
        expected_folds=spec.folds,
        temperature_bounds=spec.temperature_bounds,
        probability_floor=spec.probability_floor,
        optimizer_tolerance=spec.optimizer_tolerance,
    )
    fold_temperatures.insert(0, "seed", pack.seed)
    fold_temperatures.insert(0, "experiment_id", pack.experiment_id)
    fold_temperatures.insert(0, "candidate", pack.candidate)

    temperatures_by_fold = fold_temperatures.set_index("evaluation_fold")["temperature"]
    calibrated_oof = frame.loc[:, ["id", "fold", "y_true", "y_pred"]].copy()
    calibrated_oof.insert(0, "seed", pack.seed)
    calibrated_oof.insert(0, "experiment_id", pack.experiment_id)
    calibrated_oof.insert(0, "candidate", pack.candidate)
    calibrated_oof["temperature"] = calibrated_oof["fold"].map(temperatures_by_fold)
    if calibrated_oof["temperature"].isna().any():
        raise RuntimeError("calibration OOF row lacks its held-out-fold temperature")
    for index, column in enumerate(PROBABILITY_COLUMNS):
        calibrated_oof[column] = calibrated[:, index]

    summary_rows = []
    reliability_parts = []
    risk_parts = []
    for method, probabilities in zip(CALIBRATION_METHODS, (raw, calibrated), strict=True):
        metrics = multiclass_metrics(
            true,
            probabilities=probabilities,
            labels=SEASON_LABELS,
            y_pred=raw_labels,
            ece_bins=spec.ece_bins,
        )
        summary_rows.append(
            {
                "candidate": pack.candidate,
                "experiment_id": pack.experiment_id,
                "seed": pack.seed,
                "calibration_method": method,
                "support": len(frame),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["nll"]),
                "brier": float(metrics["brier"]),
                "ece": float(metrics["ece"]),
                "mean_confidence": float(probabilities.max(axis=1).mean()),
            }
        )
        reliability = top_label_reliability_bins(
            true,
            probabilities,
            labels=SEASON_LABELS,
            ece_bins=spec.ece_bins,
        )
        reliability.insert(0, "calibration_method", method)
        reliability.insert(0, "candidate", pack.candidate)
        reliability_parts.append(reliability)
        risk = risk_coverage_curve(
            frame["id"],
            true,
            probabilities,
            labels=SEASON_LABELS,
            coverage_start=spec.coverage_start,
            coverage_stop=spec.coverage_stop,
            coverage_step=spec.coverage_step,
        )
        risk.insert(0, "calibration_method", method)
        risk.insert(0, "candidate", pack.candidate)
        risk_parts.append(risk)
    summary = pd.DataFrame(summary_rows)
    raw_summary = summary.loc[summary["calibration_method"].eq("uncalibrated")].iloc[0]
    calibrated_summary = summary.loc[
        summary["calibration_method"].eq("cross_fitted_temperature")
    ].iloc[0]
    for metric in ("accuracy", "macro_f1"):
        if float(raw_summary[metric]) != float(calibrated_summary[metric]):
            raise RuntimeError(f"temperature scaling changed {metric}")
    for metric in ("nll", "brier", "ece", "mean_confidence"):
        summary[f"delta_{metric}_vs_uncalibrated"] = summary[metric] - float(raw_summary[metric])

    risk_coverage = pd.concat(risk_parts, ignore_index=True)
    budget_rows = []
    for method in CALIBRATION_METHODS:
        method_rows = risk_coverage.loc[risk_coverage["calibration_method"].eq(method)]
        for review_budget in spec.review_budgets:
            requested_coverage = round(1.0 - review_budget, 10)
            match = method_rows.loc[
                np.isclose(
                    method_rows["requested_coverage"],
                    requested_coverage,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(match) != 1:
                raise RuntimeError("review budget is absent from risk-coverage grid")
            row = match.iloc[0].to_dict()
            row["review_budget"] = review_budget
            budget_rows.append(row)

    deployment_temperature = fit_temperature(
        raw,
        true,
        labels=SEASON_LABELS,
        temperature_bounds=spec.temperature_bounds,
        probability_floor=spec.probability_floor,
        optimizer_tolerance=spec.optimizer_tolerance,
    )
    deployment = pd.DataFrame(
        [
            {
                "candidate": pack.candidate,
                "experiment_id": pack.experiment_id,
                "seed": pack.seed,
                "temperature": deployment_temperature,
                "fit_rows": len(frame),
                "fit_scope": "all_primary_seed_oof_rows",
                "purpose": "future_frozen_bundle_confidence_only",
                "evaluation_claim_allowed": False,
            }
        ]
    )
    return CalibrationTables(
        calibration_summary=summary,
        fold_temperatures=fold_temperatures,
        reliability_bins=pd.concat(reliability_parts, ignore_index=True),
        risk_coverage=risk_coverage,
        review_budget_summary=pd.DataFrame(budget_rows),
        deployment_temperatures=deployment,
        calibrated_oof=calibrated_oof,
    )


def analyse_calibration_packs(
    packs: Sequence[CandidateOOFPack],
    spec: CalibrationSpec,
) -> CalibrationTables:
    """Analyse the exact primary-seed C2/I2 pair in frozen order."""
    by_identity = {(pack.candidate, pack.experiment_id, pack.seed): pack for pack in packs}
    expected = tuple((row.candidate, row.experiment_id, row.seed) for row in spec.candidates)
    if set(by_identity) != set(expected) or len(by_identity) != len(packs):
        raise ValueError("calibration packs changed the frozen candidate pair")
    results = [analyse_candidate_calibration(by_identity[identity], spec) for identity in expected]
    return CalibrationTables(
        calibration_summary=pd.concat(
            [result.calibration_summary for result in results], ignore_index=True
        ),
        fold_temperatures=pd.concat(
            [result.fold_temperatures for result in results], ignore_index=True
        ),
        reliability_bins=pd.concat(
            [result.reliability_bins for result in results], ignore_index=True
        ),
        risk_coverage=pd.concat([result.risk_coverage for result in results], ignore_index=True),
        review_budget_summary=pd.concat(
            [result.review_budget_summary for result in results], ignore_index=True
        ),
        deployment_temperatures=pd.concat(
            [result.deployment_temperatures for result in results], ignore_index=True
        ),
        calibrated_oof=pd.concat([result.calibrated_oof for result in results], ignore_index=True),
    )


def _validated_reliability_plot_rows(reliability_bins: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate",
        "calibration_method",
        "bin",
        "lower_bound",
        "upper_bound",
        "count",
        "mean_confidence",
        "empirical_accuracy",
    }
    missing = sorted(required - set(reliability_bins))
    if missing:
        raise ValueError(f"reliability plot table lacks columns: {missing}")
    frame = reliability_bins.copy()
    expected_candidates = {"C2", "I2"}
    if set(frame["candidate"].astype(str)) != expected_candidates:
        raise ValueError("reliability plot requires the complete C2/I2 candidate pair")
    if set(frame["calibration_method"].astype(str)) != set(CALIBRATION_METHODS):
        raise ValueError("reliability plot requires both calibration methods")
    expected_bins = set(range(15))
    for candidate in ("C2", "I2"):
        for method in CALIBRATION_METHODS:
            mask = frame["candidate"].eq(candidate) & frame["calibration_method"].eq(method)
            rows = frame.loc[mask].copy()
            bins = _int64_vector(rows["bin"].to_numpy(dtype=object), "reliability plot bins")
            if len(rows) != 15 or set(bins) != expected_bins:
                raise ValueError("reliability plot requires every declared confidence bin")
            counts = pd.to_numeric(rows["count"], errors="raise").to_numpy(dtype=np.float64)
            if (
                not np.isfinite(counts).all()
                or (counts != np.trunc(counts)).any()
                or (counts < 0).any()
                or counts.sum() <= 0
            ):
                raise ValueError("reliability plot counts are invalid")
            lower = pd.to_numeric(rows["lower_bound"], errors="raise").to_numpy(dtype=float)
            upper = pd.to_numeric(rows["upper_bound"], errors="raise").to_numpy(dtype=float)
            order = np.argsort(bins)
            if not np.allclose(lower[order], np.arange(15) / 15, rtol=0.0, atol=1e-12):
                raise ValueError("reliability plot lower bounds changed")
            if not np.allclose(upper[order], np.arange(1, 16) / 15, rtol=0.0, atol=1e-12):
                raise ValueError("reliability plot upper bounds changed")
            populated = counts > 0
            plotted = rows.loc[populated, ["mean_confidence", "empirical_accuracy"]].to_numpy(
                dtype=float
            )
            if not np.isfinite(plotted).all() or ((plotted < 0) | (plotted > 1)).any():
                raise ValueError("reliability plot values are invalid")
    return frame.sort_values(["candidate", "calibration_method", "bin"]).reset_index(drop=True)


def _validated_risk_plot_rows(risk_coverage: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate",
        "calibration_method",
        "requested_coverage",
        "coverage",
        "selective_risk",
    }
    missing = sorted(required - set(risk_coverage))
    if missing:
        raise ValueError(f"risk plot table lacks columns: {missing}")
    frame = risk_coverage.copy()
    if set(frame["candidate"].astype(str)) != {"C2", "I2"}:
        raise ValueError("risk plot requires the complete C2/I2 candidate pair")
    if set(frame["calibration_method"].astype(str)) != set(CALIBRATION_METHODS):
        raise ValueError("risk plot requires both calibration methods")
    expected = np.asarray(_coverage_points(0.1, 1.0, 0.01), dtype=float)
    for candidate in ("C2", "I2"):
        for method in CALIBRATION_METHODS:
            mask = frame["candidate"].eq(candidate) & frame["calibration_method"].eq(method)
            rows = frame.loc[mask].copy()
            requested = pd.to_numeric(rows["requested_coverage"], errors="raise").to_numpy(
                dtype=float
            )
            values = rows.loc[:, ["coverage", "selective_risk"]].to_numpy(dtype=float)
            if (
                len(rows) != len(expected)
                or not np.isfinite(requested).all()
                or not np.allclose(np.sort(requested), expected, rtol=0.0, atol=1e-12)
                or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()
            ):
                raise ValueError("risk plot requires the complete finite coverage grid")
    return frame.sort_values(["candidate", "calibration_method", "requested_coverage"]).reset_index(
        drop=True
    )


def plot_calibration_reliability(
    reliability_bins: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot top-label reliability and confidence support before/after calibration."""
    reliability_bins = _validated_reliability_plot_rows(reliability_bins)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"uncalibrated": "#E4572E", "cross_fitted_temperature": "#2F66E8"}
    labels = {"uncalibrated": "Before", "cross_fitted_temperature": "Cross-fitted"}
    for column, candidate in enumerate(("C2", "I2")):
        candidate_rows = reliability_bins.loc[reliability_bins["candidate"].eq(candidate)]
        axes[0, column].plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Ideal")
        for method in CALIBRATION_METHODS:
            rows = candidate_rows.loc[
                candidate_rows["calibration_method"].eq(method) & candidate_rows["count"].gt(0)
            ]
            axes[0, column].plot(
                rows["mean_confidence"],
                rows["empirical_accuracy"],
                marker="o" if method == "uncalibrated" else "s",
                linewidth=2,
                color=colors[method],
                label=labels[method],
            )
            centres = (rows["lower_bound"] + rows["upper_bound"]) / 2
            axes[1, column].plot(
                centres,
                rows["count"],
                marker="o" if method == "uncalibrated" else "s",
                linewidth=2,
                color=colors[method],
                label=labels[method],
            )
        axes[0, column].set_title(f"{candidate}: top-label reliability")
        axes[0, column].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend()
        axes[1, column].set_title(f"{candidate}: confidence-bin support")
        axes[1, column].set(xlim=(0, 1), xlabel="Confidence bin centre", ylabel="Rows")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend()
    figure.suptitle(
        "Five-fold cross-fitted temperature calibration", fontsize=18, fontweight="bold"
    )
    output = Path(output_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output, buffer.getvalue())
    return output


def plot_risk_coverage(
    risk_coverage: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot selective error risk against automatic-acceptance coverage."""
    risk_coverage = _validated_risk_plot_rows(risk_coverage)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True)
    styles = {
        "uncalibrated": {"color": "#E4572E", "linestyle": "--", "label": "Before"},
        "cross_fitted_temperature": {
            "color": "#2F66E8",
            "linestyle": "-",
            "label": "Cross-fitted",
        },
    }
    for axis, candidate in zip(axes, ("C2", "I2"), strict=True):
        candidate_rows = risk_coverage.loc[risk_coverage["candidate"].eq(candidate)]
        for method in CALIBRATION_METHODS:
            rows = candidate_rows.loc[candidate_rows["calibration_method"].eq(method)]
            axis.plot(
                rows["coverage"],
                rows["selective_risk"],
                linewidth=2.5,
                **styles[method],
            )
        axis.set_title(candidate)
        axis.set(xlim=(0.1, 1.0), xlabel="Automatic coverage", ylabel="Error risk")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Confidence-ranked risk–coverage", fontsize=18, fontweight="bold")
    output = Path(output_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output, buffer.getvalue())
    return output


__all__ = [
    "analyse_calibration_packs",
    "analyse_candidate_calibration",
    "CALIBRATION_CONFIG_PATH",
    "CalibrationCandidate",
    "CalibrationSpec",
    "CalibrationTables",
    "load_calibration_spec",
    "plot_calibration_reliability",
    "plot_risk_coverage",
    "risk_coverage_curve",
    "top_label_reliability_bins",
]
