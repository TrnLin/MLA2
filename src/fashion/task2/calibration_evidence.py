"""Audited G6 cross-fitted calibration evidence for Task 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.calibration import (
    CALIBRATION_CONFIG_PATH,
    CALIBRATION_METHODS,
    CalibrationSpec,
    CalibrationTables,
    analyse_calibration_packs,
    load_calibration_spec,
    plot_calibration_reliability,
    plot_risk_coverage,
)
from fashion.task2.evidence import _portable_artifact_path, _resolve_evidence_path
from fashion.task2.robustness import ROBUSTNESS_IMPLEMENTATION_PATHS
from fashion.task2.robustness_evidence import load_verified_slice_manifest
from fashion.task2.slice_evidence import (
    load_candidate_oof_packs,
    load_declared_config_hashes,
    load_verified_stability_manifest,
)
from fashion.task2.slices import CandidateOOFPack, load_slice_analysis_spec
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256, verify_implementation_at_head
from fashion.train.metrics import SEASON_LABELS
from fashion.train.reproducibility import capture_git_state, capture_runtime

DEFAULT_ROBUSTNESS_MANIFEST = Path("results/evidence/task2/robustness_cost/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "calibration"
DEFAULT_TEMPORARY_DIRECTORY = Path("tmp/task2/calibration")

CALIBRATION_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *ROBUSTNESS_IMPLEMENTATION_PATHS,
            "src/fashion/task2/stability.py",
            "src/fashion/task2/stability_evidence.py",
            "src/fashion/task2/calibration.py",
            "src/fashion/task2/calibration_evidence.py",
        )
    )
)

_ROBUSTNESS_MANIFEST_FIELDS = {
    "analysis_config",
    "analysis_role",
    "artifacts",
    "candidate_selection_affected",
    "canonical_inputs",
    "decision_status",
    "gate",
    "git_commit",
    "git_dirty",
    "implementation_files_at_head",
    "implementation_sha256",
    "input_checkpoints",
    "input_cost_results",
    "input_probe_predictions",
    "runtime_sha256",
    "schema_version",
    "slice_manifest",
    "stability_coverage_sha256",
    "ultimate_winner_frozen",
}


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _verify_declaration(
    declaration: Mapping[str, Any],
    *,
    project_root: Path,
    name: str,
) -> Path:
    _require_exact_keys(declaration, {"path", "sha256"}, name)
    path = _resolve_evidence_path(str(declaration["path"]), project_root=project_root)
    digest = str(declaration["sha256"])
    if len(digest) != 64:
        raise ValueError(f"{name} has an invalid SHA-256 declaration")
    verify_artifact(path, digest)
    return path


def _verified_section(
    manifest: Mapping[str, Any],
    section: str,
    *,
    project_root: Path,
) -> dict[str, Path]:
    declarations = manifest.get(section)
    if not isinstance(declarations, Mapping):
        raise ValueError(f"robustness manifest has no {section} mapping")
    return {
        str(name): _verify_declaration(
            declaration,
            project_root=project_root,
            name=f"robustness {section} {name}",
        )
        for name, declaration in declarations.items()
    }


def load_verified_robustness_manifest(
    path: str | Path = DEFAULT_ROBUSTNESS_MANIFEST,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify every byte declared by the closed robustness/cost boundary."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    if not resolved.is_file():
        raise ValueError(f"robustness evidence manifest does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("robustness evidence manifest must be an object")
    _require_exact_keys(manifest, _ROBUSTNESS_MANIFEST_FIELDS, "robustness manifest")
    expected_identity = {
        "schema_version": "1.0.0",
        "gate": "G6-ROBUSTNESS-COST",
        "decision_status": "closed",
        "analysis_role": "development_stress_and_machine_cost_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "git_dirty": False,
    }
    mismatches = [
        field for field, expected in expected_identity.items() if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"robustness evidence boundary changed: {mismatches}")
    for field in (
        "git_commit",
        "implementation_sha256",
        "runtime_sha256",
        "stability_coverage_sha256",
    ):
        expected_length = 40 if field == "git_commit" else 64
        if len(str(manifest[field])) != expected_length:
            raise ValueError(f"robustness manifest has invalid {field}")
    implementation_files = manifest["implementation_files_at_head"]
    if not isinstance(implementation_files, list) or not implementation_files:
        raise ValueError("robustness manifest lacks implementation files")

    artifacts = _verified_section(manifest, "artifacts", project_root=root)
    required_artifacts = {"decision", "pooled_metrics", "clean_reconciliation", "runtime"}
    if not required_artifacts.issubset(artifacts):
        raise ValueError("robustness evidence lacks calibration boundary artifacts")
    canonical_inputs = _verified_section(manifest, "canonical_inputs", project_root=root)
    if set(canonical_inputs) != {"splits", "label_maps"}:
        raise ValueError("robustness evidence changed its canonical inputs")
    analysis_config = _verify_declaration(
        manifest["analysis_config"],
        project_root=root,
        name="robustness analysis config",
    )
    slice_manifest = _verify_declaration(
        manifest["slice_manifest"],
        project_root=root,
        name="robustness upstream slice manifest",
    )
    checkpoints = _verified_section(manifest, "input_checkpoints", project_root=root)
    probe_predictions = _verified_section(
        manifest,
        "input_probe_predictions",
        project_root=root,
    )
    cost_results = _verified_section(manifest, "input_cost_results", project_root=root)
    if len(checkpoints) != 10 or len(probe_predictions) != 50 or len(cost_results) != 4:
        raise ValueError("robustness evidence changed its 10/50/4 input boundary")
    with artifacts["decision"].open(encoding="utf-8") as handle:
        decision = json.load(handle)
    if (
        decision.get("gate") != "G6-ROBUSTNESS-COST"
        or decision.get("decision_status") != "closed"
        or decision.get("current_candidate") != "I2"
        or decision.get("candidate_selection_affected") is not False
        or decision.get("ultimate_winner_frozen") is not False
    ):
        raise ValueError("robustness decision no longer preserves the G5 I2 candidate")
    with artifacts["runtime"].open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    if canonical_sha256(runtime) != str(manifest["runtime_sha256"]):
        raise ValueError("robustness runtime semantic hash differs from its verified artifact")
    return (
        dict(manifest),
        resolved,
        {
            "artifacts": artifacts,
            "canonical_inputs": canonical_inputs,
            "analysis_config": analysis_config,
            "slice_manifest": slice_manifest,
            "input_checkpoints": checkpoints,
            "input_probe_predictions": probe_predictions,
            "input_cost_results": cost_results,
            "decision": decision,
        },
    )


def _summary_row(
    summary: pd.DataFrame,
    *,
    candidate: str,
    method: str,
) -> pd.Series:
    rows = summary.loc[
        summary["candidate"].eq(candidate) & summary["calibration_method"].eq(method)
    ]
    if len(rows) != 1:
        raise ValueError(f"calibration summary lacks one {candidate}/{method} row")
    return rows.iloc[0]


def _exact_integer_values(series: pd.Series, scope: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values != np.trunc(values)).any():
        raise ValueError(f"{scope} must contain finite integers")
    return values.astype(np.int64)


def _validate_temperature_contract(
    tables: CalibrationTables,
    spec: CalibrationSpec,
) -> None:
    fold_required = {
        "candidate",
        "experiment_id",
        "seed",
        "evaluation_fold",
        "fit_folds",
        "fit_fold_count",
        "calibration_rows",
        "evaluation_rows",
        "temperature",
        "fit_nll_before",
        "fit_nll_after",
    }
    if not fold_required.issubset(tables.fold_temperatures):
        raise ValueError("fold-temperature audit is incomplete")
    deployment_required = {
        "candidate",
        "experiment_id",
        "seed",
        "temperature",
        "fit_rows",
        "fit_scope",
        "purpose",
        "evaluation_claim_allowed",
    }
    if not deployment_required.issubset(tables.deployment_temperatures):
        raise ValueError("deployment-temperature audit is incomplete")

    expected_folds = tuple(spec.folds)
    expected_fold_set = set(expected_folds)
    expected_candidates = {candidate.candidate for candidate in spec.candidates}
    if (
        len(tables.fold_temperatures) != len(expected_candidates) * len(expected_folds)
        or set(tables.fold_temperatures["candidate"].astype(str)) != expected_candidates
    ):
        raise ValueError("fold-temperature audit changed the frozen candidate set")
    if (
        len(tables.deployment_temperatures) != len(expected_candidates)
        or set(tables.deployment_temperatures["candidate"].astype(str)) != expected_candidates
    ):
        raise ValueError("deployment-temperature audit changed the frozen candidate set")
    bounds = spec.temperature_bounds
    for candidate_spec in spec.candidates:
        candidate = candidate_spec.candidate
        rows = tables.fold_temperatures.loc[
            tables.fold_temperatures["candidate"].eq(candidate)
        ].copy()
        if len(rows) != len(expected_folds):
            raise ValueError(f"{candidate} fold-temperature audit must contain five rows")
        if set(rows["experiment_id"].astype(str)) != {candidate_spec.experiment_id}:
            raise ValueError(f"{candidate} fold-temperature experiment identity changed")
        if set(_exact_integer_values(rows["seed"], f"{candidate} fold-temperature seed")) != {
            candidate_spec.seed
        }:
            raise ValueError(f"{candidate} fold-temperature seed changed")
        evaluation_folds = _exact_integer_values(
            rows["evaluation_fold"],
            f"{candidate} evaluation folds",
        )
        if len(set(evaluation_folds)) != len(expected_folds) or set(evaluation_folds) != (
            expected_fold_set
        ):
            raise ValueError(f"{candidate} evaluation folds must contain 0-4 exactly once")
        temperatures = pd.to_numeric(rows["temperature"], errors="raise").to_numpy(dtype=float)
        nll_values = rows.loc[:, ["fit_nll_before", "fit_nll_after"]].to_numpy(dtype=float)
        if (
            not np.isfinite(temperatures).all()
            or (temperatures < bounds[0]).any()
            or (temperatures > bounds[1]).any()
            or not np.isfinite(nll_values).all()
        ):
            raise ValueError(f"{candidate} fold-temperature values are invalid")
        evaluation_rows = _exact_integer_values(
            rows["evaluation_rows"],
            f"{candidate} evaluation row counts",
        )
        calibration_rows = _exact_integer_values(
            rows["calibration_rows"],
            f"{candidate} calibration row counts",
        )
        fit_fold_counts = _exact_integer_values(
            rows["fit_fold_count"],
            f"{candidate} fit-fold counts",
        )
        if (
            (evaluation_rows <= 0).any()
            or int(evaluation_rows.sum()) != spec.expected_row_count
            or not np.array_equal(calibration_rows, spec.expected_row_count - evaluation_rows)
            or not np.all(fit_fold_counts == len(expected_folds) - 1)
        ):
            raise ValueError(f"{candidate} cross-fitting row/fold counts changed")
        for position, row in enumerate(rows.itertuples(index=False)):
            evaluation_fold = int(evaluation_folds[position])
            expected_fit_folds = tuple(fold for fold in expected_folds if fold != evaluation_fold)
            if str(row.fit_folds) != "|".join(str(fold) for fold in expected_fit_folds):
                raise ValueError(f"{candidate} fit_folds includes its evaluation fold")

        deployment_rows = tables.deployment_temperatures.loc[
            tables.deployment_temperatures["candidate"].eq(candidate)
        ]
        if len(deployment_rows) != 1:
            raise ValueError(f"{candidate} deployment-temperature audit must contain one row")
        deployment = deployment_rows.iloc[0]
        deployment_seed = int(
            _exact_integer_values(
                pd.Series([deployment["seed"]]),
                f"{candidate} deployment seed",
            )[0]
        )
        deployment_fit_rows = int(
            _exact_integer_values(
                pd.Series([deployment["fit_rows"]]),
                f"{candidate} deployment fit rows",
            )[0]
        )
        if (
            str(deployment["experiment_id"]) != candidate_spec.experiment_id
            or deployment_seed != candidate_spec.seed
            or deployment_fit_rows != spec.expected_row_count
            or str(deployment["fit_scope"]) != "all_primary_seed_oof_rows"
            or str(deployment["purpose"]) != "future_frozen_bundle_confidence_only"
            or not isinstance(deployment["evaluation_claim_allowed"], (bool, np.bool_))
            or bool(deployment["evaluation_claim_allowed"])
        ):
            raise ValueError(f"{candidate} deployment-temperature boundary changed")
        deployment_temperature = float(deployment["temperature"])
        if not np.isfinite(deployment_temperature) or not (
            bounds[0] <= deployment_temperature <= bounds[1]
        ):
            raise ValueError(f"{candidate} deployment temperature is invalid")


def build_calibration_decision(
    tables: CalibrationTables,
    spec: CalibrationSpec,
) -> dict[str, Any]:
    """Summarise measured probability quality without reopening model selection."""
    expected_candidates = tuple(candidate.candidate for candidate in spec.candidates)
    if expected_candidates != ("C2", "I2"):
        raise ValueError("calibration decision requires the frozen C2/I2 order")
    _validate_temperature_contract(tables, spec)
    required_summary = {
        "candidate",
        "calibration_method",
        "accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece",
        "mean_confidence",
    }
    if not required_summary.issubset(tables.calibration_summary):
        raise ValueError("calibration decision summary is incomplete")

    outcomes: dict[str, Any] = {}
    review_snapshot: dict[str, Any] = {}
    for candidate in expected_candidates:
        before = _summary_row(
            tables.calibration_summary,
            candidate=candidate,
            method="uncalibrated",
        )
        after = _summary_row(
            tables.calibration_summary,
            candidate=candidate,
            method="cross_fitted_temperature",
        )
        if float(before["accuracy"]) != float(after["accuracy"]) or float(
            before["macro_f1"]
        ) != float(after["macro_f1"]):
            raise ValueError("calibration decision detected changed class predictions")
        metric_rows = {}
        for metric in ("nll", "brier", "ece", "mean_confidence"):
            before_value = float(before[metric])
            after_value = float(after[metric])
            metric_rows[metric] = {
                "before": before_value,
                "cross_fitted": after_value,
                "delta": after_value - before_value,
            }
        temperatures = tables.fold_temperatures.loc[
            tables.fold_temperatures["candidate"].eq(candidate), "temperature"
        ].to_numpy(dtype=float)
        if len(temperatures) != 5 or not np.isfinite(temperatures).all():
            raise ValueError("calibration decision lacks five finite fold temperatures")
        outcomes[candidate] = {
            "probability_metrics": metric_rows,
            "cross_fitted_temperature_min": float(temperatures.min()),
            "cross_fitted_temperature_max": float(temperatures.max()),
            "nll_improved": bool(metric_rows["nll"]["delta"] < 0),
            "brier_improved": bool(metric_rows["brier"]["delta"] < 0),
            "ece_improved": bool(metric_rows["ece"]["delta"] < 0),
        }
        budget_rows = tables.review_budget_summary.loc[
            tables.review_budget_summary["candidate"].eq(candidate)
            & tables.review_budget_summary["calibration_method"].eq("cross_fitted_temperature")
            & np.isclose(
                tables.review_budget_summary["review_budget"],
                0.2,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if len(budget_rows) != 1:
            raise ValueError("calibration decision lacks the declared 20% review snapshot")
        budget = budget_rows.iloc[0]
        review_snapshot[candidate] = {
            "declared_review_budget": 0.2,
            "actual_review_rate": float(budget["review_rate"]),
            "automatic_coverage": float(budget["coverage"]),
            "selective_risk": float(budget["selective_risk"]),
            "selective_macro_f1": float(budget["selective_macro_f1"]),
        }

    return {
        "schema_version": "1.0.0",
        "gate": "G6-CALIBRATION",
        "decision_status": "closed",
        "analysis_role": "development_oof_cross_fitted_calibration_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "temperature_scaling_preserved_class_predictions": True,
        "cross_fitted_evaluation_claim_allowed": True,
        "deployment_temperature_evaluation_claim_allowed": False,
        "app_threshold_frozen": False,
        "probability_quality_by_candidate": outcomes,
        "diagnostic_review_budget_20_percent": review_snapshot,
        "interpretation_rule": (
            "Only five-fold cross-fitted temperatures support evaluation claims. The scalar "
            "fit on all OOF rows is future bundle metadata and is not evaluation evidence."
        ),
        "review_policy": (
            "Risk-coverage is diagnostic only; no confidence threshold is frozen without an "
            "explicit business error cost."
        ),
        "limitations": [
            "Temperature scaling changes confidence but cannot improve class predictions.",
            "Different held-out folds use different fitted temperatures, so global row-confidence "
            "ordering can change across folds.",
            "Calibration is measured only on development OOF predictions; holdout remains sealed.",
        ],
        "next_question": "Estimate the paired grouped-bootstrap uncertainty between C2 and I2.",
    }


def _select_calibration_packs(
    packs: list[CandidateOOFPack],
    spec: CalibrationSpec,
) -> list[CandidateOOFPack]:
    by_identity = {(pack.candidate, pack.experiment_id, pack.seed): pack for pack in packs}
    selected = []
    for candidate in spec.candidates:
        identity = (candidate.candidate, candidate.experiment_id, candidate.seed)
        if identity not in by_identity:
            raise ValueError(f"calibration lacks frozen OOF pack: {identity}")
        selected.append(by_identity[identity])
    if len(selected) != 2:
        raise ValueError("calibration requires exactly two primary-seed OOF packs")
    return selected


def _validate_output_contract(tables: CalibrationTables, spec: CalibrationSpec) -> None:
    expected_rows = {
        "calibration_summary": 4,
        "fold_temperatures": 10,
        "reliability_bins": 60,
        "risk_coverage": 364,
        "review_budget_summary": 24,
        "deployment_temperatures": 2,
        "calibrated_oof": 2 * spec.expected_row_count,
    }
    for name, expected in expected_rows.items():
        actual = len(getattr(tables, name))
        if actual != expected:
            raise ValueError(f"calibration {name} row count changed: {actual} != {expected}")
    support = tables.calibrated_oof.groupby("candidate", sort=True)["id"].nunique().to_dict()
    if support != {"C2": spec.expected_row_count, "I2": spec.expected_row_count}:
        raise ValueError("calibrated OOF coverage is incomplete")
    if set(tables.calibration_summary["calibration_method"]) != set(CALIBRATION_METHODS):
        raise ValueError("calibration output methods changed")


def build_calibration_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = CALIBRATION_CONFIG_PATH,
    robustness_manifest_path: str | Path = DEFAULT_ROBUSTNESS_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Build calibrated OOF evidence from verified frozen predictions only."""
    root = Path(project_root)
    config_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    spec = load_calibration_spec(config_path)
    robustness, resolved_robustness, robustness_sections = load_verified_robustness_manifest(
        robustness_manifest_path,
        project_root=root,
    )
    slice_manifest, resolved_slice, slice_sections = load_verified_slice_manifest(
        robustness_sections["slice_manifest"],
        project_root=root,
    )
    stability, _, stability_sections = load_verified_stability_manifest(
        slice_sections["stability_manifest"],
        project_root=root,
    )

    split_path = robustness_sections["canonical_inputs"]["splits"]
    label_map_path = robustness_sections["canonical_inputs"]["label_maps"]
    for name, path in {"splits": split_path, "label_maps": label_map_path}.items():
        if compute_sha256(path) != compute_sha256(slice_sections["canonical_inputs"][name]):
            raise ValueError(f"calibration upstream boundaries disagree on {name}")
    splits = load_splits(split_path)
    with label_map_path.open(encoding="utf-8") as handle:
        label_maps = json.load(handle)
    season_map = label_maps.get("season", {})
    if tuple(season_map.get("classes", ())) != tuple(SEASON_LABELS):
        raise ValueError("calibration changed the canonical Season labels")
    if dict(season_map.get("label_to_index", {})) != {
        label: index for index, label in enumerate(SEASON_LABELS)
    }:
        raise ValueError("calibration changed the canonical Season indices")

    development = get_samples(splits, partition="development", target="season").reset_index(
        drop=True
    )
    if len(development) != spec.expected_row_count:
        raise ValueError("calibration development Season row count changed")
    expected_ids = development["id"].astype(int).tolist()
    expected_targets = dict(
        zip(development["id"].astype(int), development["season"].astype(str), strict=True)
    )
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"].astype(int)
    )
    registry = pd.read_csv(
        slice_sections["artifacts"]["registry_snapshot"],
        dtype=str,
        keep_default_na=False,
    )
    stability_summary = pd.read_csv(stability_sections["artifacts"]["seed_stability"])
    config_hashes = load_declared_config_hashes(stability_sections["input_configs"])
    slice_spec = load_slice_analysis_spec(slice_sections["analysis_config"])
    all_packs, coverage_by_experiment, prediction_artifacts = load_candidate_oof_packs(
        registry,
        slice_spec,
        project_root=root,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        protected_ids=protected_ids,
        split_sha256=compute_sha256(split_path),
        label_map_sha256=compute_sha256(label_map_path),
        config_sha256_by_experiment=config_hashes,
        stability_summary=stability_summary,
    )
    coverage_hashes = {canonical_sha256(value) for value in coverage_by_experiment.values()}
    if coverage_hashes != {str(robustness["stability_coverage_sha256"])} or coverage_hashes != {
        str(stability["coverage_sha256"])
    }:
        raise ValueError("calibration OOF coverage differs from the frozen G5/G6 boundary")
    selected_packs = _select_calibration_packs(all_packs, spec)
    selected_run_ids = {
        str(run_id)
        for pack in selected_packs
        for run_id in pack.registry["run_id"].astype(str).tolist()
    }
    if len(selected_run_ids) != 10:
        raise ValueError("calibration requires exactly ten primary-seed OOF fold files")
    selected_predictions = {
        run_id: prediction_artifacts[run_id] for run_id in sorted(selected_run_ids)
    }
    for run_id, declaration in selected_predictions.items():
        if declaration != slice_manifest["input_predictions"].get(run_id):
            raise ValueError(f"calibration prediction declaration drifted for {run_id}")

    git_state = capture_git_state(root)
    if git_state.get("commit") is None:
        raise ValueError("calibration evidence requires a Git commit")
    if git_state.get("dirty") is not False:
        raise ValueError("calibration evidence requires a clean tracked Git worktree")
    implementation_files = verify_implementation_at_head(
        *CALIBRATION_IMPLEMENTATION_PATHS,
        root=root,
    )
    implementation_hash = implementation_sha256(
        *CALIBRATION_IMPLEMENTATION_PATHS,
        root=root,
    )
    runtime = capture_runtime()
    runtime_sha256 = canonical_sha256(runtime)

    tables = analyse_calibration_packs(selected_packs, spec)
    _validate_output_contract(tables, spec)
    decision = build_calibration_decision(tables, spec)

    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    figure_root = _resolve_evidence_path(figure_directory, project_root=root)
    temporary_root = _resolve_evidence_path(temporary_directory, project_root=root)
    paths = {
        "calibration_summary": evidence_root / "calibration_summary.csv",
        "fold_temperatures": evidence_root / "fold_temperatures.csv",
        "reliability_bins": evidence_root / "reliability_bins.csv",
        "risk_coverage": evidence_root / "risk_coverage.csv",
        "review_budget_summary": evidence_root / "review_budget_summary.csv",
        "deployment_temperatures": evidence_root / "deployment_temperatures.csv",
        "runtime": evidence_root / "runtime.json",
        "decision": evidence_root / "decision.json",
        "calibration_reliability_figure": figure_root / "calibration_reliability.png",
        "risk_coverage_figure": figure_root / "risk_coverage.png",
    }
    frames = {
        "calibration_summary": tables.calibration_summary,
        "fold_temperatures": tables.fold_temperatures,
        "reliability_bins": tables.reliability_bins,
        "risk_coverage": tables.risk_coverage,
        "review_budget_summary": tables.review_budget_summary,
        "deployment_temperatures": tables.deployment_temperatures,
    }
    for name, frame in frames.items():
        atomic_write_csv(paths[name], frame)
    atomic_write_json(paths["runtime"], runtime)
    atomic_write_json(paths["decision"], decision)
    plot_calibration_reliability(
        tables.reliability_bins,
        paths["calibration_reliability_figure"],
    )
    plot_risk_coverage(tables.risk_coverage, paths["risk_coverage_figure"])

    calibrated_oof_path = temporary_root / "calibrated_oof.csv"
    calibrated_oof = tables.calibrated_oof.sort_values(
        ["candidate", "id"],
        kind="stable",
    ).reset_index(drop=True)
    atomic_write_csv(calibrated_oof_path, calibrated_oof)
    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-CALIBRATION",
        "decision_status": "closed",
        "analysis_role": "development_oof_cross_fitted_calibration_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "cross_fitted_evaluation_claim_allowed": True,
        "deployment_temperature_evaluation_claim_allowed": False,
        "app_threshold_frozen": False,
        "holdout_opened": False,
        "labels": list(SEASON_LABELS),
        "git_commit": str(git_state["commit"]),
        "git_dirty": False,
        "analysis_config": {
            "path": _portable_artifact_path(config_path, fallback_root=root),
            "sha256": compute_sha256(config_path),
        },
        "robustness_manifest": {
            "path": _portable_artifact_path(resolved_robustness, fallback_root=root),
            "sha256": compute_sha256(resolved_robustness),
        },
        "slice_manifest": {
            "path": _portable_artifact_path(resolved_slice, fallback_root=root),
            "sha256": compute_sha256(resolved_slice),
        },
        "stability_coverage_sha256": next(iter(coverage_hashes)),
        "canonical_inputs": {
            "splits": {
                "path": _portable_artifact_path(split_path, fallback_root=root),
                "sha256": compute_sha256(split_path),
            },
            "label_maps": {
                "path": _portable_artifact_path(label_map_path, fallback_root=root),
                "sha256": compute_sha256(label_map_path),
            },
        },
        "implementation_sha256": implementation_hash,
        "implementation_files_at_head": list(implementation_files),
        "runtime_sha256": runtime_sha256,
        "input_predictions": selected_predictions,
        "temporary_calibrated_oof": {
            "path": _portable_artifact_path(calibrated_oof_path, fallback_root=root),
            "sha256": compute_sha256(calibrated_oof_path),
            "row_count": len(calibrated_oof),
            "rows_per_candidate": {
                str(candidate): int(count)
                for candidate, count in calibrated_oof.groupby("candidate")["id"].count().items()
            },
        },
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "build_calibration_decision",
    "build_calibration_evidence",
    "CALIBRATION_IMPLEMENTATION_PATHS",
    "load_verified_robustness_manifest",
]
