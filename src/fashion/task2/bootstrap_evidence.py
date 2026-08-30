"""Hash-linked paired product-family bootstrap evidence for Task 2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.bootstrap import (
    PAIRED_BOOTSTRAP_CONFIG_PATH,
    analyse_paired_bootstrap_packs,
    build_paired_bootstrap_decision,
    load_paired_bootstrap_spec,
    plot_paired_bootstrap_intervals,
)
from fashion.task2.calibration_evidence import CALIBRATION_IMPLEMENTATION_PATHS
from fashion.task2.evidence import _portable_artifact_path, _resolve_evidence_path
from fashion.task2.robustness_evidence import load_verified_slice_manifest
from fashion.task2.slice_evidence import (
    load_candidate_oof_packs,
    load_declared_config_hashes,
    load_verified_stability_manifest,
)
from fashion.task2.slices import load_slice_analysis_spec
from fashion.train.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256, verify_implementation_at_head
from fashion.train.metrics import SEASON_LABELS
from fashion.train.reproducibility import capture_git_state, capture_runtime

DEFAULT_CALIBRATION_MANIFEST = Path("results/evidence/task2/calibration/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "paired_bootstrap"
DEFAULT_TEMPORARY_DIRECTORY = Path("tmp/task2/paired_bootstrap")

BOOTSTRAP_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *CALIBRATION_IMPLEMENTATION_PATHS,
            "src/fashion/task2/bootstrap.py",
            "src/fashion/task2/bootstrap_evidence.py",
        )
    )
)

_CALIBRATION_MANIFEST_FIELDS = {
    "analysis_config",
    "analysis_role",
    "app_threshold_frozen",
    "artifacts",
    "candidate_selection_affected",
    "canonical_inputs",
    "cross_fitted_evaluation_claim_allowed",
    "decision_status",
    "deployment_temperature_evaluation_claim_allowed",
    "gate",
    "git_commit",
    "git_dirty",
    "holdout_opened",
    "implementation_files_at_head",
    "implementation_sha256",
    "input_predictions",
    "labels",
    "robustness_manifest",
    "runtime_sha256",
    "schema_version",
    "slice_manifest",
    "stability_coverage_sha256",
    "temporary_calibrated_oof",
    "ultimate_winner_frozen",
}

_CALIBRATION_ARTIFACTS = {
    "calibration_reliability_figure",
    "calibration_summary",
    "decision",
    "deployment_temperatures",
    "fold_temperatures",
    "reliability_bins",
    "review_budget_summary",
    "risk_coverage",
    "risk_coverage_figure",
    "runtime",
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
    if not isinstance(declaration, Mapping):
        raise ValueError(f"{name} declaration must be an object")
    _require_exact_keys(declaration, {"path", "sha256"}, name)
    path = _resolve_evidence_path(str(declaration["path"]), project_root=project_root)
    digest = str(declaration["sha256"])
    if len(digest) != 64:
        raise ValueError(f"{name} has an invalid SHA-256 declaration")
    verify_artifact(path, digest)
    return path


def _verify_declaration_mapping(
    declarations: Mapping[str, Any],
    *,
    project_root: Path,
    name: str,
) -> dict[str, Path]:
    if not isinstance(declarations, Mapping):
        raise ValueError(f"calibration manifest has no {name} mapping")
    return {
        str(key): _verify_declaration(
            declaration,
            project_root=project_root,
            name=f"calibration {name} {key}",
        )
        for key, declaration in declarations.items()
    }


def load_verified_calibration_manifest(
    path: str | Path = DEFAULT_CALIBRATION_MANIFEST,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify all bytes and safety flags declared by the closed calibration gate."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    if not resolved.is_file():
        raise ValueError(f"calibration evidence manifest does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("calibration evidence manifest must be an object")
    _require_exact_keys(manifest, _CALIBRATION_MANIFEST_FIELDS, "calibration manifest")

    expected_identity = {
        "schema_version": "1.0.0",
        "gate": "G6-CALIBRATION",
        "decision_status": "closed",
        "analysis_role": "development_oof_cross_fitted_calibration_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "git_dirty": False,
        "cross_fitted_evaluation_claim_allowed": True,
        "deployment_temperature_evaluation_claim_allowed": False,
        "app_threshold_frozen": False,
    }
    mismatches = [
        field for field, expected in expected_identity.items() if manifest[field] != expected
    ]
    if mismatches:
        raise ValueError(f"calibration evidence boundary changed: {mismatches}")
    if tuple(manifest["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("calibration manifest changed the canonical Season order")
    for field in (
        "implementation_sha256",
        "runtime_sha256",
        "stability_coverage_sha256",
    ):
        if len(str(manifest[field])) != 64:
            raise ValueError(f"calibration manifest has invalid {field}")
    if len(str(manifest["git_commit"])) != 40:
        raise ValueError("calibration manifest has invalid git_commit")
    implementation_files = manifest["implementation_files_at_head"]
    if not isinstance(implementation_files, list) or not implementation_files:
        raise ValueError("calibration manifest lacks implementation files")

    artifacts = _verify_declaration_mapping(
        manifest["artifacts"],
        project_root=root,
        name="artifacts",
    )
    if set(artifacts) != _CALIBRATION_ARTIFACTS:
        raise ValueError("calibration manifest changed its artifact set")
    canonical_inputs = _verify_declaration_mapping(
        manifest["canonical_inputs"],
        project_root=root,
        name="canonical_inputs",
    )
    if set(canonical_inputs) != {"splits", "label_maps"}:
        raise ValueError("calibration manifest changed its canonical input set")
    input_predictions = _verify_declaration_mapping(
        manifest["input_predictions"],
        project_root=root,
        name="input_predictions",
    )
    if len(input_predictions) != 10:
        raise ValueError("calibration manifest must retain ten primary-seed OOF files")

    analysis_config = _verify_declaration(
        manifest["analysis_config"],
        project_root=root,
        name="calibration analysis config",
    )
    robustness_manifest = _verify_declaration(
        manifest["robustness_manifest"],
        project_root=root,
        name="calibration robustness manifest",
    )
    slice_manifest = _verify_declaration(
        manifest["slice_manifest"],
        project_root=root,
        name="calibration slice manifest",
    )

    temporary = manifest["temporary_calibrated_oof"]
    if not isinstance(temporary, Mapping):
        raise ValueError("calibration temporary OOF declaration must be an object")
    _require_exact_keys(
        temporary,
        {"path", "row_count", "rows_per_candidate", "sha256"},
        "calibration temporary OOF",
    )
    temporary_path = _resolve_evidence_path(str(temporary["path"]), project_root=root)
    verify_artifact(temporary_path, str(temporary["sha256"]))
    if temporary["row_count"] != 65_506 or temporary["rows_per_candidate"] != {
        "C2": 32_753,
        "I2": 32_753,
    }:
        raise ValueError("calibration temporary OOF row counts changed")

    with artifacts["decision"].open(encoding="utf-8") as handle:
        decision = json.load(handle)
    decision_boundary = {
        "gate": "G6-CALIBRATION",
        "decision_status": "closed",
        "analysis_role": "development_oof_cross_fitted_calibration_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "cross_fitted_evaluation_claim_allowed": True,
        "deployment_temperature_evaluation_claim_allowed": False,
        "app_threshold_frozen": False,
    }
    decision_mismatches = [
        field for field, expected in decision_boundary.items() if decision.get(field) != expected
    ]
    if decision_mismatches:
        raise ValueError(f"calibration decision boundary changed: {decision_mismatches}")
    with artifacts["runtime"].open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    if canonical_sha256(runtime) != str(manifest["runtime_sha256"]):
        raise ValueError("calibration runtime semantic hash differs from its verified artifact")
    return (
        dict(manifest),
        resolved,
        {
            "artifacts": artifacts,
            "canonical_inputs": canonical_inputs,
            "input_predictions": input_predictions,
            "analysis_config": analysis_config,
            "robustness_manifest": robustness_manifest,
            "slice_manifest": slice_manifest,
            "temporary_calibrated_oof": temporary_path,
            "decision": decision,
        },
    )


def _portable_declaration(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "path": _portable_artifact_path(path, fallback_root=root),
        "sha256": compute_sha256(path),
    }


def build_paired_bootstrap_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = PAIRED_BOOTSTRAP_CONFIG_PATH,
    calibration_manifest_path: str | Path = DEFAULT_CALIBRATION_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Build G6 family-bootstrap evidence from verified frozen OOF predictions only."""
    root = Path(project_root)
    config_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    spec = load_paired_bootstrap_spec(config_path)
    calibration, resolved_calibration, calibration_sections = load_verified_calibration_manifest(
        calibration_manifest_path,
        project_root=root,
    )
    slice_manifest, resolved_slice, slice_sections = load_verified_slice_manifest(
        calibration_sections["slice_manifest"],
        project_root=root,
    )
    stability, resolved_stability, stability_sections = load_verified_stability_manifest(
        slice_sections["stability_manifest"],
        project_root=root,
    )

    coverage_hash = str(calibration["stability_coverage_sha256"])
    if {
        coverage_hash,
        str(slice_manifest.get("coverage_sha256", "")),
        str(stability.get("coverage_sha256", "")),
    } != {coverage_hash}:
        raise ValueError("paired bootstrap upstream OOF coverage hashes disagree")
    for name in ("splits", "label_maps"):
        if compute_sha256(calibration_sections["canonical_inputs"][name]) != compute_sha256(
            slice_sections["canonical_inputs"][name]
        ):
            raise ValueError(f"paired bootstrap upstream boundaries disagree on {name}")
    calibration_prediction_declarations = dict(calibration["input_predictions"])
    slice_prediction_declarations = dict(slice_manifest.get("input_predictions", {}))
    if len(slice_prediction_declarations) != 20 or any(
        slice_prediction_declarations.get(run_id) != declaration
        for run_id, declaration in calibration_prediction_declarations.items()
    ):
        raise ValueError("paired bootstrap calibration predictions differ from the slice boundary")

    split_path = calibration_sections["canonical_inputs"]["splits"]
    label_map_path = calibration_sections["canonical_inputs"]["label_maps"]
    splits = load_splits(split_path)
    with label_map_path.open(encoding="utf-8") as handle:
        label_maps = json.load(handle)
    season_map = label_maps.get("season", {})
    if tuple(season_map.get("classes", ())) != tuple(SEASON_LABELS) or dict(
        season_map.get("label_to_index", {})
    ) != {label: index for index, label in enumerate(SEASON_LABELS)}:
        raise ValueError("paired bootstrap changed the canonical Season label map")
    development = get_samples(splits, partition="development", target="season").reset_index(
        drop=True
    )
    if len(development) != spec.expected_row_count:
        raise ValueError("paired bootstrap development Season row count changed")
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
    packs, coverage_by_experiment, prediction_artifacts = load_candidate_oof_packs(
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
    if coverage_hashes != {coverage_hash}:
        raise ValueError("paired bootstrap OOF coverage differs from the frozen boundary")
    if prediction_artifacts != slice_prediction_declarations:
        raise ValueError("paired bootstrap prediction bytes differ from the slice boundary")

    git_state = capture_git_state(root)
    if git_state.get("commit") is None:
        raise ValueError("paired bootstrap evidence requires a Git commit")
    if git_state.get("dirty") is not False:
        raise ValueError("paired bootstrap evidence requires a clean tracked Git worktree")
    implementation_files = verify_implementation_at_head(
        *BOOTSTRAP_IMPLEMENTATION_PATHS,
        root=root,
    )
    implementation_hash = implementation_sha256(
        *BOOTSTRAP_IMPLEMENTATION_PATHS,
        root=root,
    )
    runtime = capture_runtime()
    runtime_sha256 = canonical_sha256(runtime)

    tables = analyse_paired_bootstrap_packs(packs, development, spec)
    decision = build_paired_bootstrap_decision(tables, spec)
    if (
        len(tables.observed_metrics) != 2
        or len(tables.interval_summary) != 12
        or len(tables.group_audit) != 1
        or len(tables.draws) != 2 * spec.replicates
    ):
        raise ValueError("paired bootstrap output row counts changed")

    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    figure_root = _resolve_evidence_path(figure_directory, project_root=root)
    temporary_root = _resolve_evidence_path(temporary_directory, project_root=root)
    paths = {
        "observed_metrics": evidence_root / "observed_metrics.csv",
        "interval_summary": evidence_root / "interval_summary.csv",
        "group_audit": evidence_root / "group_audit.csv",
        "registry_snapshot": evidence_root / "registry_snapshot.csv",
        "runtime": evidence_root / "runtime.json",
        "decision": evidence_root / "decision.json",
        "paired_bootstrap_figure": figure_root / "paired_group_bootstrap.png",
    }
    registry_snapshot = pd.concat(
        [pack.registry.assign(candidate=pack.candidate) for pack in packs],
        ignore_index=True,
    ).sort_values(["candidate", "seed", "fold", "run_id"], kind="stable")
    frames = {
        "observed_metrics": tables.observed_metrics,
        "interval_summary": tables.interval_summary,
        "group_audit": tables.group_audit,
        "registry_snapshot": registry_snapshot,
    }
    for name, frame in frames.items():
        atomic_write_csv(paths[name], frame)
    atomic_write_json(paths["runtime"], runtime)
    atomic_write_json(paths["decision"], decision)
    plot_paired_bootstrap_intervals(tables, paths["paired_bootstrap_figure"])

    draw_path = temporary_root / "bootstrap_draws.csv"
    atomic_write_csv(draw_path, tables.draws)
    artifacts = {name: _portable_declaration(path, root=root) for name, path in paths.items()}
    input_predictions = {
        run_id: {
            "path": _portable_artifact_path(path, fallback_root=root),
            "sha256": compute_sha256(path),
        }
        for run_id, path in sorted(slice_sections["input_predictions"].items())
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-PAIRED-BOOTSTRAP",
        "decision_status": "closed",
        "analysis_role": "development_oof_fitted_pair_uncertainty_only",
        "candidate_selection_affected": False,
        "new_candidates_allowed": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "random_seed_generalisability_claim_allowed": False,
        "labels": list(SEASON_LABELS),
        "git_commit": str(git_state["commit"]),
        "git_dirty": False,
        "analysis_config": _portable_declaration(config_path, root=root),
        "calibration_manifest": _portable_declaration(resolved_calibration, root=root),
        "slice_manifest": _portable_declaration(resolved_slice, root=root),
        "stability_manifest": _portable_declaration(resolved_stability, root=root),
        "stability_coverage_sha256": coverage_hash,
        "slice_assignment_sha256": str(slice_manifest["slice_assignment_sha256"]),
        "canonical_inputs": {
            "splits": _portable_declaration(split_path, root=root),
            "label_maps": _portable_declaration(label_map_path, root=root),
        },
        "implementation_sha256": implementation_hash,
        "implementation_files_at_head": list(implementation_files),
        "runtime_sha256": runtime_sha256,
        "input_predictions": input_predictions,
        "temporary_bootstrap_draws": {
            "path": _portable_artifact_path(draw_path, fallback_root=root),
            "sha256": compute_sha256(draw_path),
            "row_count": len(tables.draws),
            "rows_per_comparison": {
                str(comparison_id): int(count)
                for comparison_id, count in tables.draws.groupby("comparison_id")["replicate"]
                .count()
                .items()
            },
        },
        "artifacts": artifacts,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "BOOTSTRAP_IMPLEMENTATION_PATHS",
    "build_paired_bootstrap_evidence",
    "load_verified_calibration_manifest",
]
