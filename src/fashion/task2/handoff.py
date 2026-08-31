"""Verified Task 2 component handoff that keeps final evaluation locked."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from fashion.config import (
    ROOT,
    RUNS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_MODEL_MANIFEST_JSON,
)
from fashion.data.hashing import compute_sha256
from fashion.task2.inference import SeasonPrediction
from fashion.task2.refit import load_verified_development_refit_manifest
from fashion.task2.ultimate_judgement import (
    load_verified_selection_freeze,
    load_verified_ultimate_judgement_manifest,
)
from fashion.train.artifacts import atomic_write_csv, atomic_write_json, verify_artifact

DEFAULT_HANDOFF_DIRECTORY = TASK2_EVIDENCE_DIR / "final_handoff"
DEFAULT_ULTIMATE_MANIFEST = TASK2_EVIDENCE_DIR / "ultimate_judgement/manifest.json"
SEASON_LABELS = ("Fall", "Spring", "Summer", "Winter")

_AUDIT_COLUMNS = (
    "artifact",
    "path",
    "expected_sha256",
    "actual_sha256",
    "status",
    "role",
)
_REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {
        "ultimate_judgement_manifest",
        "selection_freeze",
        "development_refit_manifest",
        "development_refit_bundle",
        "development_refit_history",
        "development_refit_runtime",
        "canonical_splits",
        "canonical_label_maps",
        "inference_source",
        "registry_binding",
    }
)


def _resolve_within_root(path: str | Path, *, root: Path, scope: str) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{scope} is outside project root: {resolved}") from error
    return resolved


def _declaration(path: Path, *, root: Path) -> dict[str, str]:
    resolved = _resolve_within_root(path, root=root, scope="handoff artifact")
    if not resolved.is_file():
        raise FileNotFoundError(f"handoff artifact does not exist: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": compute_sha256(resolved),
    }


def _declared_path(
    declaration: Mapping[str, Any],
    *,
    root: Path,
    scope: str,
) -> Path:
    if set(declaration) != {"path", "sha256"}:
        raise ValueError(f"{scope} must contain only path and sha256")
    resolved = _resolve_within_root(declaration["path"], root=root, scope=scope)
    verify_artifact(resolved, str(declaration["sha256"]))
    return resolved


def _audit_file(
    name: str,
    path: Path,
    expected_sha256: str,
    role: str,
    *,
    root: Path,
) -> dict[str, str]:
    resolved = _resolve_within_root(path, root=root, scope=name)
    actual = verify_artifact(resolved, expected_sha256)
    return {
        "artifact": name,
        "path": resolved.relative_to(root).as_posix(),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "status": "PASS",
        "role": role,
    }


def audit_task2_artifacts(
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
    ultimate_manifest_path: str | Path = DEFAULT_ULTIMATE_MANIFEST,
    model_manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
) -> pd.DataFrame:
    """Verify the frozen decision, refit package, registry row, and deployed source."""
    root = Path(project_root).resolve()
    ultimate, resolved_ultimate, ultimate_artifacts = (
        load_verified_ultimate_judgement_manifest(
            ultimate_manifest_path,
            project_root=root,
        )
    )
    refit, resolved_refit, _ = load_verified_development_refit_manifest(
        model_manifest_path,
        project_root=root,
        registry_path=registry_path,
    )
    if (
        ultimate["selected_candidate"] != refit["selected_candidate"]
        or ultimate["selected_experiment_id"] != refit["selected_experiment_id"]
        or ultimate["holdout_opened"] is not False
        or ultimate["holdout_metrics_present"] is not False
        or refit["holdout_opened"] is not False
        or refit["holdout_metrics_present"] is not False
        or refit["scratch"] is not True
        or refit["weights"] is not None
        or refit["final_eligible"] is not True
    ):
        raise ValueError("Task 2 decision and development refit boundaries disagree")
    if ultimate["artifacts"]["selection_freeze"] != refit["selection_freeze"]:
        raise ValueError("Task 2 selection freeze changed between G7 and G8")

    declarations = {
        "selection_freeze": refit["selection_freeze"],
        "development_refit_bundle": refit["bundle"],
        "development_refit_history": refit["artifacts"]["history"],
        "development_refit_runtime": refit["artifacts"]["runtime"],
        "canonical_splits": refit["canonical_inputs"]["splits"],
        "canonical_label_maps": refit["canonical_inputs"]["label_maps"],
    }
    roles = {
        "selection_freeze": "immutable pre-holdout model decision",
        "development_refit_bundle": "scratch image-only model weights",
        "development_refit_history": "24-epoch optimisation diagnostics",
        "development_refit_runtime": "training environment and cost provenance",
        "canonical_splits": "only allowed data partition contract",
        "canonical_label_maps": "frozen Season class order",
    }
    rows = [
        _audit_file(
            "ultimate_judgement_manifest",
            resolved_ultimate,
            compute_sha256(resolved_ultimate),
            "verified G7 winner and rejected alternatives",
            root=root,
        ),
        _audit_file(
            "development_refit_manifest",
            resolved_refit,
            compute_sha256(resolved_refit),
            "registry-bound G8 package contract",
            root=root,
        ),
    ]
    for name, declaration in declarations.items():
        path = _declared_path(declaration, root=root, scope=name)
        rows.append(
            _audit_file(
                name,
                path,
                str(declaration["sha256"]),
                roles[name],
                root=root,
            )
        )
    inference_source = root / "src/fashion/task2/inference.py"
    rows.append(
        _audit_file(
            "inference_source",
            inference_source,
            compute_sha256(inference_source),
            "verified image-only prediction interface",
            root=root,
        )
    )
    registry = _resolve_within_root(registry_path, root=root, scope="run registry")
    rows.append(
        {
            "artifact": "registry_binding",
            "path": registry.relative_to(root).as_posix(),
            "expected_sha256": "",
            "actual_sha256": "",
            "status": "PASS",
            "role": f"exact completed row for {refit['run_id']}",
        }
    )
    if ultimate_artifacts["selection_freeze"] != _declared_path(
        refit["selection_freeze"], root=root, scope="selection freeze"
    ):
        raise ValueError("verified G7 and G8 selection-freeze paths disagree")
    audit = pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
    _validate_artifact_audit(audit)
    return audit


def _validate_artifact_audit(audit: pd.DataFrame) -> None:
    if tuple(audit.columns) != _AUDIT_COLUMNS:
        raise ValueError("Task 2 artifact audit columns changed")
    if audit["artifact"].duplicated().any():
        raise ValueError("Task 2 artifact audit contains duplicate rows")
    if frozenset(audit["artifact"]) != _REQUIRED_AUDIT_ARTIFACTS:
        raise ValueError("Task 2 artifact audit is incomplete")
    if not audit["status"].eq("PASS").all():
        raise ValueError("Task 2 artifact audit contains a failed check")


def _validate_smoke_prediction(
    prediction: SeasonPrediction,
    *,
    refit: Mapping[str, Any],
    root: Path,
) -> tuple[str, str]:
    if (
        prediction.run_id != refit["run_id"]
        or prediction.bundle_sha256 != refit["bundle"]["sha256"]
        or tuple(prediction.probabilities) != SEASON_LABELS
        or prediction.predicted_label not in SEASON_LABELS
        or prediction.review_required is not None
        or not math.isclose(
            sum(prediction.probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("Task 2 inference smoke prediction changed its frozen contract")
    image = _resolve_within_root(prediction.image_path, root=root, scope="smoke image")
    relative_image = image.relative_to(root).as_posix()
    splits_path = _declared_path(
        refit["canonical_inputs"]["splits"], root=root, scope="canonical splits"
    )
    splits = pd.read_csv(splits_path, dtype={"id": "string"})
    normalised_paths = splits["path"].astype("string").str.replace("\\", "/", regex=False)
    matches = splits.loc[normalised_paths.eq(relative_image)]
    if len(matches) != 1:
        raise ValueError("Task 2 inference smoke image is not unique in canonical splits")
    row = matches.iloc[0]
    if (
        row["partition"] != "development"
        or str(row["has_season_label"]).strip().lower() != "true"
    ):
        raise ValueError("Task 2 inference smoke image must be labelled development data")
    return str(row["id"]), relative_image


def build_task2_handoff_evidence(
    artifact_audit: pd.DataFrame,
    prediction: SeasonPrediction,
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
    model_manifest_path: str | Path = TASK2_MODEL_MANIFEST_JSON,
    output_directory: str | Path = DEFAULT_HANDOFF_DIRECTORY,
) -> tuple[dict[str, Any], Path]:
    """Write a Task 2 component handoff while keeping Notebook 06 locked."""
    root = Path(project_root).resolve()
    _validate_artifact_audit(artifact_audit)
    refit, resolved_refit, _ = load_verified_development_refit_manifest(
        model_manifest_path,
        project_root=root,
        registry_path=registry_path,
    )
    if prediction.manifest_sha256 != compute_sha256(resolved_refit):
        raise ValueError("Task 2 inference smoke used a different model manifest")
    product_id, relative_image = _validate_smoke_prediction(
        prediction,
        refit=refit,
        root=root,
    )
    output = _resolve_within_root(
        output_directory,
        root=root,
        scope="Task 2 handoff directory",
    )
    audit_path = output / "artifact_audit.csv"
    smoke_path = output / "inference_smoke.json"
    manifest_path = output / "manifest.json"
    atomic_write_csv(audit_path, artifact_audit)
    smoke = {
        "schema_version": "1.0.0",
        "input_scope": "one labelled development image",
        "product_id": product_id,
        "image_path": relative_image,
        "predicted_label": prediction.predicted_label,
        "probabilities": {
            label: round(float(prediction.probabilities[label]), 10)
            for label in SEASON_LABELS
        },
        "probability_order": list(SEASON_LABELS),
        "review_required": None,
        "latency_measured": prediction.latency_ms >= 0,
        "run_id": prediction.run_id,
        "model_manifest_sha256": prediction.manifest_sha256,
        "bundle_sha256": prediction.bundle_sha256,
        "holdout_opened": False,
    }
    atomic_write_json(smoke_path, smoke)
    freeze, freeze_path = load_verified_selection_freeze(
        root / str(refit["selection_freeze"]["path"]),
        project_root=root,
    )
    manifest = {
        "schema_version": "1.0.0",
        "gate": "TASK2-COMPONENT-HANDOFF",
        "status": "ready_for_group_freeze",
        "task2_component_ready": True,
        "group_freeze_verified": False,
        "notebook_06_unlocked": False,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "evaluation_claim_allowed": False,
        "model_change_allowed": False,
        "selected_candidate": refit["selected_candidate"],
        "selected_experiment_id": refit["selected_experiment_id"],
        "run_id": refit["run_id"],
        "labels": list(SEASON_LABELS),
        "inference_inputs": ["image"],
        "review_threshold": None,
        "artifacts": {
            "artifact_audit": _declaration(audit_path, root=root),
            "inference_smoke": _declaration(smoke_path, root=root),
            "selection_freeze": _declaration(freeze_path, root=root),
            "model_manifest": _declaration(resolved_refit, root=root),
            "model_bundle": _declaration(
                root / str(refit["bundle"]["path"]), root=root
            ),
            "inference_source": _declaration(
                root / "src/fashion/task2/inference.py", root=root
            ),
        },
        "unresolved_risks": list(freeze["limitations"]),
        "next_gate": "whole-group freeze before Notebook 06",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest, manifest_path


def load_verified_task2_handoff(
    path: str | Path = DEFAULT_HANDOFF_DIRECTORY / "manifest.json",
    *,
    project_root: str | Path = ROOT,
    registry_path: str | Path = RUNS_CSV,
) -> tuple[dict[str, Any], Path, pd.DataFrame, dict[str, Any]]:
    """Verify the Task 2 handoff and confirm final evaluation remains locked."""
    root = Path(project_root).resolve()
    resolved = _resolve_within_root(path, root=root, scope="Task 2 handoff manifest")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    fixed = {
        "schema_version": "1.0.0",
        "gate": "TASK2-COMPONENT-HANDOFF",
        "status": "ready_for_group_freeze",
        "task2_component_ready": True,
        "group_freeze_verified": False,
        "notebook_06_unlocked": False,
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "evaluation_claim_allowed": False,
        "model_change_allowed": False,
        "labels": list(SEASON_LABELS),
        "inference_inputs": ["image"],
        "review_threshold": None,
        "next_gate": "whole-group freeze before Notebook 06",
    }
    changed = [name for name, expected in fixed.items() if manifest.get(name) != expected]
    if changed:
        raise ValueError(f"Task 2 handoff boundary changed: {changed}")
    required_fields = set(fixed) | {
        "selected_candidate",
        "selected_experiment_id",
        "run_id",
        "artifacts",
        "unresolved_risks",
    }
    if set(manifest) != required_fields:
        raise ValueError("Task 2 handoff fields changed")
    artifacts = manifest["artifacts"]
    required_artifacts = {
        "artifact_audit",
        "inference_smoke",
        "selection_freeze",
        "model_manifest",
        "model_bundle",
        "inference_source",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("Task 2 handoff artifact set changed")
    verified = {
        name: _declared_path(declaration, root=root, scope=f"handoff {name}")
        for name, declaration in artifacts.items()
    }
    audit = pd.read_csv(verified["artifact_audit"], dtype="string").fillna("")
    _validate_artifact_audit(audit)
    with verified["inference_smoke"].open(encoding="utf-8") as handle:
        smoke = json.load(handle)
    refit, resolved_refit, _ = load_verified_development_refit_manifest(
        verified["model_manifest"],
        project_root=root,
        registry_path=registry_path,
    )
    if (
        resolved_refit != verified["model_manifest"]
        or manifest["selected_candidate"] != refit["selected_candidate"]
        or manifest["selected_experiment_id"] != refit["selected_experiment_id"]
        or manifest["run_id"] != refit["run_id"]
        or smoke.get("run_id") != refit["run_id"]
        or smoke.get("model_manifest_sha256") != compute_sha256(resolved_refit)
        or smoke.get("bundle_sha256") != refit["bundle"]["sha256"]
        or smoke.get("holdout_opened") is not False
    ):
        raise ValueError("Task 2 handoff no longer matches the verified refit")
    return manifest, resolved, audit, smoke


__all__ = [
    "DEFAULT_HANDOFF_DIRECTORY",
    "audit_task2_artifacts",
    "build_task2_handoff_evidence",
    "load_verified_task2_handoff",
]
