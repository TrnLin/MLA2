"""Fail-closed loader for completed Task 4 holdout evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fashion.config import ROOT
from fashion.task4_evaluation.blind import (
    EVALUATION_MANIFEST_PATH,
    FINAL_EVALUATION_DIR,
    FINAL_EVALUATION_FIGURE_DIR,
    PREDICTION_RECEIPT_PATH,
    UNLOCK_RECEIPT_PATH,
    _at_project_root,
)
from fashion.task4_evaluation.encoders import R5_METHOD
from fashion.task4_evaluation.spec import load_holdout_evaluation_spec
from fashion.train.artifacts import ArtifactVerificationError

_SCHEMA_VERSION = "1.0.0"
_UNLOCK_ATTEMPT_FILENAME = "unlock_attempt.json"
_SCORING_APPROVAL_REF = "refs/tags/task4-holdout-scoring-approved-v1"
_PREDICTION_FALSE_FLAGS = (
    "labels_opened",
    "teacher_test_scored",
    "model_changed",
    "retuning_allowed",
)
_UNLOCK_FALSE_FLAGS = (
    "model_retrained",
    "winner_changed",
    "metric_changed",
    "conditions_changed",
    "retuning_allowed",
)
_BLIND_ARTIFACT_FILES = {
    "holdout_query_manifest": "holdout_query_manifest.csv",
    "gallery_manifest": "gallery_manifest.csv",
    "holdout_primary_rankings": "holdout_primary_rankings.csv",
    "holdout_family_rankings": "holdout_family_rankings.csv",
    "runtime": "blind_runtime.json",
}
_EVIDENCE_FILES = {
    "holdout_coverage": "holdout_coverage.csv",
    "holdout_scorecard": "holdout_scorecard.csv",
    "holdout_per_query_primary": "holdout_per_query_primary.csv",
    "holdout_per_query_family": "holdout_per_query_family.csv",
    "holdout_metrics": "holdout_metrics.json",
    "holdout_bootstrap_intervals": "holdout_bootstrap_intervals.csv",
    "holdout_selective_retrieval": "holdout_selective_retrieval.csv",
    "holdout_slice_metrics": "holdout_slice_metrics.csv",
    "holdout_robustness_metrics": "holdout_robustness_metrics.csv",
    "holdout_error_routes": "holdout_error_routes.csv",
    "holdout_error_examples": "holdout_error_examples.csv",
    "holdout_source_robustness": "holdout_source_robustness.csv",
    "deployment_summary": "deployment_summary.csv",
    "ultimate_judgement": "ultimate_judgement.json",
}
_FIGURE_FILES = {
    "figure_holdout_scorecard": "holdout_scorecard.png",
    "figure_holdout_bootstrap_intervals": "holdout_bootstrap_intervals.png",
    "figure_holdout_selective_retrieval": "holdout_selective_retrieval.png",
    "figure_holdout_slices_robustness": "holdout_slices_robustness.png",
    "figure_holdout_error_examples": "holdout_error_examples.png",
    "figure_holdout_source_robustness": "holdout_source_robustness.png",
}


def _after_audit_bytes_captured(_label: str, _path: Path) -> None:
    """Private test seam after one immutable read and before JSON parsing."""


def _parse_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return parsed


def _read_json_once(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    _after_audit_bytes_captured(label, path)
    return _parse_json_bytes(payload, label=label)


def _verify_record(
    record: object,
    *,
    root: Path,
    expected_path: Path,
    label: str,
) -> bytes:
    if not isinstance(record, Mapping):
        raise ArtifactVerificationError(f"{label} artifact record is missing")
    path = (root / str(record.get("path") or "")).resolve()
    if path != expected_path.resolve():
        raise ArtifactVerificationError(f"{label} artifact path changed")
    digest = record.get("sha256")
    size = record.get("bytes")
    if not isinstance(digest, str):
        raise ArtifactVerificationError(f"{label} artifact SHA-256 is missing")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactVerificationError(f"{label} artifact cannot be read") from error
    if hashlib.sha256(payload).hexdigest() != digest.lower():
        raise ArtifactVerificationError(f"{label} artifact SHA-256 mismatch")
    if isinstance(size, bool) or not isinstance(size, int) or len(payload) != size:
        raise ArtifactVerificationError(f"{label} artifact byte count changed")
    _after_audit_bytes_captured(label, path)
    return payload


def _read_verified_json(
    record: object,
    *,
    root: Path,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    payload = _verify_record(
        record,
        root=root,
        expected_path=expected_path,
        label=label,
    )
    return _parse_json_bytes(payload, label=label.replace("_", " "))


def _require_false(payload: Mapping[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    wrong = [key for key in keys if payload.get(key) is not False]
    if wrong:
        raise ValueError(f"{label} no-change flags are invalid: {wrong}")


def _verify_git_provenance(
    *,
    prediction: Mapping[str, Any],
    attempt: Mapping[str, Any],
    unlock: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    prediction_git = prediction.get("git")
    blind_source_commit = (
        prediction_git.get("commit") if isinstance(prediction_git, Mapping) else None
    )
    if (
        not isinstance(blind_source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", blind_source_commit) is None
    ):
        raise ValueError("prediction receipt blind source commit is invalid")

    git_state = attempt.get("git")
    if not isinstance(git_state, Mapping) or (
        re.fullmatch(r"[0-9a-f]{40}", str(git_state.get("commit") or "")) is None
        or git_state.get("tracked_files_dirty") is not False
    ):
        raise ValueError("recorded holdout git state is invalid")
    changed_paths = attempt.get("post_blind_changed_paths")
    if (
        not isinstance(changed_paths, list)
        or not all(isinstance(path, str) for path in changed_paths)
        or changed_paths != sorted(set(changed_paths))
    ):
        raise ValueError("recorded post-blind changed paths are invalid")

    for label, payload in (
        ("unlock attempt", attempt),
        ("unlock receipt", unlock),
        ("evaluation manifest", manifest),
    ):
        if (
            payload.get("git") != git_state
            or payload.get("blind_source_commit") != blind_source_commit
            or payload.get("post_blind_changed_paths") != changed_paths
        ):
            raise ValueError(f"{label} git provenance does not agree")
    if attempt.get("git_commit") != git_state.get("commit"):
        raise ValueError("unlock attempt git commit does not agree")
    approval_commit = attempt.get("scoring_approval_commit")
    if (
        attempt.get("scoring_approval_ref") != _SCORING_APPROVAL_REF
        or not isinstance(approval_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", approval_commit) is None
        or approval_commit != git_state.get("commit")
    ):
        raise ValueError("recorded scoring approval is invalid")
    for label, payload in (
        ("unlock receipt", unlock),
        ("evaluation manifest", manifest),
    ):
        if (
            payload.get("scoring_approval_ref") != _SCORING_APPROVAL_REF
            or payload.get("scoring_approval_commit") != approval_commit
        ):
            raise ValueError(f"{label} scoring approval does not agree")


def load_verified_holdout_evaluation(
    *,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Verify the immutable final-evaluation ledger without recomputing metrics."""
    root = Path(project_root).resolve()
    spec = load_holdout_evaluation_spec(project_root=root)
    manifest_path = _at_project_root(EVALUATION_MANIFEST_PATH, root)
    prediction_path = _at_project_root(PREDICTION_RECEIPT_PATH, root)
    attempt_path = _at_project_root(FINAL_EVALUATION_DIR, root) / _UNLOCK_ATTEMPT_FILENAME
    unlock_path = _at_project_root(UNLOCK_RECEIPT_PATH, root)
    manifest = _read_json_once(manifest_path, label="evaluation manifest")
    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Task 4 holdout evaluation manifest is not complete")
    if manifest.get("evaluation_id") != spec.evaluation_id:
        raise ValueError("Task 4 holdout evaluation identity changed")

    model = manifest.get("model")
    if not isinstance(model, Mapping) or (
        model.get("method") != R5_METHOD
        or model.get("run_id") != spec.model_run_id
        or model.get("checkpoint_sha256") != spec.model_checkpoint_sha256
        or model.get("scratch") is not True
    ):
        raise ValueError("Task 4 holdout model identity changed")

    prediction = _read_verified_json(
        manifest.get("prediction_receipt"),
        root=root,
        expected_path=prediction_path,
        label="prediction_receipt",
    )
    blind_artifacts = prediction.get("artifacts")
    if (
        not isinstance(blind_artifacts, Mapping)
        or set(blind_artifacts) != set(_BLIND_ARTIFACT_FILES)
    ):
        raise ValueError("prediction receipt blind artifact ledger is incomplete")
    output_dir = _at_project_root(FINAL_EVALUATION_DIR, root)
    for name, filename in _BLIND_ARTIFACT_FILES.items():
        _verify_record(
            blind_artifacts[name],
            root=root,
            expected_path=output_dir / filename,
            label=name,
        )
    attempt = _read_verified_json(
        manifest.get("unlock_attempt"),
        root=root,
        expected_path=attempt_path,
        label="unlock_attempt",
    )
    unlock = _read_verified_json(
        manifest.get("unlock_receipt"),
        root=root,
        expected_path=unlock_path,
        label="unlock_receipt",
    )
    artifacts = manifest.get("artifacts")
    expected_artifacts = set(_EVIDENCE_FILES) | set(_FIGURE_FILES)
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("Task 4 holdout artifact ledger is incomplete")
    figure_dir = _at_project_root(FINAL_EVALUATION_FIGURE_DIR, root)
    for name, filename in _EVIDENCE_FILES.items():
        _verify_record(
            artifacts[name],
            root=root,
            expected_path=output_dir / filename,
            label=name,
        )
    for name, filename in _FIGURE_FILES.items():
        _verify_record(
            artifacts[name],
            root=root,
            expected_path=figure_dir / filename,
            label=name,
        )

    if (
        prediction.get("schema_version") != _SCHEMA_VERSION
        or prediction.get("evaluation_id") != spec.evaluation_id
    ):
        raise ValueError("prediction receipt identity changed")
    _require_false(prediction, _PREDICTION_FALSE_FLAGS, label="prediction receipt")
    if (
        attempt.get("schema_version") != _SCHEMA_VERSION
        or attempt.get("evaluation_id") != spec.evaluation_id
        or attempt.get("state") != "label_access_started"
        or attempt.get("prediction_receipt") != manifest.get("prediction_receipt")
    ):
        raise ValueError("unlock attempt identity or state changed")
    if (
        unlock.get("schema_version") != _SCHEMA_VERSION
        or unlock.get("evaluation_id") != spec.evaluation_id
        or unlock.get("holdout_opened") is not True
    ):
        raise ValueError("unlock receipt identity or holdout-opened flag changed")
    _require_false(unlock, _UNLOCK_FALSE_FLAGS, label="unlock receipt")
    _require_false(
        manifest.get("no_change_after_unlock", {}),
        _UNLOCK_FALSE_FLAGS,
        label="evaluation manifest",
    )
    unlock_prediction = unlock.get("prediction_receipt")
    manifest_prediction = manifest.get("prediction_receipt")
    if (
        unlock_prediction != manifest_prediction
        or unlock.get("unlock_attempt") != manifest.get("unlock_attempt")
        or manifest.get("receipts_agree") is not True
    ):
        raise ValueError("prediction and unlock receipts do not agree")
    _verify_git_provenance(
        prediction=prediction,
        attempt=attempt,
        unlock=unlock,
        manifest=manifest,
    )
    return manifest


__all__ = ["load_verified_holdout_evaluation"]
