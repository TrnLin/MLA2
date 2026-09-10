"""One-shot scoring of the already-frozen Task 4 holdout rankings."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fashion.config import ROOT, SPLITS_CSV, TEACHER_TRAIN_CSV
from fashion.data.dataset import load_splits_for_final_evaluation
from fashion.git_paths import git_project_prefix
from fashion.task4.analysis import (
    build_query_support,
    mark_failure_slices,
    summarize_failure_slices,
)
from fashion.task4.gallery_artifact import load_teacher_gallery_artifact
from fashion.task4.portable_model import load_r5_inference_package
from fashion.task4.preprocessing import PreprocessingContract, preprocess_image
from fashion.task4.protocol import (
    RetrievalViews,
    compute_relevance_coverage,
    evaluate_family_rankings,
    evaluate_primary_rankings,
)
from fashion.task4_evaluation.blind import (
    EVALUATION_MANIFEST_PATH,
    FINAL_EVALUATION_DIR,
    FINAL_EVALUATION_FIGURE_DIR,
    PREDICTION_RECEIPT_PATH,
    UNLOCK_RECEIPT_PATH,
    _artifact_record,
    _at_project_root,
    _git_state,
    _utc_now,
    _write_full_precision_csv,
)
from fashion.task4_evaluation.encoders import (
    R5_METHOD,
    RANDOM_FLOOR_METHOD,
    R5Encoder,
)
from fashion.task4_evaluation.figures import (
    build_holdout_bootstrap_figure,
    build_holdout_error_examples_figure,
    build_holdout_scorecard_figure,
    build_holdout_selective_retrieval_figure,
    build_holdout_slices_robustness_figure,
    build_holdout_source_robustness_figure,
)
from fashion.task4_evaluation.spec import HoldoutEvaluationSpec, load_holdout_evaluation_spec
from fashion.task4_evaluation.views import build_holdout_views
from fashion.train.artifacts import (
    ArtifactVerificationError,
    atomic_write_json,
)
from fashion.train.ranking_metrics import (
    paired_family_ranking_bootstrap,
    summarise_ranking_bootstrap,
)

_SCHEMA_VERSION = "1.0.0"
_BLIND_PHASE = "blind_prediction_before_holdout_label_access"
_PRIMARY_K = 20
_FAMILY_K = 10
_CONTRACT = PreprocessingContract(width=240, height=320)
_SELECTIVE_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3)
_ERROR_EXAMPLE_COUNT = 6
_FILE_INPUTS = (
    "config",
    "splits",
    "variant_index",
    "gallery_manifest",
    "model_manifest",
    "runs",
)
_BLIND_ARTIFACTS = {
    "holdout_query_manifest",
    "gallery_manifest",
    "holdout_primary_rankings",
    "holdout_family_rankings",
    "runtime",
}
_UNLOCK_ATTEMPT_FILENAME = "unlock_attempt.json"
_SCORING_APPROVAL_REF = "refs/tags/task4-holdout-scoring-approved-v1"
_POST_BLIND_STATE_PATHS = frozenset(
    {
        "src/fashion/task4_evaluation/score.py",
        "src/fashion/task4_evaluation/audit.py",
        "tests/task4_evaluation/test_score.py",
        "docs/decisions/0027-task4-post-blind-git-gate.md",
    }
)


@dataclass(frozen=True, slots=True)
class _VerifiedDeploymentInputs:
    model_manifest: dict[str, Any]
    model_manifest_bytes: bytes
    gallery_manifest: dict[str, Any]
    gallery_manifest_bytes: bytes


@dataclass(frozen=True, slots=True)
class _VerifiedBlindPackage:
    receipt: dict[str, Any]
    prediction_record: dict[str, Any]
    spec: HoldoutEvaluationSpec
    split_bytes: bytes
    query_manifest: pd.DataFrame
    gallery_manifest: pd.DataFrame
    primary_rankings: pd.DataFrame
    family_rankings: pd.DataFrame
    deployment_inputs: _VerifiedDeploymentInputs
    git_state: dict[str, Any]
    blind_source_commit: str
    post_blind_changed_paths: list[str]
    scoring_approval_ref: str
    scoring_approval_commit: str


def _after_verified_bytes_captured(_label: str, _path: Path) -> None:
    """Private test seam after one immutable read and before parsing."""


def _before_unlock_attempt_claim() -> None:
    """Private test seam immediately before the atomic one-shot claim."""


def _runtime_is_isolated() -> bool:
    return bool(sys.flags.isolated)


def _parse_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return parsed


def _read_bytes_once(path: Path, *, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactVerificationError(f"{label} artifact cannot be read: {path}") from error
    return payload


def _record_for_bytes(path: Path, payload: bytes, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ArtifactVerificationError(f"artifact is outside project root: {resolved}") from error
    return {
        "path": portable,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _capture_verified_bytes(
    record: object,
    *,
    root: Path,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping):
        raise ArtifactVerificationError(f"{label} artifact record is missing")
    relative = Path(str(record.get("path") or ""))
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArtifactVerificationError(f"{label} artifact is outside project root") from error
    if expected_path is not None and path != expected_path.resolve():
        raise ArtifactVerificationError(f"{label} artifact path changed")
    digest = record.get("sha256")
    size = record.get("bytes")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ArtifactVerificationError(f"{label} artifact SHA-256 is invalid")
    payload = _read_bytes_once(path, label=label)
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != digest.lower():
        raise ArtifactVerificationError(
            f"{label} artifact SHA-256 mismatch: expected {digest.lower()}, "
            f"got {actual_digest}"
        )
    if isinstance(size, bool) or not isinstance(size, int) or len(payload) != size:
        raise ArtifactVerificationError(f"{label} artifact byte count changed")
    _after_verified_bytes_captured(label, path)
    return path, payload


def _capture_nested_verified_bytes(
    record: object,
    *,
    directory: Path,
    label: str,
    expected_name: str | None = None,
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping):
        raise ArtifactVerificationError(f"{label} artifact record is missing")
    relative = Path(str(record.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts or (
        expected_name is not None and relative != Path(expected_name)
    ):
        raise ArtifactVerificationError(f"{label} artifact path changed")
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ArtifactVerificationError(f"{label} artifact is outside its package") from error
    digest = record.get("sha256")
    size = record.get("bytes")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ArtifactVerificationError(f"{label} artifact SHA-256 is invalid")
    payload = _read_bytes_once(path, label=label)
    if hashlib.sha256(payload).hexdigest() != digest.lower():
        raise ArtifactVerificationError(f"{label} artifact SHA-256 mismatch")
    if isinstance(size, bool) or not isinstance(size, int) or len(payload) != size:
        raise ArtifactVerificationError(f"{label} artifact byte count changed")
    _after_verified_bytes_captured(label, path)
    return path, payload


def _require_false(payload: Mapping[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    wrong = [key for key in keys if payload.get(key) is not False]
    if wrong:
        raise ValueError(f"{label} no-change flags are invalid: {wrong}")


def _load_spec_from_snapshot_bytes(config_bytes: bytes) -> HoldoutEvaluationSpec:
    with tempfile.TemporaryDirectory(prefix="task4-config-snapshot-") as directory:
        snapshot_root = Path(directory)
        config_path = snapshot_root / "configs/task4/holdout_final_evaluation.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes(config_bytes)
        return load_holdout_evaluation_spec(project_root=snapshot_root)


def _load_unlocked_from_snapshot_bytes(
    split_bytes: bytes,
    raw_teacher_bytes: bytes,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="task4-unlock-snapshot-") as directory:
        snapshot_root = Path(directory)
        split_path = snapshot_root / "data/processed/splits.csv"
        raw_path = snapshot_root / "data/raw/teacher/train/styles_train.csv"
        split_path.parent.mkdir(parents=True)
        raw_path.parent.mkdir(parents=True)
        split_path.write_bytes(split_bytes)
        raw_path.write_bytes(raw_teacher_bytes)
        return load_splits_for_final_evaluation(
            split_path,
            raw_teacher_csv=raw_path,
            evaluation_unlocked=True,
        )


def _claim_unlock_attempt(
    path: Path,
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            "holdout scoring is one-way; unlock attempt already claimed"
        ) from error
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("unlock attempt marker write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _record_for_bytes(path, encoded, root=root)


def _verify_prediction_identity(
    receipt: Mapping[str, Any],
    spec: HoldoutEvaluationSpec,
    *,
    root: Path,
) -> None:
    if (
        receipt.get("schema_version") != _SCHEMA_VERSION
        or receipt.get("evaluation_id") != spec.evaluation_id
        or receipt.get("phase") != _BLIND_PHASE
    ):
        raise ValueError("prediction receipt identity or phase changed")
    _require_false(
        receipt,
        ("labels_opened", "teacher_test_scored", "model_changed", "retuning_allowed"),
        label="prediction receipt",
    )
    git = receipt.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("prediction receipt git state is missing")
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or git.get("tracked_files_dirty") is not False
    ):
        raise ValueError("prediction receipt git state is invalid")
    model = receipt.get("model")
    if not isinstance(model, Mapping) or (
        model.get("run_id") != spec.model_run_id
        or model.get("checkpoint_sha256") != spec.model_checkpoint_sha256
        or model.get("development_winner_score") != spec.development_winner_score
        or model.get("scratch") is not True
    ):
        raise ValueError("prediction receipt model identity changed")


def _verify_post_blind_git_state(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> tuple[dict[str, Any], str, list[str], str, str]:
    receipt_git = receipt.get("git")
    source_commit = (
        receipt_git.get("commit") if isinstance(receipt_git, Mapping) else None
    )
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise RuntimeError("blind source commit is invalid")

    current_git = _git_state(root)
    clean_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=no",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    if clean_status.stdout:
        raise RuntimeError(
            "working tree has nonignored changes or untracked files before holdout unlock"
        )

    evidence_commit = str(current_git["commit"])
    approval = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            "--quiet",
            f"{_SCORING_APPROVAL_REF}^{{commit}}",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if approval.returncode != 0:
        raise RuntimeError(f"scoring approval ref is missing: {_SCORING_APPROVAL_REF}")
    approval_commit = approval.stdout.strip()
    if approval_commit != evidence_commit:
        raise RuntimeError(
            f"scoring approval ref does not approve current HEAD: {_SCORING_APPROVAL_REF}"
        )

    valid_commit = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{source_commit}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if valid_commit.returncode != 0:
        raise RuntimeError("blind source commit is invalid")

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, evidence_commit],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        if ancestor.returncode == 1:
            raise RuntimeError("blind source commit is not an ancestor of current HEAD")
        raise RuntimeError("blind source commit ancestry could not be verified")

    changed_result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{source_commit}..{evidence_commit}",
            "--",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    changed_paths = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in changed_result.stdout.split(b"\0")
        if path
    )
    artifacts = receipt.get("artifacts")
    artifact_paths = {
        str(record.get("path"))
        for record in artifacts.values()
        if isinstance(artifacts, Mapping) and isinstance(record, Mapping)
    }
    allowed_paths = {
        *artifact_paths,
        receipt_path.resolve().relative_to(root).as_posix(),
        *_POST_BLIND_STATE_PATHS,
    }
    # Git diff reports checkout-relative paths, while receipts use the model root.
    # Keep outside-core changes visible so they still fail this strict allowlist.
    prefix = git_project_prefix(root)
    allowed_paths = {prefix + path for path in allowed_paths}
    disallowed_paths = sorted(set(changed_paths).difference(allowed_paths))
    if disallowed_paths:
        raise RuntimeError(
            f"post-blind changed paths are not allowed: {disallowed_paths}"
        )
    return (
        current_git,
        source_commit,
        changed_paths,
        _SCORING_APPROVAL_REF,
        approval_commit,
    )


def _expected_primary_groups(
    spec: HoldoutEvaluationSpec,
) -> set[tuple[str, str, str]]:
    groups = {
        (method, "teacher", condition)
        for method in spec.methods
        if method != RANDOM_FLOOR_METHOD
        for condition in spec.conditions
    }
    groups.update(
        (method, "v1", "clean")
        for method in spec.methods
        if method != RANDOM_FLOOR_METHOD
    )
    groups.add((RANDOM_FLOOR_METHOD, "teacher", "clean"))
    return groups


def _expected_family_groups(spec: HoldoutEvaluationSpec) -> set[tuple[str, str]]:
    groups = {
        (method, direction)
        for method in spec.methods
        if method != RANDOM_FLOOR_METHOD
        for direction in spec.directions
    }
    groups.add((RANDOM_FLOOR_METHOD, "teacher"))
    return groups


def _integer_series(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric).all()
        or not numeric.mod(1).eq(0).all()
    ):
        raise ValueError(f"{label} must contain integers")
    return numeric.astype(int)


def _validate_ranking_groups(
    rankings: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    expected_groups: set[tuple[str, ...]],
    query_ids: set[int],
    candidate_ids: set[int],
    k: int,
    label: str,
) -> None:
    required = {
        *group_columns,
        "query_id",
        "candidate_id",
        "distance",
        "rank",
    }
    if missing := sorted(required.difference(rankings.columns)):
        raise ValueError(f"{label} rankings are missing columns: {missing}")
    actual_groups = {
        tuple(str(value) for value in values)
        for values in rankings.loc[:, list(group_columns)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if actual_groups != expected_groups:
        raise ValueError(f"{label} ranking coverage differs from the frozen specification")

    normalized = rankings.copy()
    normalized["query_id"] = _integer_series(normalized["query_id"], label="query_id")
    normalized["candidate_id"] = _integer_series(
        normalized["candidate_id"], label="candidate_id"
    )
    normalized["rank"] = _integer_series(normalized["rank"], label="rank")
    distances = pd.to_numeric(normalized["distance"], errors="coerce")
    if distances.isna().any() or not np.isfinite(distances).all():
        raise ValueError(f"{label} ranking distances must be finite")
    if not set(normalized["candidate_id"]).issubset(candidate_ids):
        raise ValueError(f"{label} rankings contain candidates outside the frozen manifest")

    for group in sorted(expected_groups):
        selected = normalized
        for column, value in zip(group_columns, group, strict=True):
            selected = selected.loc[selected[column].astype(str).eq(value)]
        if set(selected["query_id"]) != query_ids:
            raise ValueError(f"{label} rankings do not cover every query")
        counts = selected.groupby("query_id", sort=False).size()
        unique_candidates = selected.groupby("query_id", sort=False)["candidate_id"].nunique()
        ranks_ok = selected.groupby("query_id", sort=False)["rank"].agg(
            lambda values: np.array_equal(
                np.sort(values.to_numpy(dtype=int)),
                np.arange(1, k + 1),
            )
        )
        if not (counts.eq(k).all() and unique_candidates.eq(k).all() and ranks_ok.all()):
            raise ValueError(
                f"{label} rankings need complete consecutive ranks 1 through {k}"
            )


def _verify_blind_package(
    *,
    root: Path,
    receipt_path: Path,
) -> _VerifiedBlindPackage:
    prediction_bytes = _read_bytes_once(receipt_path, label="prediction_receipt")
    _after_verified_bytes_captured("prediction_receipt", receipt_path)
    receipt = _parse_json_bytes(prediction_bytes, label="prediction receipt")
    prediction_record = _record_for_bytes(receipt_path, prediction_bytes, root=root)
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("prediction receipt inputs are missing")
    if "config" not in inputs:
        raise ValueError("prediction receipt input is missing: config")
    _, config_bytes = _capture_verified_bytes(
        inputs["config"],
        root=root,
        label="config",
        expected_path=root / "configs/task4/holdout_final_evaluation.json",
    )
    spec = _load_spec_from_snapshot_bytes(config_bytes)
    _verify_prediction_identity(receipt, spec, root=root)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _BLIND_ARTIFACTS:
        raise ValueError("prediction receipt artifact ledger is incomplete")
    output_dir = _at_project_root(FINAL_EVALUATION_DIR, root)
    expected_artifact_paths = {
        "holdout_query_manifest": output_dir / "holdout_query_manifest.csv",
        "gallery_manifest": output_dir / "gallery_manifest.csv",
        "holdout_primary_rankings": output_dir / "holdout_primary_rankings.csv",
        "holdout_family_rankings": output_dir / "holdout_family_rankings.csv",
        "runtime": output_dir / "blind_runtime.json",
    }
    captured_artifacts = {
        name: _capture_verified_bytes(
            record,
            root=root,
            label=name,
            expected_path=expected_artifact_paths[name],
        )
        for name, record in artifacts.items()
    }
    expected_input_paths = {
        "config": root / "configs/task4/holdout_final_evaluation.json",
        "splits": _at_project_root(SPLITS_CSV, root),
        "variant_index": root / "data/processed/task4/external_variant_index.csv.gz",
        "gallery_manifest": root / spec.gallery_directory / "manifest.json",
        "model_manifest": root / spec.model_package_directory / "manifest.json",
        "runs": root / "results/runs.csv",
    }
    captured_inputs = {"config": config_bytes}
    for name in _FILE_INPUTS:
        if name not in inputs:
            raise ValueError(f"prediction receipt input is missing: {name}")
        if name != "config":
            _, captured_inputs[name] = _capture_verified_bytes(
                inputs[name],
                root=root,
                label=name,
                expected_path=expected_input_paths[name],
            )

    expected_coverage = {
        "development_gallery_rows": spec.expected_development_rows,
        "holdout_rows": spec.expected_holdout_rows,
        "holdout_family_groups": spec.expected_holdout_family_groups,
        "quarantine_rows_excluded": spec.expected_quarantine_rows,
        "directions": list(spec.directions),
        "conditions": list(spec.conditions),
        "methods": list(spec.methods),
    }
    if receipt.get("coverage") != expected_coverage:
        raise ValueError("prediction receipt coverage differs from the frozen specification")

    query_manifest = pd.read_csv(
        io.BytesIO(captured_artifacts["holdout_query_manifest"][1]),
        keep_default_na=False,
        low_memory=False,
    )
    gallery_manifest = pd.read_csv(
        io.BytesIO(captured_artifacts["gallery_manifest"][1]),
        keep_default_na=False,
        low_memory=False,
    )
    primary = pd.read_csv(
        io.BytesIO(captured_artifacts["holdout_primary_rankings"][1]),
        low_memory=False,
    )
    family = pd.read_csv(
        io.BytesIO(captured_artifacts["holdout_family_rankings"][1]),
        low_memory=False,
    )
    for name, frame in (
        ("holdout_query_manifest", query_manifest),
        ("gallery_manifest", gallery_manifest),
        ("holdout_primary_rankings", primary),
        ("holdout_family_rankings", family),
    ):
        rows = artifacts[name].get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows != len(frame):
            raise ArtifactVerificationError(f"{name} artifact row count changed")

    if {"id", "direction"}.difference(query_manifest.columns):
        raise ValueError("holdout query manifest is missing id or direction")
    query_manifest["id"] = _integer_series(
        query_manifest["id"], label="holdout query manifest IDs"
    )
    if query_manifest.duplicated(["direction", "id"]).any():
        raise ValueError("holdout query manifest contains duplicate direction IDs")
    if set(query_manifest["direction"].astype(str)) != set(spec.directions):
        raise ValueError("holdout query manifest directions changed")
    direction_ids = {
        direction: set(
            query_manifest.loc[
                query_manifest["direction"].astype(str).eq(direction), "id"
            ]
        )
        for direction in spec.directions
    }
    query_ids = direction_ids[spec.directions[0]]
    if (
        len(query_ids) != spec.expected_holdout_rows
        or any(ids != query_ids for ids in direction_ids.values())
    ):
        raise ValueError("holdout query manifest ID coverage changed")

    if "id" not in gallery_manifest:
        raise ValueError("gallery manifest is missing IDs")
    gallery_numeric_ids = _integer_series(gallery_manifest["id"], label="gallery IDs")
    gallery_ids = set(gallery_numeric_ids)
    if (
        len(gallery_manifest) != spec.expected_development_rows
        or gallery_numeric_ids.duplicated().any()
        or len(gallery_ids) != spec.expected_development_rows
    ):
        raise ValueError("gallery manifest row coverage changed")

    _validate_ranking_groups(
        primary,
        group_columns=("method", "direction", "condition"),
        expected_groups=_expected_primary_groups(spec),
        query_ids=query_ids,
        candidate_ids=gallery_ids,
        k=_PRIMARY_K,
        label="primary",
    )
    _validate_ranking_groups(
        family,
        group_columns=("method", "direction"),
        expected_groups=_expected_family_groups(spec),
        query_ids=query_ids,
        candidate_ids=query_ids,
        k=_FAMILY_K,
        label="family",
    )
    model_manifest_bytes = captured_inputs["model_manifest"]
    gallery_artifact_manifest_bytes = captured_inputs["gallery_manifest"]
    deployment_inputs = _VerifiedDeploymentInputs(
        model_manifest=_parse_json_bytes(
            model_manifest_bytes,
            label="captured model manifest",
        ),
        model_manifest_bytes=model_manifest_bytes,
        gallery_manifest=_parse_json_bytes(
            gallery_artifact_manifest_bytes,
            label="captured gallery manifest",
        ),
        gallery_manifest_bytes=gallery_artifact_manifest_bytes,
    )
    (
        current_git,
        blind_source_commit,
        post_blind_changed_paths,
        scoring_approval_ref,
        scoring_approval_commit,
    ) = _verify_post_blind_git_state(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
    )
    return _VerifiedBlindPackage(
        receipt=receipt,
        prediction_record=prediction_record,
        spec=spec,
        split_bytes=captured_inputs["splits"],
        query_manifest=query_manifest,
        gallery_manifest=gallery_manifest,
        primary_rankings=primary,
        family_rankings=family,
        deployment_inputs=deployment_inputs,
        git_state=current_git,
        blind_source_commit=blind_source_commit,
        post_blind_changed_paths=post_blind_changed_paths,
        scoring_approval_ref=scoring_approval_ref,
        scoring_approval_commit=scoring_approval_commit,
    )


def _metric_mean(frame: pd.DataFrame, column: str) -> float:
    value = frame[column].mean()
    return float(value) if pd.notna(value) else np.nan


def _evaluate_frozen_rankings(
    primary_rankings: pd.DataFrame,
    family_rankings: pd.DataFrame,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    spec: HoldoutEvaluationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_queries: list[pd.DataFrame] = []
    primary_summaries: list[pd.DataFrame] = []
    for method, direction, condition in sorted(_expected_primary_groups(spec)):
        selected = primary_rankings.loc[
            primary_rankings["method"].eq(method)
            & primary_rankings["direction"].eq(direction)
            & primary_rankings["condition"].eq(condition)
        ]
        per_query, summary = evaluate_primary_rankings(
            selected,
            primary_views,
            k_values=spec.k_values,
        )
        for position, (column, value) in enumerate(
            (("method", method), ("direction", direction), ("condition", condition))
        ):
            per_query.insert(position, column, value)
            summary.insert(position, column, value)
        primary_queries.append(per_query)
        primary_summaries.append(summary)

    family_queries: list[pd.DataFrame] = []
    family_summaries: list[pd.DataFrame] = []
    for method, direction in sorted(_expected_family_groups(spec)):
        selected = family_rankings.loc[
            family_rankings["method"].eq(method)
            & family_rankings["direction"].eq(direction)
        ]
        per_query, summary = evaluate_family_rankings(selected, family_views, k=10)
        for position, (column, value) in enumerate(
            (("method", method), ("direction", direction))
        ):
            per_query.insert(position, column, value)
            summary.insert(position, column, value)
        family_queries.append(per_query)
        family_summaries.append(summary)
    return (
        pd.concat(primary_queries, ignore_index=True),
        pd.concat(family_queries, ignore_index=True),
        pd.concat(primary_summaries, ignore_index=True),
        pd.concat(family_summaries, ignore_index=True),
    )


def _build_scorecard(
    primary_per_query: pd.DataFrame,
    family_per_query: pd.DataFrame,
    spec: HoldoutEvaluationSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, direction, condition in sorted(_expected_primary_groups(spec)):
        if condition != "clean":
            continue
        primary = primary_per_query.loc[
            primary_per_query["method"].eq(method)
            & primary_per_query["direction"].eq(direction)
            & primary_per_query["condition"].eq("clean")
        ]
        family = family_per_query.loc[
            family_per_query["method"].eq(method)
            & family_per_query["direction"].eq(direction)
        ]
        scorable = primary["ndcg_at_10"].notna()
        class_macro = (
            primary.loc[scorable]
            .groupby("articleType", dropna=False)["ndcg_at_10"]
            .mean()
            .mean()
        )
        is_r5 = method == R5_METHOD
        rows.append(
            {
                "method": method,
                "direction": direction,
                "benchmark_only": method in spec.benchmark_only_methods,
                "ndcg_at_10_query_mean": _metric_mean(primary, "ndcg_at_10"),
                "ndcg_at_10_article_type_macro": (
                    float(class_macro) if pd.notna(class_macro) else np.nan
                ),
                "precision_any_at_10": _metric_mean(primary, "precision_any_at_10"),
                "precision_strict_at_10": _metric_mean(
                    primary, "precision_strict_at_10"
                ),
                "tie_rate_at_10": _metric_mean(primary, "tie_rate_at_10"),
                "recall_at_10": _metric_mean(family, "recall_at_10"),
                "hit_rate_at_10": _metric_mean(family, "hit_rate_at_10"),
                "precision_family_at_10": _metric_mean(family, "precision_at_10"),
                "family_tie_rate_at_10": _metric_mean(family, "tie_rate_at_10"),
                "scored_queries": int(scorable.sum()),
                "excluded_queries": int((~scorable).sum()),
                "development_reference_ndcg_at_10": (
                    spec.development_winner_score if is_r5 else np.nan
                ),
                "development_reference_comparable": False if is_r5 else pd.NA,
                "development_reference_basis": (
                    "development four-pairing same-source average over the "
                    "26,217-row fold-1 galleries; holdout uses one 32,773-row "
                    "teacher gallery"
                    if is_r5
                    else ""
                ),
                "holdout_minus_development_ndcg_at_10": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _source_robustness(
    scorecard: pd.DataFrame,
    spec: HoldoutEvaluationSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in spec.methods:
        selected = scorecard.loc[scorecard["method"].eq(method)].set_index("direction")
        if method == RANDOM_FLOOR_METHOD:
            rows.append(
                {
                    "method": method,
                    "holdout_ratio_definition": "not_applicable",
                    "teacher_to_teacher_ndcg_at_10": float(
                        selected.loc["teacher", "ndcg_at_10_query_mean"]
                    ),
                    "v1_to_teacher_gallery_ndcg_at_10": np.nan,
                    "holdout_source_robustness_ratio": np.nan,
                    "status": (
                        "not_applicable_source_independent_single_frozen_ranking"
                    ),
                    "development_source_robustness_ratio": np.nan,
                    "development_ratio_comparable": pd.NA,
                    "development_ratio_basis": "",
                }
            )
            continue
        denominator = float(selected.loc["teacher", "ndcg_at_10_query_mean"])
        numerator = float(selected.loc["v1", "ndcg_at_10_query_mean"])
        ratio = numerator / denominator if denominator != 0 else np.nan
        is_r5 = method == R5_METHOD
        rows.append(
            {
                "method": method,
                "holdout_ratio_definition": (
                    "v1_to_teacher_gallery / teacher_to_teacher_gallery"
                ),
                "teacher_to_teacher_ndcg_at_10": denominator,
                "v1_to_teacher_gallery_ndcg_at_10": numerator,
                "holdout_source_robustness_ratio": ratio,
                "status": "measured",
                "development_source_robustness_ratio": (
                    spec.development_source_robustness_ratio if is_r5 else np.nan
                ),
                "development_ratio_comparable": False if is_r5 else pd.NA,
                "development_ratio_basis": (
                    "development used four source pairings and two fold-1 galleries; "
                    "holdout uses teacher and V1 queries against one teacher gallery"
                    if is_r5
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_intervals(
    primary_per_query: pd.DataFrame,
    holdout: pd.DataFrame,
    spec: Any,
) -> pd.DataFrame:
    clean = primary_per_query.loc[
        primary_per_query["direction"].eq("teacher")
        & primary_per_query["condition"].eq("clean")
    ].copy()
    query_ids = sorted(
        pd.to_numeric(holdout["id"], errors="raise").astype(int).tolist()
    )
    groups_by_id = holdout.assign(
        id=pd.to_numeric(holdout["id"], errors="raise").astype(int)
    ).set_index("id")["product_family_group"]
    groups = groups_by_id.loc[query_ids].astype(str).to_numpy()
    scores: dict[str, np.ndarray] = {}
    for method in spec.methods:
        selected = clean.loc[clean["method"].eq(method), ["query_id", "ndcg_at_10"]].copy()
        selected["query_id"] = pd.to_numeric(
            selected["query_id"], errors="raise"
        ).astype(int)
        if selected["query_id"].duplicated().any() or set(selected["query_id"]) != set(
            query_ids
        ):
            raise ValueError("bootstrap methods do not share exact clean teacher query IDs")
        scores[method] = (
            selected.set_index("query_id").loc[query_ids, "ndcg_at_10"].to_numpy(dtype=float)
        )
    draws = paired_family_ranking_bootstrap(
        scores,
        groups,
        reference=R5_METHOD,
        replicates=spec.bootstrap_replicates,
        random_seed=spec.bootstrap_seed,
    )
    r5_draws = draws.loc[draws["method"].eq(R5_METHOD)]
    r5 = summarise_ranking_bootstrap(
        r5_draws,
        value_column="mean_score",
        random_seed=spec.bootstrap_seed,
    )
    r5.loc[:, "metric"] = "r5_mean_ndcg_at_10"
    rows = [r5]
    for comparator in spec.methods:
        if comparator == R5_METHOD:
            continue
        comparison = draws.loc[draws["method"].eq(comparator)].copy()
        comparison["r5_minus_comparator"] = -comparison["mean_minus_reference"]
        interval = summarise_ranking_bootstrap(
            comparison,
            value_column="r5_minus_comparator",
            random_seed=spec.bootstrap_seed,
        )
        interval.loc[:, "metric"] = f"r5_minus_{comparator}_ndcg_at_10"
        rows.append(interval)
    return pd.concat(rows, ignore_index=True)


def _selective_retrieval(
    primary_rankings: pd.DataFrame,
    primary_per_query: pd.DataFrame,
    family_per_query: pd.DataFrame,
) -> pd.DataFrame:
    selector = (
        primary_rankings.loc[
            primary_rankings["method"].eq(R5_METHOD)
            & primary_rankings["direction"].eq("teacher")
            & primary_rankings["condition"].eq("clean")
            & primary_rankings["rank"].eq(1),
            ["query_id", "distance"],
        ]
        .rename(columns={"distance": "rank1_distance"})
        .copy()
    )
    primary = primary_per_query.loc[
        primary_per_query["method"].eq(R5_METHOD)
        & primary_per_query["direction"].eq("teacher")
        & primary_per_query["condition"].eq("clean"),
        ["query_id", "ndcg_at_10"],
    ]
    family = family_per_query.loc[
        family_per_query["method"].eq(R5_METHOD)
        & family_per_query["direction"].eq("teacher"),
        ["query_id", "recall_at_10"],
    ]
    joined = selector.merge(primary, on="query_id", validate="one_to_one").merge(
        family, on="query_id", validate="one_to_one"
    )
    joined["_query_id"] = pd.to_numeric(joined["query_id"], errors="raise").astype(int)
    joined = joined.sort_values(
        ["rank1_distance", "_query_id"], kind="mergesort"
    ).reset_index(drop=True)
    total = len(joined)
    rows = []
    for target in _SELECTIVE_COVERAGES:
        retained = min(total, max(1, int(math.ceil(total * target))))
        selected = joined.iloc[:retained]
        rows.append(
            {
                "target_coverage": target,
                "actual_coverage": retained / total,
                "distance_threshold": float(selected["rank1_distance"].iloc[-1]),
                "retained_queries": retained,
                "selective_ndcg_at_10": _metric_mean(selected, "ndcg_at_10"),
                "selective_recall_at_10": _metric_mean(selected, "recall_at_10"),
                "boundary_note": (
                    "retained_queries is authoritative when rank-1 distances tie; "
                    "query ID breaks ties"
                ),
            }
        )
    return pd.DataFrame(rows)


def _failure_slices(
    primary_per_query: pd.DataFrame,
    family_per_query: pd.DataFrame,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
) -> pd.DataFrame:
    membership = mark_failure_slices(build_query_support(primary_views, family_views))
    common = {
        "scope": "holdout",
        "fold": "holdout",
        "size": "240x320",
        "query_source": "teacher",
        "gallery_source": "teacher",
    }
    primary = primary_per_query.loc[
        primary_per_query["method"].eq(R5_METHOD)
        & primary_per_query["direction"].eq("teacher")
        & primary_per_query["condition"].eq("clean")
    ].assign(**common, protocol="primary")
    family = family_per_query.loc[
        family_per_query["method"].eq(R5_METHOD)
        & family_per_query["direction"].eq("teacher")
    ].assign(**common, protocol="family")
    return summarize_failure_slices(pd.concat([primary, family]), membership)


def _top10_sets(rankings: pd.DataFrame) -> pd.Series:
    return (
        rankings.loc[rankings["rank"].le(10)]
        .groupby("query_id", sort=True)["candidate_id"]
        .agg(lambda values: frozenset(int(value) for value in values))
    )


def _robustness_metrics(
    primary_rankings: pd.DataFrame,
    primary_per_query: pd.DataFrame,
    spec: HoldoutEvaluationSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in spec.methods:
        if method == RANDOM_FLOOR_METHOD:
            selected = primary_per_query.loc[
                primary_per_query["method"].eq(method)
                & primary_per_query["direction"].eq("teacher")
                & primary_per_query["condition"].eq("clean")
            ]
            rows.append(
                {
                    "method": method,
                    "condition": "clean",
                    "ndcg_at_10": _metric_mean(selected, "ndcg_at_10"),
                    "delta_vs_clean": 0.0,
                    "mean_top10_overlap_with_clean": 1.0,
                    "tie_rate_at_10": _metric_mean(selected, "tie_rate_at_10"),
                    "status": "clean_only_source_independent_frozen_ranking",
                }
            )
            continue
        clean_rankings = primary_rankings.loc[
            primary_rankings["method"].eq(method)
            & primary_rankings["direction"].eq("teacher")
            & primary_rankings["condition"].eq("clean")
        ]
        clean_sets = _top10_sets(clean_rankings)
        clean_metrics = primary_per_query.loc[
            primary_per_query["method"].eq(method)
            & primary_per_query["direction"].eq("teacher")
            & primary_per_query["condition"].eq("clean")
        ]
        clean_score = _metric_mean(clean_metrics, "ndcg_at_10")
        for condition in spec.conditions:
            condition_rankings = primary_rankings.loc[
                primary_rankings["method"].eq(method)
                & primary_rankings["direction"].eq("teacher")
                & primary_rankings["condition"].eq(condition)
            ]
            condition_sets = _top10_sets(condition_rankings)
            if set(condition_sets.index) != set(clean_sets.index):
                raise ValueError("robustness rankings do not share exact query coverage")
            overlaps = [
                len(clean_sets.loc[query_id] & condition_sets.loc[query_id]) / 10
                for query_id in clean_sets.index
            ]
            selected = primary_per_query.loc[
                primary_per_query["method"].eq(method)
                & primary_per_query["direction"].eq("teacher")
                & primary_per_query["condition"].eq(condition)
            ]
            score = _metric_mean(selected, "ndcg_at_10")
            rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "ndcg_at_10": score,
                    "delta_vs_clean": score - clean_score,
                    "mean_top10_overlap_with_clean": float(np.mean(overlaps)),
                    "tie_rate_at_10": _metric_mean(selected, "tie_rate_at_10"),
                    "status": "measured",
                }
            )
    return pd.DataFrame(rows)


def _error_evidence(
    primary_rankings: pd.DataFrame,
    primary_per_query: pd.DataFrame,
    primary_views: RetrievalViews,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = primary_per_query.loc[
        primary_per_query["method"].eq(R5_METHOD)
        & primary_per_query["direction"].eq("teacher")
        & primary_per_query["condition"].eq("clean"),
        ["query_id", "ndcg_at_10"],
    ].copy()
    top1 = primary_rankings.loc[
        primary_rankings["method"].eq(R5_METHOD)
        & primary_rankings["direction"].eq("teacher")
        & primary_rankings["condition"].eq("clean")
        & primary_rankings["rank"].eq(1),
        ["query_id", "candidate_id", "distance"],
    ]
    queries = primary_views.queries.loc[
        :, ["id", "path", "articleType", "baseColour"]
    ].rename(
        columns={
            "id": "query_id",
            "path": "query_path",
            "articleType": "query_articleType",
            "baseColour": "query_baseColour",
        }
    )
    gallery = primary_views.gallery.loc[
        :, ["id", "path", "articleType", "baseColour"]
    ].rename(
        columns={
            "id": "candidate_id",
            "path": "candidate_path",
            "articleType": "candidate_articleType",
            "baseColour": "candidate_baseColour",
        }
    )
    joined = (
        metrics.merge(top1, on="query_id", validate="one_to_one")
        .merge(queries, on="query_id", validate="one_to_one")
        .merge(gallery, on="candidate_id", validate="many_to_one")
    )
    errors = joined.loc[joined["ndcg_at_10"].isna() | joined["ndcg_at_10"].lt(1.0)]
    routes = (
        errors.groupby(
            ["query_articleType", "candidate_articleType"],
            sort=True,
            dropna=False,
        )
        .agg(count=("query_id", "size"), mean_ndcg_at_10=("ndcg_at_10", "mean"))
        .reset_index()
        .sort_values(
            ["count", "query_articleType", "candidate_articleType"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    examples = (
        joined.assign(
            _query_id=pd.to_numeric(joined["query_id"], errors="raise").astype(int)
        )
        .sort_values(
            ["ndcg_at_10", "_query_id"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
        .head(_ERROR_EXAMPLE_COUNT)
        .drop(columns="_query_id")
        .reset_index(drop=True)
    )
    return routes, examples


def _measure_deployment(
    root: Path,
    spec: HoldoutEvaluationSpec,
    query_manifest: pd.DataFrame,
    inputs: _VerifiedDeploymentInputs,
) -> pd.DataFrame:
    package = root / spec.model_package_directory
    gallery_directory = root / spec.gallery_directory
    weights_record = inputs.model_manifest.get("weights")
    _, weights_bytes = _capture_nested_verified_bytes(
        weights_record,
        directory=package,
        label="deployment_model_weights",
        expected_name="weights.pt",
    )
    gallery_records = inputs.gallery_manifest.get("files")
    expected_gallery_files = {"README.md", "ids.npy", "features.npy", "metadata.csv"}
    if not isinstance(gallery_records, Mapping) or set(gallery_records) != expected_gallery_files:
        raise ArtifactVerificationError("captured gallery manifest file ledger is incomplete")
    gallery_file_bytes = {
        name: _capture_nested_verified_bytes(
            gallery_records[name],
            directory=gallery_directory,
            label=f"deployment_gallery_{name}",
            expected_name=name,
        )[1]
        for name in sorted(expected_gallery_files)
    }
    queries = (
        query_manifest.loc[query_manifest["direction"].eq("teacher")]
        .sort_values("id", kind="mergesort")
        .head(8)
    )
    if queries.empty:
        raise ValueError("deployment measurement has no teacher holdout queries")
    query_rows = list(queries.itertuples(index=False))
    captured_queries: list[tuple[Any, bytes]] = []
    for row in query_rows:
        size = int(row.file_size_bytes)
        _, payload = _capture_verified_bytes(
            {
                "path": str(row.path),
                "sha256": str(row.sha256),
                "bytes": size,
            },
            root=root,
            label=f"deployment_query_{int(row.id)}",
            expected_path=root / str(row.path),
        )
        captured_queries.append((row, payload))

    normalization = inputs.model_manifest.get("normalization")
    statistics = (
        normalization.get("teacher") if isinstance(normalization, Mapping) else None
    )
    if not isinstance(statistics, Mapping):
        raise ValueError("captured model manifest has no teacher normalization statistics")

    def triple(field: str, *, positive: bool) -> tuple[float, float, float]:
        values = statistics.get(field)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"captured model teacher normalization {field} is invalid")
        parsed = tuple(float(value) for value in values)
        if not all(math.isfinite(value) and (not positive or value > 0) for value in parsed):
            raise ValueError(f"captured model teacher normalization {field} is invalid")
        return parsed  # type: ignore[return-value]

    snapshot_owner = tempfile.TemporaryDirectory(prefix="task4-deployment-snapshot-")
    gallery = None
    try:
        snapshot_root = Path(snapshot_owner.name)
        snapshot_package = snapshot_root / "model"
        snapshot_gallery = snapshot_root / "gallery"
        snapshot_package.mkdir()
        snapshot_gallery.mkdir()
        (snapshot_package / "manifest.json").write_bytes(inputs.model_manifest_bytes)
        (snapshot_package / "weights.pt").write_bytes(weights_bytes)
        (snapshot_gallery / "manifest.json").write_bytes(inputs.gallery_manifest_bytes)
        for name, payload in gallery_file_bytes.items():
            (snapshot_gallery / name).write_bytes(payload)
        model = load_r5_inference_package(snapshot_package, device="cpu")
        encoder = R5Encoder(
            model=model,
            direction="teacher",
            mean=triple("mean", positive=False),
            std=triple("std", positive=True),
            device=torch.device("cpu"),
            dimension=int(inputs.model_manifest.get("embedding_dim", 0)),
        )
        gallery = load_teacher_gallery_artifact(snapshot_gallery)

        def run(payload: bytes) -> None:
            with Image.open(io.BytesIO(payload)) as source:
                image = preprocess_image(source, _CONTRACT)
            embedding = encoder.encode(image)
            distances = np.linalg.norm(gallery.features - embedding[None, :], axis=1)
            count = min(10, len(distances))
            nearest = np.argpartition(distances, count - 1)[:count]
            _ = nearest[np.argsort(distances[nearest], kind="mergesort")]

        for _row, payload in captured_queries[:2]:
            run(payload)
        elapsed_ms = []
        for _row, payload in captured_queries:
            started = time.perf_counter_ns()
            run(payload)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        if gallery is not None and gallery._snapshot_owner is not None:
            gallery._snapshot_owner.cleanup()
        snapshot_owner.cleanup()
    return pd.DataFrame(
        [
            {
                "method": R5_METHOD,
                "parameter_count": int(parameter_count),
                "model_package_bytes": (
                    len(inputs.model_manifest_bytes) + len(weights_bytes)
                ),
                "gallery_bytes": (
                    len(inputs.gallery_manifest_bytes)
                    + sum(len(payload) for payload in gallery_file_bytes.values())
                ),
                "timed_queries": len(elapsed_ms),
                "single_query_cpu_p50_ms": float(np.quantile(elapsed_ms, 0.50)),
                "single_query_cpu_p95_ms": float(np.quantile(elapsed_ms, 0.95)),
            }
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    item = getattr(value, "item", None)
    return _json_safe(item()) if callable(item) else value


def _calculate_scored_evidence(
    *,
    root: Path,
    spec: HoldoutEvaluationSpec,
    receipt: dict[str, Any],
    unlock: dict[str, Any],
    query_manifest: pd.DataFrame,
    primary_rankings: pd.DataFrame,
    family_rankings: pd.DataFrame,
    unlocked: pd.DataFrame,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    deployment_inputs: _VerifiedDeploymentInputs,
) -> dict[str, Any]:
    output_dir = _at_project_root(FINAL_EVALUATION_DIR, root)
    figure_dir = _at_project_root(FINAL_EVALUATION_FIGURE_DIR, root)
    holdout = unlocked.loc[unlocked["partition"].eq("holdout")].copy()
    coverage = compute_relevance_coverage(
        primary_views,
        family_views,
        k_values=spec.k_values,
    )
    (
        primary_per_query,
        family_per_query,
        primary_summary,
        family_summary,
    ) = _evaluate_frozen_rankings(
        primary_rankings,
        family_rankings,
        primary_views,
        family_views,
        spec,
    )
    scorecard = _build_scorecard(primary_per_query, family_per_query, spec)
    bootstrap = _bootstrap_intervals(primary_per_query, holdout, spec)
    selective = _selective_retrieval(
        primary_rankings,
        primary_per_query,
        family_per_query,
    )
    slices = _failure_slices(
        primary_per_query,
        family_per_query,
        primary_views,
        family_views,
    )
    robustness = _robustness_metrics(primary_rankings, primary_per_query, spec)
    error_routes, error_examples = _error_evidence(
        primary_rankings,
        primary_per_query,
        primary_views,
    )
    source_robustness = _source_robustness(scorecard, spec)
    deployment = _measure_deployment(root, spec, query_manifest, deployment_inputs)

    csv_paths = {
        "holdout_coverage": output_dir / "holdout_coverage.csv",
        "holdout_scorecard": output_dir / "holdout_scorecard.csv",
        "holdout_per_query_primary": output_dir / "holdout_per_query_primary.csv",
        "holdout_per_query_family": output_dir / "holdout_per_query_family.csv",
        "holdout_bootstrap_intervals": output_dir / "holdout_bootstrap_intervals.csv",
        "holdout_selective_retrieval": output_dir / "holdout_selective_retrieval.csv",
        "holdout_slice_metrics": output_dir / "holdout_slice_metrics.csv",
        "holdout_robustness_metrics": output_dir / "holdout_robustness_metrics.csv",
        "holdout_error_routes": output_dir / "holdout_error_routes.csv",
        "holdout_error_examples": output_dir / "holdout_error_examples.csv",
        "holdout_source_robustness": output_dir / "holdout_source_robustness.csv",
        "deployment_summary": output_dir / "deployment_summary.csv",
    }
    csv_frames = {
        "holdout_coverage": coverage,
        "holdout_scorecard": scorecard,
        "holdout_per_query_primary": primary_per_query,
        "holdout_per_query_family": family_per_query,
        "holdout_bootstrap_intervals": bootstrap,
        "holdout_selective_retrieval": selective,
        "holdout_slice_metrics": slices,
        "holdout_robustness_metrics": robustness,
        "holdout_error_routes": error_routes,
        "holdout_error_examples": error_examples,
        "holdout_source_robustness": source_robustness,
        "deployment_summary": deployment,
    }
    for name, frame in csv_frames.items():
        _write_full_precision_csv(csv_paths[name], frame)

    metrics = _json_safe(
        {
            "schema_version": _SCHEMA_VERSION,
            "evaluation_id": spec.evaluation_id,
            "primary_summary": primary_summary.to_dict(orient="records"),
            "family_summary": family_summary.to_dict(orient="records"),
            "undefined_metrics_policy": (
                "NaN queries are excluded from means and counted; they are never zero-filled"
            ),
        }
    )
    metrics_path = output_dir / "holdout_metrics.json"
    atomic_write_json(metrics_path, metrics)

    figure_paths = {
        "figure_holdout_scorecard": build_holdout_scorecard_figure(
            scorecard, figure_dir / "holdout_scorecard.png"
        ),
        "figure_holdout_bootstrap_intervals": build_holdout_bootstrap_figure(
            bootstrap, figure_dir / "holdout_bootstrap_intervals.png"
        ),
        "figure_holdout_selective_retrieval": (
            build_holdout_selective_retrieval_figure(
                selective, figure_dir / "holdout_selective_retrieval.png"
            )
        ),
        "figure_holdout_slices_robustness": (
            build_holdout_slices_robustness_figure(
                slices,
                robustness,
                figure_dir / "holdout_slices_robustness.png",
            )
        ),
        "figure_holdout_error_examples": build_holdout_error_examples_figure(
            error_examples,
            project_root=root,
            path=figure_dir / "holdout_error_examples.png",
        ),
        "figure_holdout_source_robustness": build_holdout_source_robustness_figure(
            source_robustness,
            figure_dir / "holdout_source_robustness.png",
        ),
    }

    judgement = {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_id": spec.evaluation_id,
        "submitted_model": R5_METHOD,
        "submitted_model_scratch": True,
        "judgement": "R5 remains the submitted scratch model",
        "benchmark_ceiling": "b1_pretrained_resnet18",
        "benchmark_ceiling_role": "benchmark_only",
        "model_retrained_after_unlock": False,
        "winner_changed_after_unlock": False,
        "metric_changed_after_unlock": False,
        "conditions_changed_after_unlock": False,
        "retuning_allowed": False,
        "limitations": [
            (
                "The development reference is not comparable to holdout because it uses "
                "a different four-pairing estimand and fold-1 galleries."
            ),
            "Undefined queries are excluded rather than treated as zero.",
            "The holdout is opened once, so no post-hoc retuning or rerun is allowed.",
        ],
    }
    judgement_path = output_dir / "ultimate_judgement.json"
    atomic_write_json(judgement_path, judgement)

    scored_paths = {
        **csv_paths,
        "holdout_metrics": metrics_path,
        "ultimate_judgement": judgement_path,
        **figure_paths,
    }
    artifacts = {
        name: _artifact_record(
            path,
            root=root,
            rows=len(csv_frames[name]) if name in csv_frames else None,
        )
        for name, path in scored_paths.items()
    }
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_id": spec.evaluation_id,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "git": unlock["git"],
        "blind_source_commit": unlock["blind_source_commit"],
        "post_blind_changed_paths": unlock["post_blind_changed_paths"],
        "scoring_approval_ref": unlock["scoring_approval_ref"],
        "scoring_approval_commit": unlock["scoring_approval_commit"],
        "prediction_receipt": unlock["prediction_receipt"],
        "unlock_attempt": unlock["unlock_attempt"],
        "unlock_receipt": _artifact_record(
            _at_project_root(UNLOCK_RECEIPT_PATH, root),
            root=root,
        ),
        "model": {
            "method": R5_METHOD,
            "run_id": spec.model_run_id,
            "checkpoint_sha256": spec.model_checkpoint_sha256,
            "scratch": True,
        },
        "no_change_after_unlock": {
            "model_retrained": False,
            "winner_changed": False,
            "metric_changed": False,
            "conditions_changed": False,
            "retuning_allowed": False,
        },
        "coverage": {
            "holdout_rows": len(holdout),
            "holdout_family_groups": int(holdout["product_family_group"].nunique()),
            "scorecard_clean_combinations": len(scorecard),
            "scored_artifacts": len(artifacts),
        },
        "receipts_agree": bool(
            receipt["model_changed"] is False
            and receipt["retuning_allowed"] is False
            and unlock["holdout_opened"] is True
        ),
        "artifacts": artifacts,
    }
    atomic_write_json(_at_project_root(EVALUATION_MANIFEST_PATH, root), manifest)
    return manifest


def score_holdout(
    *,
    evaluation_unlocked: bool = False,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    if not _runtime_is_isolated():
        raise RuntimeError("holdout scoring requires isolated Python; run with python -I")
    if not evaluation_unlocked:
        raise ValueError("holdout scoring requires evaluation_unlocked=True")
    root = Path(project_root).resolve()
    output_dir = _at_project_root(FINAL_EVALUATION_DIR, root)
    prediction_path = output_dir / PREDICTION_RECEIPT_PATH.name
    attempt_path = output_dir / _UNLOCK_ATTEMPT_FILENAME
    unlock_path = output_dir / UNLOCK_RECEIPT_PATH.name
    manifest_path = output_dir / EVALUATION_MANIFEST_PATH.name
    if attempt_path.exists() or unlock_path.exists() or manifest_path.exists():
        raise RuntimeError("holdout scoring is one-way and cannot be rerun after label access")
    if not prediction_path.is_file():
        raise FileNotFoundError("prediction_receipt.json must exist before holdout scoring")

    package = _verify_blind_package(root=root, receipt_path=prediction_path)
    spec = package.spec
    attempt_payload = {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_id": spec.evaluation_id,
        "created_at_utc": _utc_now(),
        "state": "label_access_started",
        "git_commit": package.git_state["commit"],
        "git": package.git_state,
        "blind_source_commit": package.blind_source_commit,
        "post_blind_changed_paths": package.post_blind_changed_paths,
        "scoring_approval_ref": package.scoring_approval_ref,
        "scoring_approval_commit": package.scoring_approval_commit,
        "prediction_receipt": package.prediction_record,
    }
    _before_unlock_attempt_claim()
    attempt_record = _claim_unlock_attempt(
        attempt_path,
        attempt_payload,
        root=root,
    )
    raw_teacher_path = _at_project_root(TEACHER_TRAIN_CSV, root)
    raw_teacher_bytes = _read_bytes_once(raw_teacher_path, label="raw_teacher_csv")
    _after_verified_bytes_captured("raw_teacher_csv", raw_teacher_path)
    raw_teacher_record = _record_for_bytes(
        raw_teacher_path,
        raw_teacher_bytes,
        root=root,
    )
    unlocked = _load_unlocked_from_snapshot_bytes(
        package.split_bytes,
        raw_teacher_bytes,
    )
    counts = unlocked["partition"].value_counts()
    actual_counts = (
        int(counts.get("development", 0)),
        int(counts.get("holdout", 0)),
        int(counts.get("quarantine", 0)),
    )
    expected_counts = (
        spec.expected_development_rows,
        spec.expected_holdout_rows,
        spec.expected_quarantine_rows,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"canonical partition counts changed: expected {expected_counts}, "
            f"observed {actual_counts}"
        )
    if unlocked["id"].isna().any() or unlocked["id"].duplicated().any():
        raise ValueError("unlocked split IDs must be globally unique")
    holdout = unlocked.loc[unlocked["partition"].eq("holdout")].copy()
    quarantine = unlocked.loc[unlocked["partition"].eq("quarantine")].copy()
    if set(holdout["id"].astype(int)) & set(quarantine["id"].astype(int)):
        raise ValueError("holdout and quarantine IDs overlap after unlock")
    if holdout["articleType"].astype(str).str.strip().eq("").any():
        raise ValueError("holdout articleType labels are blank after unlock")
    primary_views, family_views = build_holdout_views(unlocked)
    if primary_views.queries["product_family_group"].nunique() != (
        spec.expected_holdout_family_groups
    ):
        raise ValueError("holdout family-group count changed after unlock")

    current_git = _git_state(root)
    unlock = {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_id": spec.evaluation_id,
        "opened_at_utc": _utc_now(),
        "authorization": "explicit_user_instruction_in_current_task",
        "prediction_receipt": package.prediction_record,
        "unlock_attempt": attempt_record,
        "raw_teacher_csv": raw_teacher_record,
        "git": current_git,
        "blind_source_commit": package.blind_source_commit,
        "post_blind_changed_paths": package.post_blind_changed_paths,
        "scoring_approval_ref": package.scoring_approval_ref,
        "scoring_approval_commit": package.scoring_approval_commit,
        "holdout_opened": True,
        "holdout_rows": len(holdout),
        "holdout_article_type_rows": int(
            holdout["articleType"].astype(str).str.strip().ne("").sum()
        ),
        "quarantine_rows_excluded": len(quarantine),
        "model_retrained": False,
        "winner_changed": False,
        "metric_changed": False,
        "conditions_changed": False,
        "retuning_allowed": False,
    }
    atomic_write_json(unlock_path, unlock)
    if current_git != package.git_state:
        raise RuntimeError("tracked source changed during holdout unlock; metrics were not run")
    return _calculate_scored_evidence(
        root=root,
        spec=spec,
        receipt=package.receipt,
        unlock=unlock,
        query_manifest=package.query_manifest,
        primary_rankings=package.primary_rankings,
        family_rankings=package.family_rankings,
        unlocked=unlocked,
        primary_views=primary_views,
        family_views=family_views,
        deployment_inputs=package.deployment_inputs,
    )


__all__ = ["score_holdout"]
