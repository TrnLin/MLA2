"""Freeze Task 4 holdout rankings while protected article types stay sealed."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fashion.config import ROOT, SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.task4.gallery_artifact import load_teacher_gallery_artifact
from fashion.task4.preprocessing import PreprocessingContract, preprocess_image
from fashion.task4.probe import rank_probe_embeddings
from fashion.task4.protocol import RetrievalViews, family_candidate_mask, prepare_rankings
from fashion.task4_evaluation.conditions import apply_query_condition
from fashion.task4_evaluation.encoders import (
    R5_METHOD,
    RANDOM_FLOOR_METHOD,
    MethodEncoder,
    build_method_encoders,
    build_random_floor_rankings,
)
from fashion.task4_evaluation.spec import HoldoutEvaluationSpec, load_holdout_evaluation_spec
from fashion.task4_evaluation.views import build_holdout_views
from fashion.train.artifacts import (
    ArtifactVerificationError,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)

FINAL_EVALUATION_DIR = ROOT / "results/evidence/task4/final_evaluation"
FINAL_EVALUATION_FIGURE_DIR = ROOT / "results/figures/task4/final_evaluation"
PREDICTION_RECEIPT_PATH = FINAL_EVALUATION_DIR / "prediction_receipt.json"
UNLOCK_RECEIPT_PATH = FINAL_EVALUATION_DIR / "unlock_receipt.json"
EVALUATION_MANIFEST_PATH = FINAL_EVALUATION_DIR / "evaluation_manifest.json"

_VARIANT_INDEX_PATH = ROOT / "data/processed/task4/external_variant_index.csv.gz"
_RUNS_PATH = ROOT / "results/runs.csv"
_CONTRACT = PreprocessingContract(width=240, height=320)
_PRIMARY_MAX_K = 20
_FAMILY_MAX_K = 10
_PROTECTED_TARGETS = frozenset({"articleType", "season", "gender", "usage"})
_INPUT_NAMES = (
    "config",
    "splits",
    "variant_index",
    "gallery_manifest",
    "model_manifest",
    "runs",
)
_PUBLISH_ONCE_MESSAGE = (
    "blind prediction is publish-once and requires a fresh destination"
)

EncoderFactory = Callable[..., dict[str, MethodEncoder]]


@dataclass(frozen=True, slots=True)
class _CapturedInput:
    path: Path
    payload: bytes


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _portable(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the project root: {resolved}") from error


def _artifact_record(
    path: str | Path, *, root: Path, rows: int | None = None
) -> dict[str, Any]:
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


def _captured_input_record(
    captured: _CapturedInput, *, root: Path, rows: int | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _portable(captured.path, root),
        "sha256": hashlib.sha256(captured.payload).hexdigest(),
        "bytes": len(captured.payload),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _published_artifact_record(
    source_path: Path,
    published_path: Path,
    *,
    root: Path,
    rows: int | None = None,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"evaluation artifact does not exist: {source_path.resolve()}"
        )
    record: dict[str, Any] = {
        "path": _portable(published_path, root),
        "sha256": compute_sha256(source_path),
        "bytes": source_path.stat().st_size,
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


def _at_project_root(path: Path, root: Path) -> Path:
    return root / path.relative_to(ROOT)


def _ensure_fresh_destination(output_dir: Path) -> None:
    state_paths = (
        output_dir,
        output_dir / PREDICTION_RECEIPT_PATH.name,
        output_dir / UNLOCK_RECEIPT_PATH.name,
        output_dir / EVALUATION_MANIFEST_PATH.name,
    )
    if any(path.exists() for path in state_paths):
        raise RuntimeError(_PUBLISH_ONCE_MESSAGE)


def _acquire_publish_lock(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_fresh_destination(output_dir)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(_PUBLISH_ONCE_MESSAGE) from error
    os.close(descriptor)
    try:
        _ensure_fresh_destination(output_dir)
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path


def _read_verified_image_bytes(path: Path, expected_sha256: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactVerificationError(f"artifact cannot be read: {path}") from error
    actual = hashlib.sha256(payload).hexdigest()
    expected = str(expected_sha256).lower()
    if actual != expected:
        raise ArtifactVerificationError(
            f"artifact SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return payload


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _after_blind_input_bytes_captured(_label: str, _path: Path) -> None:
    """Private test seam after one live input read and before its first parse."""


def _capture_input(path: Path, *, label: str) -> _CapturedInput:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactVerificationError(
            f"{label} blind input cannot be read: {path}"
        ) from error
    _after_blind_input_bytes_captured(label, path)
    return _CapturedInput(path=path, payload=payload)


def _load_spec_from_captured_bytes(payload: bytes) -> HoldoutEvaluationSpec:
    with tempfile.TemporaryDirectory(prefix="task4-blind-config-snapshot-") as directory:
        snapshot_root = Path(directory)
        path = snapshot_root / "configs/task4/holdout_final_evaluation.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        return load_holdout_evaluation_spec(project_root=snapshot_root)


def _capture_blind_inputs(
    root: Path,
) -> tuple[HoldoutEvaluationSpec, dict[str, _CapturedInput]]:
    config_path = root / "configs/task4/holdout_final_evaluation.json"
    config = _capture_input(config_path, label="config")
    spec = _load_spec_from_captured_bytes(config.payload)
    paths = {
        "config": config_path,
        "splits": _at_project_root(SPLITS_CSV, root),
        "variant_index": _at_project_root(_VARIANT_INDEX_PATH, root),
        "gallery_manifest": root / spec.gallery_directory / "manifest.json",
        "model_manifest": root / spec.model_package_directory / "manifest.json",
        "runs": _at_project_root(_RUNS_PATH, root),
    }
    captured = {"config": config}
    for name in _INPUT_NAMES[1:]:
        captured[name] = _capture_input(paths[name], label=name)
    return spec, captured


def _link_snapshot_children(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())


def _materialize_blind_input_snapshot(
    *,
    root: Path,
    spec: HoldoutEvaluationSpec,
    captured: Mapping[str, _CapturedInput],
) -> Path:
    snapshot_root = Path(tempfile.mkdtemp(prefix="task4-blind-input-snapshot-"))
    try:
        for item in captured.values():
            relative = item.path.relative_to(root)
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.payload)

        _link_snapshot_children(
            root / spec.model_package_directory,
            snapshot_root / spec.model_package_directory,
        )
        _link_snapshot_children(
            root / spec.gallery_directory,
            snapshot_root / spec.gallery_directory,
        )
        evidence = root / "results/evidence"
        if evidence.is_dir():
            snapshot_evidence = snapshot_root / "results/evidence"
            snapshot_evidence.parent.mkdir(parents=True, exist_ok=True)
            snapshot_evidence.symlink_to(evidence, target_is_directory=True)
    except Exception:
        shutil.rmtree(snapshot_root)
        raise
    return snapshot_root


def _checked_frozen_facts(
    spec: HoldoutEvaluationSpec,
    *,
    root: Path,
) -> tuple[Path, Path, int]:
    model_manifest_path = root / spec.model_package_directory / "manifest.json"
    model_manifest = _read_json(model_manifest_path, label="R5 model manifest")
    source = model_manifest.get("source_checkpoint")
    if not isinstance(source, Mapping):
        raise ValueError("R5 model manifest source_checkpoint is missing")
    if str(source.get("sha256")) != spec.model_checkpoint_sha256:
        raise ValueError("spec model checkpoint SHA-256 differs from the R5 manifest")
    if float(source.get("score", float("nan"))) != spec.development_winner_score:
        raise ValueError("spec development winner score differs from the R5 manifest")

    runs_path = _at_project_root(_RUNS_PATH, root)
    runs = pd.read_csv(runs_path, keep_default_na=False, low_memory=False)
    required = {"run_id", "split_fingerprint"}
    if missing := sorted(required.difference(runs.columns)):
        raise ValueError(f"run registry is missing columns: {missing}")
    selected = runs.loc[runs["run_id"].astype(str).eq(spec.model_run_id)]
    if len(selected) != 1:
        raise ValueError("run registry must contain exactly one selected R5 row")
    if str(selected.iloc[0]["split_fingerprint"]) != spec.split_fingerprint:
        raise ValueError("spec split fingerprint differs from the selected R5 run")
    return model_manifest_path, runs_path, len(runs)


def _partition_frames(
    splits: pd.DataFrame,
    spec: HoldoutEvaluationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = splits.loc[splits["partition"].eq("development")].copy()
    holdout = splits.loc[splits["partition"].eq("holdout")].copy()
    quarantine = splits.loc[splits["partition"].eq("quarantine")].copy()
    actual = (len(development), len(holdout), len(quarantine))
    expected = (
        spec.expected_development_rows,
        spec.expected_holdout_rows,
        spec.expected_quarantine_rows,
    )
    if actual != expected:
        raise ValueError(
            f"canonical partition counts changed: expected {expected}, observed {actual}"
        )
    if holdout["articleType"].astype(str).str.strip().ne("").any():
        raise RuntimeError("blind prediction phase received visible holdout article types")
    # baseColour deliberately stays visible: it is not a protected target.
    if set(holdout["id"].astype(int)) & set(quarantine["id"].astype(int)):
        raise ValueError("holdout and quarantine IDs overlap")
    return development, holdout, quarantine


def _direction_manifests(
    holdout: pd.DataFrame,
    *,
    root: Path,
    variant_path: Path,
    directions: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int]:
    variants = pd.read_csv(variant_path, keep_default_na=False, low_memory=False)
    if leaked := sorted(_PROTECTED_TARGETS.intersection(variants.columns)):
        raise ValueError(f"V1 variant index contains protected target columns: {leaked}")
    required = {
        "id",
        "external_path",
        "external_mode",
        "external_aspect_ratio",
    }
    if missing := sorted(required.difference(variants.columns)):
        raise ValueError(f"V1 variant index is missing columns: {missing}")
    numeric_ids = pd.to_numeric(variants["id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.mod(1).eq(0).all():
        raise ValueError("V1 variant IDs must be integers")
    variants = variants.assign(id=numeric_ids.astype(int))
    if variants["id"].duplicated().any():
        raise ValueError("V1 variant IDs must be unique")
    holdout_ids = set(holdout["id"].astype(int))
    selected = variants.loc[variants["id"].isin(holdout_ids)].copy()
    if set(selected["id"]) != holdout_ids:
        raise ValueError("V1 variant index does not cover every holdout ID")

    teacher = holdout.loc[
        :,
        [
            "id",
            "path",
            "sha256",
            "product_family_group",
            "baseColour",
            "mode",
            "aspect_ratio",
            "file_size_bytes",
        ],
    ].copy()
    v1 = holdout.loc[:, ["id", "product_family_group", "baseColour"]].merge(
        selected.loc[
            :, ["id", "external_path", "external_mode", "external_aspect_ratio"]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )
    v1 = v1.rename(
        columns={
            "external_path": "path",
            "external_mode": "mode",
            "external_aspect_ratio": "aspect_ratio",
        }
    )
    v1["sha256"] = ""
    v1["file_size_bytes"] = 0

    available = {"teacher": teacher, "v1": v1}
    if set(directions) != set(available):
        raise ValueError(f"unsupported holdout directions: {list(directions)}")
    manifests: dict[str, pd.DataFrame] = {}
    digests: dict[str, str] = {}
    for direction in directions:
        frame = available[direction].sort_values("id", kind="stable").reset_index(drop=True)
        portable_paths: list[str] = []
        hashes: list[str] = []
        sizes: list[int] = []
        for row in frame.itertuples(index=False):
            path = (root / str(row.path)).resolve()
            portable = _portable(path, root)
            digest = str(row.sha256) if direction == "teacher" else compute_sha256(path)
            portable_paths.append(portable)
            hashes.append(digest)
            sizes.append(path.stat().st_size)
        frame["path"] = portable_paths
        frame["sha256"] = hashes
        frame["file_size_bytes"] = sizes
        digests[direction] = _verify_image_manifest(frame, root=root)
        manifests[direction] = frame
    return manifests, digests, len(variants)


def _validate_encoder_registry(
    encoders: dict[str, MethodEncoder],
    spec: HoldoutEvaluationSpec,
) -> None:
    expected = set(spec.methods) - {RANDOM_FLOOR_METHOD}
    if set(encoders) != expected:
        raise ValueError("encoder registry does not cover every embedding method")
    for method, encoder in encoders.items():
        if encoder.name != method or encoder.dimension < 1:
            raise ValueError(f"encoder registry entry is invalid: {method}")


def _encode_frame(
    frame: pd.DataFrame,
    *,
    encoders: Mapping[str, MethodEncoder],
    condition: str,
    direction: str,
    dataset: str,
    device: str,
    root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {"id", "path", "sha256"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"encoding frame is missing columns: {missing}")
    feature_rows: dict[str, list[np.ndarray]] = {method: [] for method in encoders}
    started = time.perf_counter()
    for row in frame.loc[:, ["id", "path", "sha256"]].itertuples(index=False):
        path = (root / str(row.path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"image path is outside project root: {path}") from error
        payload = _read_verified_image_bytes(path, str(row.sha256))
        with Image.open(io.BytesIO(payload)) as source:
            conditioned = apply_query_condition(source, condition)
            image = preprocess_image(conditioned, _CONTRACT)
        for method, encoder in encoders.items():
            vector = np.asarray(encoder.encode(image), dtype=np.float32)
            if (
                vector.shape != (encoder.dimension,)
                or not np.isfinite(vector).all()
                or float(np.linalg.norm(vector)) <= 0.0
            ):
                raise ValueError(f"encoder returned an invalid embedding: {method}")
            feature_rows[method].append(vector)
    elapsed = time.perf_counter() - started
    features = {
        method: np.stack(rows).astype(np.float32, copy=False)
        for method, rows in feature_rows.items()
    }
    runtime = {
        "dataset": dataset,
        "direction": direction,
        "condition": condition,
        "methods": list(encoders),
        "device": device,
        "rows": len(frame),
        "runtime_seconds": elapsed,
        "images_per_second": len(frame) / elapsed,
        "prediction_failures": 0,
    }
    return features, runtime


def _method_rankings(
    *,
    method: str,
    direction: str,
    condition: str,
    query_ids: np.ndarray,
    query_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_features: np.ndarray,
    views: RetrievalViews,
    protocol: str,
    max_k: int,
) -> pd.DataFrame:
    ranked = rank_probe_embeddings(
        query_ids=query_ids,
        query_features=query_features,
        gallery_ids=gallery_ids,
        gallery_features=gallery_features,
        views=views,
        protocol=protocol,
        max_k=max_k,
    )
    prepared = prepare_rankings(ranked, views, protocol, max_k=max_k)
    prepared.insert(0, "method", method)
    prepared.insert(1, "direction", direction)
    if protocol == "primary":
        prepared.insert(2, "condition", condition)
    return prepared


def _random_family_rankings(
    query_ids: np.ndarray,
    views: RetrievalViews,
    *,
    seed: int,
    max_k: int,
) -> pd.DataFrame:
    queries = views.queries.set_index("id", drop=False)
    gallery = views.gallery.set_index("id", drop=False)
    ineligible_after_self = []
    for query_id in query_ids:
        eligible = family_candidate_mask(queries.loc[int(query_id)], gallery)
        ineligible_after_self.append(int((~eligible).sum()) - 1)
    draw_k = max_k + max(ineligible_after_self, default=0)
    rankings = build_random_floor_rankings(
        query_ids,
        query_ids,
        seed=seed,
        max_k=draw_k,
    )
    return prepare_rankings(rankings, views, "family", max_k=max_k)


def _build_blind_holdout_evidence_from_snapshot(
    *,
    device: str,
    root: Path,
    snapshot_root: Path,
    output_dir: Path,
    encoder_factory: EncoderFactory,
    start_git_state: dict[str, Any],
    spec: HoldoutEvaluationSpec,
    captured: Mapping[str, _CapturedInput],
) -> dict[str, Any]:
    splits_path = _at_project_root(SPLITS_CSV, snapshot_root)
    splits = load_splits(splits_path)
    _, holdout, quarantine = _partition_frames(splits, spec)
    primary_views, family_views = build_holdout_views(splits)
    if spec.stressed_direction != "teacher":
        raise ValueError("the frozen stressed direction must be teacher")
    if primary_views.queries["product_family_group"].nunique() != (
        spec.expected_holdout_family_groups
    ):
        raise ValueError("holdout family-group count differs from the frozen spec")

    _, _, runs_rows = _checked_frozen_facts(spec, root=snapshot_root)
    gallery = load_teacher_gallery_artifact(snapshot_root / spec.gallery_directory)
    gallery_ids = np.asarray(gallery.ids, dtype=np.int64)
    primary_gallery_ids = primary_views.gallery["id"].to_numpy(dtype=np.int64)
    if len(gallery_ids) != spec.expected_development_rows or set(gallery_ids) != set(
        primary_gallery_ids
    ):
        raise ValueError("gallery artifact does not match the full development gallery")
    checkpoint = gallery.manifest.get("r5_checkpoint", {})
    if (
        checkpoint.get("run_id") != spec.model_run_id
        or checkpoint.get("sha256") != spec.model_checkpoint_sha256
        or gallery.manifest.get("split_fingerprint") != spec.split_fingerprint
    ):
        raise ValueError("gallery artifact provenance differs from the frozen spec")
    gallery_images = gallery.metadata.copy()
    gallery_images["path"] = [
        _portable(root / str(path), root) for path in gallery_images["path"]
    ]
    gallery_image_hash = _verify_image_manifest(gallery_images, root=root)

    query_manifests, query_hashes, variant_rows = _direction_manifests(
        holdout,
        root=root,
        variant_path=_at_project_root(_VARIANT_INDEX_PATH, snapshot_root),
        directions=spec.directions,
    )
    encoders_by_direction: dict[str, dict[str, MethodEncoder]] = {}
    for direction in spec.directions:
        encoders = encoder_factory(
            spec,
            project_root=snapshot_root,
            direction=direction,
            device=device,
        )
        _validate_encoder_registry(encoders, spec)
        encoders_by_direction[direction] = encoders

    runtime_rows: list[dict[str, Any]] = []
    teacher_encoders = encoders_by_direction["teacher"]
    non_r5_encoders = {
        method: encoder
        for method, encoder in teacher_encoders.items()
        if method != R5_METHOD
    }
    encoded_gallery, gallery_runtime = _encode_frame(
        gallery_images,
        encoders=non_r5_encoders,
        condition="clean",
        direction="teacher",
        dataset="development_gallery",
        device=device,
        root=root,
    )
    runtime_rows.append(gallery_runtime)
    gallery_features = {R5_METHOD: np.asarray(gallery.features, dtype=np.float32)}
    gallery_features.update(encoded_gallery)

    query_ids = primary_views.queries["id"].to_numpy(dtype=np.int64)
    primary_parts: list[pd.DataFrame] = []
    family_parts: list[pd.DataFrame] = []
    for direction in spec.directions:
        conditions = spec.conditions if direction == "teacher" else ("clean",)
        clean_features: dict[str, np.ndarray] | None = None
        for condition in conditions:
            query_features, runtime = _encode_frame(
                query_manifests[direction],
                encoders=encoders_by_direction[direction],
                condition=condition,
                direction=direction,
                dataset="holdout_queries",
                device=device,
                root=root,
            )
            runtime_rows.append(runtime)
            if condition == "clean":
                clean_features = query_features
            for method in spec.methods:
                if method == RANDOM_FLOOR_METHOD:
                    continue
                primary_parts.append(
                    _method_rankings(
                        method=method,
                        direction=direction,
                        condition=condition,
                        query_ids=query_ids,
                        query_features=query_features[method],
                        gallery_ids=gallery_ids,
                        gallery_features=gallery_features[method],
                        views=primary_views,
                        protocol="primary",
                        max_k=_PRIMARY_MAX_K,
                    )
                )
        if clean_features is None:
            raise RuntimeError(f"direction {direction} did not execute the clean condition")
        for method in spec.methods:
            if method == RANDOM_FLOOR_METHOD:
                continue
            family_parts.append(
                _method_rankings(
                    method=method,
                    direction=direction,
                    condition="clean",
                    query_ids=query_ids,
                    query_features=clean_features[method],
                    gallery_ids=query_ids,
                    gallery_features=clean_features[method],
                    views=family_views,
                    protocol="family",
                    max_k=_FAMILY_MAX_K,
                )
            )

    random_primary = build_random_floor_rankings(
        query_ids,
        gallery_ids,
        seed=spec.random_floor_seed,
        max_k=_PRIMARY_MAX_K,
    )
    random_primary = prepare_rankings(
        random_primary,
        primary_views,
        "primary",
        max_k=_PRIMARY_MAX_K,
    )
    random_primary.insert(0, "method", RANDOM_FLOOR_METHOD)
    random_primary.insert(1, "direction", "teacher")
    random_primary.insert(2, "condition", "clean")
    primary_parts.append(random_primary)

    random_family = _random_family_rankings(
        query_ids,
        family_views,
        seed=spec.random_floor_seed,
        max_k=_FAMILY_MAX_K,
    )
    random_family.insert(0, "method", RANDOM_FLOOR_METHOD)
    random_family.insert(1, "direction", "teacher")
    family_parts.append(random_family)

    primary_rankings = pd.concat(primary_parts, ignore_index=True).loc[
        :,
        [
            "method",
            "direction",
            "condition",
            "query_id",
            "candidate_id",
            "distance",
            "rank",
        ],
    ]
    family_rankings = pd.concat(family_parts, ignore_index=True).loc[
        :,
        ["method", "direction", "query_id", "candidate_id", "distance", "rank"],
    ]

    query_manifest_parts = []
    for direction in spec.directions:
        frame = query_manifests[direction].copy()
        frame.insert(1, "direction", direction)
        frame.insert(5, "articleType_redacted", "")
        query_manifest_parts.append(frame)
    query_manifest = pd.concat(query_manifest_parts, ignore_index=True).loc[
        :,
        [
            "id",
            "direction",
            "path",
            "sha256",
            "product_family_group",
            "articleType_redacted",
            "baseColour",
            "mode",
            "aspect_ratio",
            "file_size_bytes",
        ],
    ]
    gallery_manifest = gallery_images.loc[:, ["id", "path", "sha256", "cv_fold"]].copy()
    runtime_payload = {
        "schema_version": "1.0.0",
        "phase": "blind_prediction_before_holdout_label_access",
        "environment": _runtime_environment(),
        "runs": runtime_rows,
    }

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        staged_paths = {
            "holdout_query_manifest": staging_dir / "holdout_query_manifest.csv",
            "gallery_manifest": staging_dir / "gallery_manifest.csv",
            "holdout_primary_rankings": staging_dir / "holdout_primary_rankings.csv",
            "holdout_family_rankings": staging_dir / "holdout_family_rankings.csv",
            "runtime": staging_dir / "blind_runtime.json",
        }
        _write_full_precision_csv(
            staged_paths["holdout_query_manifest"], query_manifest
        )
        _write_full_precision_csv(staged_paths["gallery_manifest"], gallery_manifest)
        _write_full_precision_csv(
            staged_paths["holdout_primary_rankings"], primary_rankings
        )
        _write_full_precision_csv(
            staged_paths["holdout_family_rankings"], family_rankings
        )
        atomic_write_json(staged_paths["runtime"], runtime_payload)

        git_state = _git_state(root)
        if git_state["tracked_files_dirty"] or git_state["commit"] != (
            start_git_state["commit"]
        ):
            raise RuntimeError(
                "tracked source changed during blind prediction; receipt was not written"
            )
        row_counts = {
            "holdout_query_manifest": len(query_manifest),
            "gallery_manifest": len(gallery_manifest),
            "holdout_primary_rankings": len(primary_rankings),
            "holdout_family_rankings": len(family_rankings),
        }
        artifacts = {
            name: _published_artifact_record(
                staged_path,
                output_dir / staged_path.name,
                root=root,
                rows=row_counts.get(name),
            )
            for name, staged_path in staged_paths.items()
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
                "run_id": spec.model_run_id,
                "checkpoint_sha256": spec.model_checkpoint_sha256,
                "development_winner_score": spec.development_winner_score,
                "scratch": True,
            },
            "inputs": {
                "config": _captured_input_record(captured["config"], root=root),
                "splits": _captured_input_record(
                    captured["splits"], root=root, rows=len(splits)
                ),
                "variant_index": _captured_input_record(
                    captured["variant_index"], root=root, rows=variant_rows
                ),
                "gallery_manifest": _captured_input_record(
                    captured["gallery_manifest"],
                    root=root,
                ),
                "model_manifest": _captured_input_record(
                    captured["model_manifest"], root=root
                ),
                "runs": _captured_input_record(
                    captured["runs"], root=root, rows=runs_rows
                ),
                "query_image_set_sha256": query_hashes,
                "gallery_image_set_sha256": gallery_image_hash,
                "gallery_identity_sha256": gallery.identity_sha256,
            },
            "coverage": {
                "development_gallery_rows": len(gallery_ids),
                "holdout_rows": len(holdout),
                "holdout_family_groups": int(
                    primary_views.queries["product_family_group"].nunique()
                ),
                "quarantine_rows_excluded": len(quarantine),
                "directions": list(spec.directions),
                "conditions": list(spec.conditions),
                "methods": list(spec.methods),
            },
            "artifacts": artifacts,
        }
        atomic_write_json(staging_dir / PREDICTION_RECEIPT_PATH.name, receipt)
        _ensure_fresh_destination(output_dir)
        try:
            staging_dir.rename(output_dir)
        except OSError as error:
            if output_dir.exists():
                raise RuntimeError(_PUBLISH_ONCE_MESSAGE) from error
            raise
        return receipt
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _build_blind_holdout_evidence(
    *,
    device: str,
    root: Path,
    output_dir: Path,
    encoder_factory: EncoderFactory,
    start_git_state: dict[str, Any],
) -> dict[str, Any]:
    spec, captured = _capture_blind_inputs(root)
    snapshot_root = _materialize_blind_input_snapshot(
        root=root,
        spec=spec,
        captured=captured,
    )
    try:
        return _build_blind_holdout_evidence_from_snapshot(
            device=device,
            root=root,
            snapshot_root=snapshot_root,
            output_dir=output_dir,
            encoder_factory=encoder_factory,
            start_git_state=start_git_state,
            spec=spec,
            captured=captured,
        )
    finally:
        shutil.rmtree(snapshot_root)


def build_blind_holdout_evidence(
    *,
    device: str = "cpu",
    project_root: str | Path = ROOT,
    encoder_factory: EncoderFactory = build_method_encoders,
) -> dict[str, Any]:
    """Freeze every holdout ranking before protected article types are opened."""
    root = Path(project_root).resolve()
    output_dir = _at_project_root(FINAL_EVALUATION_DIR, root)
    lock_path = _acquire_publish_lock(output_dir)
    try:
        git_state = _git_state(root)
        if git_state["tracked_files_dirty"]:
            raise RuntimeError(
                "tracked source changed during blind prediction; receipt was not written"
            )
        return _build_blind_holdout_evidence(
            device=device,
            root=root,
            output_dir=output_dir,
            encoder_factory=encoder_factory,
            start_git_state=git_state,
        )
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = [
    "EVALUATION_MANIFEST_PATH",
    "FINAL_EVALUATION_DIR",
    "FINAL_EVALUATION_FIGURE_DIR",
    "PREDICTION_RECEIPT_PATH",
    "UNLOCK_RECEIPT_PATH",
    "build_blind_holdout_evidence",
]
