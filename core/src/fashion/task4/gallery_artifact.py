"""Stable, hash-checked development teacher-gallery artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import CV_FOLD_COUNT
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.splits import cv_assignment_digest
from fashion.task4.image_safety import reject_sealed_image_rows
from fashion.task4.models import EMBEDDING_DIM, EMBEDDING_NORM_ATOL
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.protocol import FIXED_VALIDATION_FOLD, build_development_views

GALLERY_ARTIFACT_SCHEMA_VERSION = "1.0.0"
GALLERY_ARTIFACT_TYPE = "task4_teacher_gallery"
ALL_DEVELOPMENT_FOLD: int = -1
_ALLOWED_FOLDS: tuple[int, ...] = (ALL_DEVELOPMENT_FOLD, FIXED_VALIDATION_FOLD)
SOURCE_IDENTITY_FIELDS = (
    "schema_version",
    "scope",
    "run_id",
    "run_kind",
    "method",
    "checkpoint_sha256",
    "config_hash",
    "split_fingerprint",
    "source",
    "image_cache_manifest_sha256",
    "source_fingerprint",
    "source_statistics_sha256",
    "fold",
    "contract",
    "rows",
    "dimension",
)
METADATA_COLUMNS = (
    "id",
    "path",
    "sha256",
    "articleType",
    "baseColour",
    "productDisplayName",
    "duplicate_group",
    "product_family_group",
    "partition",
    "cv_fold",
)

_SOURCE_EXTRA_FIELDS = (
    "feature_cache_identity_sha256",
    "feature_method",
    "transform_seconds",
    "source_bytes",
    "ids_sha256",
    "features_sha256",
)
_ARTIFACT_FILES = ("README.md", "ids.npy", "features.npy", "metadata.csv")
_MANIFEST_FILENAME = "manifest.json"
_CONTRACT = PreprocessingContract(240, 320)
_SAFETY = {
    "development_only": True,
    "holdout_opened": False,
    "official_teacher_test_opened": False,
    "quarantine_opened": False,
}
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "source_feature_cache_identity_sha256",
    "r5_checkpoint",
    "split_fingerprint",
    "fold",
    "source",
    "contract",
    "rows",
    "dimension",
    "ids_dtype",
    "features_dtype",
    "safety",
    "files",
    "artifact_identity_sha256",
}


@dataclass(frozen=True)
class TeacherGallery:
    """One validated, immutable teacher-gallery artifact."""

    directory: Path
    ids: np.ndarray
    features: np.ndarray
    metadata: pd.DataFrame
    manifest: dict[str, Any]
    identity_sha256: str
    _snapshot_owner: tempfile.TemporaryDirectory[str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_json_integer(value: object) -> bool:
    return type(value) is int


def _matches_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or value != _CONTRACT.to_dict():
        return False
    pad_color = value.get("pad_color")
    return (
        _is_json_integer(value.get("width"))
        and _is_json_integer(value.get("height"))
        and isinstance(pad_color, list)
        and len(pad_color) == 3
        and all(_is_json_integer(channel) for channel in pad_color)
    )


def _matches_safety(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(_SAFETY)
        and all(type(value[field]) is bool for field in _SAFETY)
        and value == _SAFETY
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _validate_arrays(
    ids: np.ndarray,
    features: np.ndarray,
    *,
    rows: int,
    label: str,
) -> None:
    valid = (
        ids.dtype == np.int64
        and ids.shape == (rows,)
        and np.array_equal(ids, np.sort(ids))
        and len(np.unique(ids)) == len(ids)
        and features.dtype == np.float32
        and features.shape == (rows, EMBEDDING_DIM)
        and np.isfinite(features).all()
        and np.allclose(
            np.linalg.norm(features, axis=1),
            1.0,
            atol=EMBEDDING_NORM_ATOL,
            rtol=0.0,
        )
    )
    if not valid:
        raise ValueError(
            f"{label} arrays must contain sorted unique int64 IDs and finite, "
            "unit-normalized float32 128-value features"
        )


def _validate_metadata_paths(metadata: pd.DataFrame) -> None:
    for raw_path in metadata["path"].astype(str):
        normalized = raw_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or (path.parts and path.parts[0].endswith(":"))
        ):
            raise ValueError("gallery metadata path must be relative without '..'")


def _load_source_cache(
    source_cache: Path,
    *,
    expected_source_identity_sha256: str,
    expected_checkpoint_sha256: str,
    expected_run_id: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = _read_json(source_cache / _MANIFEST_FILENAME, label="source feature manifest")
    expected_fields = set(SOURCE_IDENTITY_FIELDS) | set(_SOURCE_EXTRA_FIELDS)
    if set(manifest) != expected_fields:
        raise ValueError("source feature manifest schema is malformed")

    identity = {field: manifest[field] for field in SOURCE_IDENTITY_FIELDS}
    computed_identity = hashlib.sha256(_canonical_json(identity)).hexdigest()
    stored_identity = manifest["feature_cache_identity_sha256"]
    if (
        not _is_sha256(expected_source_identity_sha256)
        or not _is_sha256(stored_identity)
        or computed_identity != stored_identity
        or stored_identity != expected_source_identity_sha256
    ):
        raise ValueError("source feature-cache identity does not match the pinned identity")

    # ``identity["fold"]`` is cache provenance: the R5 features were encoded by the
    # fold-1 training session. It stays pinned to FIXED_VALIDATION_FOLD even when the
    # exported gallery spans every development fold, because ALL_DEVELOPMENT_FOLD
    # describes gallery membership, not which session produced these features.
    if (
        identity["schema_version"] != "1.0.0"
        or identity["scope"] != "development"
        or identity["source"] != "teacher"
        or identity["method"] != "R5"
        or not _is_json_integer(identity["fold"])
        or identity["fold"] != FIXED_VALIDATION_FOLD
        or not _is_json_integer(identity["dimension"])
        or identity["dimension"] != EMBEDDING_DIM
        or not _matches_contract(identity["contract"])
        or identity["run_id"] != expected_run_id
        or identity["checkpoint_sha256"] != expected_checkpoint_sha256
        or manifest["feature_method"] != "R5"
    ):
        raise ValueError("source feature-cache identity fields do not match the selected R5 run")
    for sha_field in (
        "checkpoint_sha256",
        "config_hash",
        "split_fingerprint",
        "image_cache_manifest_sha256",
        "source_fingerprint",
        "source_statistics_sha256",
        "ids_sha256",
        "features_sha256",
    ):
        if not _is_sha256(manifest[sha_field]):
            raise ValueError(
                f"source feature manifest {sha_field} is not a valid SHA-256"
            )

    ids_path = source_cache / "ids.npy"
    features_path = source_cache / "features.npy"
    with ids_path.open("rb") as ids_file:
        ids_bytes = ids_file.read()
    with features_path.open("rb") as features_file:
        features_bytes = features_file.read()
    if (
        hashlib.sha256(ids_bytes).hexdigest() != manifest["ids_sha256"]
        or hashlib.sha256(features_bytes).hexdigest() != manifest["features_sha256"]
    ):
        raise ValueError("source feature-cache array SHA-256 does not match")
    try:
        ids = np.load(io.BytesIO(ids_bytes), allow_pickle=False)
        features = np.load(io.BytesIO(features_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"source feature-cache arrays cannot be loaded: {error}") from error
    rows = manifest["rows"]
    if not _is_json_integer(rows) or rows < 1:
        raise ValueError("source feature-cache row count is invalid")
    _validate_arrays(ids, features, rows=rows, label="source feature-cache")
    return manifest, ids, features


def _fold_scope(fold: int) -> str:
    return "all-development" if fold == ALL_DEVELOPMENT_FOLD else f"fold-{fold}"


def _readme(fold: int) -> str:
    return f"""# Task 4 {_fold_scope(fold)} teacher gallery

This folder contains the fixed development-only teacher gallery for Task 4 visual search.
It stores sorted product IDs, unit-normalized R5 features, and safe display metadata.

Load it with `fashion.task4.load_teacher_gallery_artifact`.
"""


def _artifact_manifest(
    staging: Path,
    *,
    source_manifest: Mapping[str, Any],
    rows: int,
    fold: int,
) -> dict[str, Any]:
    files = {
        name: {
            "path": name,
            "sha256": compute_sha256(staging / name),
            "bytes": (staging / name).stat().st_size,
        }
        for name in _ARTIFACT_FILES
    }
    manifest: dict[str, Any] = {
        "schema_version": GALLERY_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": GALLERY_ARTIFACT_TYPE,
        "source_feature_cache_identity_sha256": source_manifest[
            "feature_cache_identity_sha256"
        ],
        "r5_checkpoint": {
            "run_id": source_manifest["run_id"],
            "sha256": source_manifest["checkpoint_sha256"],
        },
        "split_fingerprint": source_manifest["split_fingerprint"],
        "fold": int(fold),
        "source": "teacher",
        "contract": _CONTRACT.to_dict(),
        "rows": rows,
        "dimension": EMBEDDING_DIM,
        "ids_dtype": "int64",
        "features_dtype": "float32",
        "safety": dict(_SAFETY),
        "files": files,
    }
    manifest["artifact_identity_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return manifest


def export_teacher_gallery_artifact(
    source_cache: str | Path,
    destination: str | Path,
    *,
    splits_path: str | Path,
    expected_source_identity_sha256: str,
    expected_checkpoint_sha256: str,
    expected_run_id: str,
    fold: int = FIXED_VALIDATION_FOLD,
) -> Path:
    """Export the selected R5 teacher cache as a fixed fold-1 or all-development gallery."""

    if not _is_json_integer(fold) or fold not in _ALLOWED_FOLDS:
        raise ValueError(f"fold must be one of {_ALLOWED_FOLDS}")
    source_directory = Path(source_cache)
    destination_directory = Path(destination)
    if destination_directory.exists():
        raise ValueError(f"destination already exists: {destination_directory}")
    source_manifest, source_ids, source_features = _load_source_cache(
        source_directory,
        expected_source_identity_sha256=expected_source_identity_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_run_id=expected_run_id,
    )

    splits = load_splits(splits_path)
    if cv_assignment_digest(splits) != source_manifest["split_fingerprint"]:
        raise ValueError("canonical split fingerprint does not match the source feature cache")
    if fold == ALL_DEVELOPMENT_FOLD:
        gallery_frame = splits.loc[splits["partition"].eq("development")]
    else:
        primary, _ = build_development_views(splits, validation_fold=fold)
        gallery_frame = primary.gallery
    development_ids = np.sort(
        splits.loc[splits["partition"].eq("development"), "id"].to_numpy(dtype=np.int64)
    )
    if not np.array_equal(source_ids, development_ids):
        raise ValueError("source feature-cache IDs do not match canonical development IDs")

    gallery_ids = np.sort(gallery_frame["id"].to_numpy(dtype=np.int64))
    source_row_by_id = {
        int(product_id): row for row, product_id in enumerate(source_ids.tolist())
    }
    gallery_rows = np.array(
        [source_row_by_id[int(product_id)] for product_id in gallery_ids],
        dtype=np.int64,
    )
    gallery_features = np.asarray(source_features[gallery_rows], dtype=np.float32)
    _validate_arrays(
        gallery_ids,
        gallery_features,
        rows=len(gallery_ids),
        label="gallery output",
    )

    missing_metadata = set(METADATA_COLUMNS).difference(gallery_frame.columns)
    if missing_metadata:
        raise ValueError(f"gallery metadata is missing columns: {sorted(missing_metadata)}")
    metadata = gallery_frame.loc[:, METADATA_COLUMNS].copy()
    metadata["id"] = pd.to_numeric(metadata["id"], errors="raise").astype(np.int64)
    metadata["cv_fold"] = pd.to_numeric(metadata["cv_fold"], errors="raise").astype(np.int64)
    metadata = metadata.sort_values("id", kind="stable").reset_index(drop=True)
    if metadata["id"].tolist() != gallery_ids.tolist():
        raise ValueError("gallery metadata IDs do not align with selected gallery IDs")
    reject_sealed_image_rows(metadata, require_development=True)
    _validate_metadata_paths(metadata)

    destination_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_directory.name}.",
            dir=destination_directory.parent,
        )
    )
    try:
        (staging / "README.md").write_text(_readme(fold), encoding="utf-8")
        np.save(staging / "ids.npy", gallery_ids, allow_pickle=False)
        np.save(staging / "features.npy", gallery_features, allow_pickle=False)
        write_deterministic_csv(metadata, staging / "metadata.csv", index=False)
        manifest = _artifact_manifest(
            staging,
            source_manifest=source_manifest,
            rows=len(gallery_ids),
            fold=fold,
        )
        (staging / _MANIFEST_FILENAME).write_bytes(_canonical_json(manifest))
        load_teacher_gallery_artifact(staging)
        if destination_directory.exists():
            raise ValueError(f"destination already exists: {destination_directory}")
        staging.rename(destination_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination_directory


def _validate_file_map(directory: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_ARTIFACT_FILES):
        raise ValueError("gallery manifest file map is malformed")
    for filename in _ARTIFACT_FILES:
        record = files[filename]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError("gallery manifest file record is malformed")
        raw_path = record["path"]
        if not isinstance(raw_path, str):
            raise ValueError("gallery manifest file path is invalid")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != filename
            or relative.name in {"", ".", ".."}
        ):
            raise ValueError("gallery manifest file path must be one relative filename")
        file_path = directory / relative.name
        expected_bytes = record["bytes"]
        expected_sha256 = record["sha256"]
        if (
            not _is_json_integer(expected_bytes)
            or expected_bytes < 0
            or not _is_sha256(expected_sha256)
        ):
            raise ValueError("gallery manifest file size or SHA-256 is invalid")
        try:
            actual_bytes = file_path.stat().st_size
            actual_sha256 = compute_sha256(file_path)
        except OSError as error:
            raise ValueError(f"gallery artifact file cannot be read: {error}") from error
        if actual_bytes != expected_bytes:
            raise ValueError(f"gallery artifact byte count does not match for {filename}")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"gallery artifact SHA-256 does not match for {filename}")


def _load_metadata(path: Path) -> pd.DataFrame:
    string_columns = {
        column: "string"
        for column in METADATA_COLUMNS
        if column not in {"id", "cv_fold"}
    }
    try:
        metadata = pd.read_csv(
            path,
            keep_default_na=False,
            dtype={"id": "int64", "cv_fold": "int64", **string_columns},
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"gallery metadata cannot be loaded: {error}") from error
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise ValueError("gallery metadata columns do not match the exact schema")
    return metadata


def _copy_snapshot_file(source: Path, destination: Path) -> None:
    try:
        with source.open("rb") as source_handle, destination.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
    except OSError as error:
        raise ValueError(f"gallery artifact file cannot be copied: {error}") from error


def _load_teacher_gallery_snapshot(
    snapshot_directory: Path,
    *,
    artifact_directory: Path,
    snapshot_owner: tempfile.TemporaryDirectory[str],
) -> TeacherGallery:
    """Validate and map one already-copied private gallery snapshot."""

    manifest = _read_json(
        snapshot_directory / _MANIFEST_FILENAME,
        label="gallery manifest",
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("gallery manifest schema is malformed")
    if (
        manifest["schema_version"] != GALLERY_ARTIFACT_SCHEMA_VERSION
        or manifest["artifact_type"] != GALLERY_ARTIFACT_TYPE
    ):
        raise ValueError("gallery manifest identity is invalid")

    identity = manifest.get("artifact_identity_sha256")
    manifest_without_identity = dict(manifest)
    manifest_without_identity.pop("artifact_identity_sha256")
    if (
        not _is_sha256(identity)
        or hashlib.sha256(_canonical_json(manifest_without_identity)).hexdigest() != identity
    ):
        raise ValueError("gallery artifact identity SHA-256 does not match")

    checkpoint = manifest.get("r5_checkpoint")
    rows = manifest.get("rows")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"run_id", "sha256"}
        or not isinstance(checkpoint["run_id"], str)
        or not checkpoint["run_id"].strip()
        or not _is_sha256(checkpoint["sha256"])
        or not _is_sha256(manifest["source_feature_cache_identity_sha256"])
        or not _is_sha256(manifest["split_fingerprint"])
        or not _is_json_integer(manifest["fold"])
        or manifest["fold"] not in _ALLOWED_FOLDS
        or manifest["source"] != "teacher"
        or not _matches_contract(manifest["contract"])
        or not _is_json_integer(rows)
        or rows < 1
        or not _is_json_integer(manifest["dimension"])
        or manifest["dimension"] != EMBEDDING_DIM
        or manifest["ids_dtype"] != "int64"
        or manifest["features_dtype"] != "float32"
        or not _matches_safety(manifest["safety"])
    ):
        raise ValueError("gallery manifest identity fields are invalid")
    _validate_file_map(snapshot_directory, manifest)

    try:
        ids = np.load(
            snapshot_directory / "ids.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        features = np.load(
            snapshot_directory / "features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"gallery arrays cannot be loaded: {error}") from error
    _validate_arrays(ids, features, rows=rows, label="gallery")

    metadata = _load_metadata(snapshot_directory / "metadata.csv")
    if len(metadata) != rows or metadata["id"].tolist() != ids.tolist():
        raise ValueError("gallery metadata row count or ID order does not match arrays")
    manifest_fold = int(manifest["fold"])
    development_folds = set(range(CV_FOLD_COUNT))
    gallery_folds = development_folds
    if manifest_fold != ALL_DEVELOPMENT_FOLD:
        gallery_folds = development_folds - {manifest_fold}
    observed_folds = {int(fold) for fold in metadata["cv_fold"]}
    if not metadata["partition"].eq("development").all() or not observed_folds <= gallery_folds:
        raise ValueError(
            f"gallery metadata must contain development {_fold_scope(manifest_fold)} gallery rows"
        )
    # A subset check alone cannot tell an all-development gallery apart from a
    # single-fold one, so the sentinel scope also has to prove every fold is present.
    if manifest_fold == ALL_DEVELOPMENT_FOLD and observed_folds != development_folds:
        missing_folds = sorted(development_folds - observed_folds)
        raise ValueError(
            "gallery metadata must span every development fold; "
            f"missing folds {missing_folds}"
        )
    reject_sealed_image_rows(metadata, require_development=True)
    _validate_metadata_paths(metadata)

    ids.flags.writeable = False
    features.flags.writeable = False
    return TeacherGallery(
        directory=artifact_directory,
        ids=ids,
        features=features,
        metadata=metadata,
        manifest=dict(manifest),
        identity_sha256=str(identity),
        _snapshot_owner=snapshot_owner,
    )


def load_teacher_gallery_artifact(directory: str | Path) -> TeacherGallery:
    """Copy, strictly validate, and memory-map a private teacher-gallery snapshot."""

    artifact_directory = Path(directory)
    snapshot_owner = tempfile.TemporaryDirectory(prefix="task4-gallery-snapshot-")
    snapshot_directory = Path(snapshot_owner.name)
    try:
        for filename in (_MANIFEST_FILENAME, *_ARTIFACT_FILES):
            _copy_snapshot_file(
                artifact_directory / filename,
                snapshot_directory / filename,
            )
        return _load_teacher_gallery_snapshot(
            snapshot_directory,
            artifact_directory=artifact_directory,
            snapshot_owner=snapshot_owner,
        )
    except Exception:
        snapshot_owner.cleanup()
        raise


__all__ = (
    "ALL_DEVELOPMENT_FOLD",
    "GALLERY_ARTIFACT_SCHEMA_VERSION",
    "GALLERY_ARTIFACT_TYPE",
    "METADATA_COLUMNS",
    "SOURCE_IDENTITY_FIELDS",
    "TeacherGallery",
    "export_teacher_gallery_artifact",
    "load_teacher_gallery_artifact",
)
