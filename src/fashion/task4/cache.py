"""Versioned, development-only lossless image caches for Task 4."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.images import StreamingStats
from fashion.task4.preprocessing import (
    PreprocessingContract,
    load_preprocessed_image,
)

CACHE_SCHEMA_VERSION = "1.0.0"

__all__ = (
    "CACHE_SCHEMA_VERSION",
    "DevelopmentImageCache",
    "ensure_development_image_cache",
    "fit_cached_fold_rgb_statistics",
    "load_development_image_cache",
)


@dataclass(frozen=True)
class DevelopmentImageCache:
    """Opened memory-mappable cache arrays and validated metadata."""

    cache_dir: Path
    ids: np.ndarray
    images: np.ndarray
    content_bounds: np.ndarray
    manifest: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    if missing := required.difference(frame.columns):
        raise ValueError(f"cache frame is missing columns: {sorted(missing)}")


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _digest_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    path_column: str,
    sha_column: str | None,
) -> pd.DataFrame:
    required = {"id", "partition", path_column}
    if sha_column is not None:
        required.add(sha_column)
    _require_columns(frame, required)
    if frame.empty or not frame["partition"].eq("development").all():
        raise ValueError("image caches require development rows only")

    working = frame.copy()
    numeric_ids = pd.to_numeric(working["id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.mod(1).eq(0).all():
        raise ValueError("cache IDs must be integer-compatible")
    working["id"] = numeric_ids.astype(np.int64)
    if working["id"].duplicated().any():
        raise ValueError("cache IDs must be unique")
    if working[path_column].astype(str).str.strip().eq("").any():
        raise ValueError("cache source paths must not be blank")
    return working.sort_values("id").reset_index(drop=True)


def _source_fingerprint(
    frame: pd.DataFrame,
    *,
    path_column: str,
    sha_column: str | None,
    root: Path,
) -> str:
    lines: list[str] = []
    for row in frame.to_dict("records"):
        path = _resolve_path(row[path_column], root)
        source_sha = str(row[sha_column]) if sha_column is not None else ""
        lines.append(
            f"{int(row['id'])}\t{row[path_column]}\t{source_sha}\t{path.stat().st_size}"
        )
    return _digest_lines(lines)


def _expected_manifest(
    frame: pd.DataFrame,
    *,
    path_column: str,
    sha_column: str | None,
    source: str,
    contract: PreprocessingContract,
    root: Path,
) -> dict[str, Any]:
    ids = frame["id"].astype(np.int64)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scope": "development",
        "source": source,
        "path_column": path_column,
        "sha_column": sha_column,
        "rows": len(frame),
        "array_shape": [len(frame), contract.height, contract.width, 3],
        "array_dtype": "uint8",
        "bounds_shape": [len(frame), 4],
        "bounds_dtype": "int32",
        "id_dtype": "int64",
        "id_sha256": _digest_lines([str(int(value)) for value in ids]),
        "source_fingerprint": _source_fingerprint(
            frame,
            path_column=path_column,
            sha_column=sha_column,
            root=root,
        ),
        "contract": contract.to_dict(),
        "files": {
            "ids": "ids.npy",
            "images": "images.npy",
            "content_bounds": "content_bounds.npy",
        },
    }


def load_development_image_cache(cache_dir: str | Path) -> DevelopmentImageCache:
    """Open and validate an existing Task 4 image cache."""
    directory = Path(cache_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "development":
        raise ValueError("cache manifest scope must be development")
    files = manifest.get("files", {})
    ids = np.load(directory / files["ids"], mmap_mode="r")
    images = np.load(directory / files["images"], mmap_mode="r")
    bounds = np.load(directory / files["content_bounds"], mmap_mode="r")

    if ids.dtype != np.int64 or ids.shape != (int(manifest["rows"]),):
        raise ValueError("cached IDs do not match the manifest")
    if images.dtype != np.uint8 or list(images.shape) != manifest["array_shape"]:
        raise ValueError("cached images do not match the manifest")
    if bounds.dtype != np.int32 or list(bounds.shape) != manifest["bounds_shape"]:
        raise ValueError("cached content bounds do not match the manifest")
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, np.sort(ids)):
        raise ValueError("cached IDs must be sorted and unique")
    if _digest_lines([str(int(value)) for value in ids]) != manifest["id_sha256"]:
        raise ValueError("cached IDs fail their manifest digest")
    return DevelopmentImageCache(
        cache_dir=directory,
        ids=ids,
        images=images,
        content_bounds=bounds,
        manifest=manifest,
    )


def _build_cache(
    cache_dir: Path,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    path_column: str,
    contract: PreprocessingContract,
    root: Path,
) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}-", dir=cache_dir.parent)
    )
    try:
        ids = np.lib.format.open_memmap(
            temporary / "ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(frame),),
        )
        images = np.lib.format.open_memmap(
            temporary / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=tuple(manifest["array_shape"]),
        )
        bounds = np.lib.format.open_memmap(
            temporary / "content_bounds.npy",
            mode="w+",
            dtype=np.int32,
            shape=tuple(manifest["bounds_shape"]),
        )
        for index, row in enumerate(frame.to_dict("records")):
            transformed = load_preprocessed_image(
                _resolve_path(row[path_column], root),
                contract,
            )
            ids[index] = int(row["id"])
            images[index] = transformed.pixels
            bounds[index] = transformed.content_bounds
        ids.flush()
        images.flush()
        bounds.flush()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_directory(temporary, cache_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _replace_directory(temporary: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}-backup-",
                dir=destination.parent,
            )
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup is not None and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


@contextmanager
def _cache_lock(parent: Path, source: str):
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{source}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_development_image_cache(
    frame: pd.DataFrame,
    *,
    path_column: str,
    source: str,
    contract: PreprocessingContract,
    cache_root: str | Path,
    root: str | Path = ROOT,
    sha_column: str,
) -> DevelopmentImageCache:
    """Reuse or build an exact, lossless cache from development rows only."""
    if not re.fullmatch(r"[a-z0-9_-]+", source):
        raise ValueError("source must use lowercase letters, numbers, underscores, or hyphens")
    if not isinstance(sha_column, str) or not sha_column.strip():
        raise ValueError("cache requires a non-empty content-hash column name")
    working = _prepare_frame(
        frame,
        path_column=path_column,
        sha_column=sha_column,
    )
    if working[sha_column].astype(str).str.strip().eq("").any():
        raise ValueError("cache content hashes must not be blank")
    root_path = Path(root)
    manifest = _expected_manifest(
        working,
        path_column=path_column,
        sha_column=sha_column,
        source=source,
        contract=contract,
        root=root_path,
    )
    cache_dir = Path(cache_root) / contract.key / source
    with _cache_lock(cache_dir.parent, source):
        if cache_dir.is_dir():
            try:
                existing = load_development_image_cache(cache_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                existing = None
            if existing is not None and existing.manifest == manifest:
                return existing

        _build_cache(
            cache_dir,
            working,
            manifest,
            path_column=path_column,
            contract=contract,
            root=root_path,
        )
        return load_development_image_cache(cache_dir)


def fit_cached_fold_rgb_statistics(
    cache: DevelopmentImageCache,
    frame: pd.DataFrame,
    *,
    validation_fold: int,
) -> dict[str, object]:
    """Fit current-round RGB statistics from a validated development cache."""
    if (
        isinstance(validation_fold, bool)
        or not isinstance(validation_fold, Integral)
        or validation_fold not in range(5)
    ):
        raise ValueError("validation_fold must be an integer in range(5)")
    _require_columns(frame, {"id", "partition", "cv_fold"})
    if frame.empty or not frame["partition"].eq("development").all():
        raise ValueError("cached RGB statistics require development rows only")

    working = frame.copy()
    numeric_ids = pd.to_numeric(working["id"], errors="coerce")
    numeric_folds = pd.to_numeric(working["cv_fold"], errors="coerce")
    if (
        numeric_ids.isna().any()
        or not numeric_ids.mod(1).eq(0).all()
        or numeric_folds.isna().any()
        or not numeric_folds.mod(1).eq(0).all()
    ):
        raise ValueError("cached RGB IDs and folds must be integer-compatible")
    working["id"] = numeric_ids.astype(np.int64)
    working["cv_fold"] = numeric_folds.astype(int)
    if working["id"].duplicated().any():
        raise ValueError("cached RGB statistics require unique IDs")
    if not working["cv_fold"].isin(range(5)).all():
        raise ValueError("cached RGB statistics require CV folds in range(5)")
    if set(working["id"]) != set(int(value) for value in cache.ids):
        raise ValueError("statistics frame IDs must exactly match the cache")

    folds_by_id = working.set_index("id")["cv_fold"]
    stats = StreamingStats()
    training_ids: list[int] = []
    for index, product_id in enumerate(cache.ids):
        numeric_id = int(product_id)
        if int(folds_by_id.loc[numeric_id]) == int(validation_fold):
            continue
        top, left, bottom, right = (
            int(value) for value in cache.content_bounds[index]
        )
        content = cache.images[index, top:bottom, left:right]
        if content.size == 0:
            raise ValueError(f"cache row {numeric_id} has empty content bounds")
        stats.update(content.astype(np.float32) / 255.0)
        training_ids.append(numeric_id)
    if not training_ids:
        raise ValueError("cached RGB statistics have no training rows")
    return {
        "validation_fold": int(validation_fold),
        "training_rows": len(training_ids),
        "training_id_sha256": _digest_lines([str(value) for value in training_ids]),
        "content_pixels": stats.total_pixels,
        "mean": stats.mean,
        "std": stats.std,
        "source": cache.manifest["source"],
        "source_fingerprint": cache.manifest["source_fingerprint"],
        "contract": cache.manifest["contract"],
    }
