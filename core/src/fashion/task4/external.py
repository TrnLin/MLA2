"""Read-only audit helpers for the optional high-resolution V1 image variant."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import RANDOM_SEED, ROOT
from fashion.data.audit import IMAGE_AUDIT_COLUMNS, audit_image
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.perceptual import (
    compute_image_hashes,
    compute_pair_pixel_metrics,
    hamming_distance,
)

EXPECTED_EXTERNAL_CSV_SHA256 = "64dfd2449f22e39120e2ab4b0230a4521f27a3b3513e5eee5cc000ad865df831"
EXTERNAL_CATALOGUE_COLUMNS = ("filename", "link")
SAFE_SPLIT_COLUMNS = (
    "id",
    "path",
    "width",
    "height",
    "file_size_bytes",
    "partition",
    "cv_fold",
    "product_family_group",
    "duplicate_group",
)

__all__ = (
    "EXPECTED_EXTERNAL_CSV_SHA256",
    "EXTERNAL_CATALOGUE_COLUMNS",
    "SAFE_SPLIT_COLUMNS",
    "audit_external_images",
    "build_external_variant_index",
    "compare_variant_pairs",
    "ensure_external_image_audit",
    "inventory_external_images",
    "inventory_fingerprint",
    "read_external_catalogue",
    "reconcile_external_ids",
    "select_development_pairs",
)


def _read_ids(path: str | Path) -> set[int]:
    frame = pd.read_csv(path, usecols=["id"], keep_default_na=False)
    ids = pd.to_numeric(frame["id"], errors="raise").astype(int)
    if ids.duplicated().any():
        raise ValueError(f"IDs must be unique in {path}")
    return set(ids)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_external_catalogue(path: str | Path) -> pd.DataFrame:
    """Load the label-free V1 catalogue and derive one numeric ID per filename."""
    source = Path(path)
    frame = pd.read_csv(source, keep_default_na=False)
    if tuple(frame.columns) != EXTERNAL_CATALOGUE_COLUMNS:
        raise ValueError(
            f"external catalogue columns must be {EXTERNAL_CATALOGUE_COLUMNS}, "
            f"found {tuple(frame.columns)}"
        )
    filenames = frame["filename"].astype(str).str.strip()
    if filenames.eq("").any():
        raise ValueError("external catalogue contains blank filenames")
    if filenames.map(lambda value: Path(value).name != value).any():
        raise ValueError("external catalogue filenames must not contain directories")
    suffixes = filenames.map(lambda value: Path(value).suffix.lower())
    if not suffixes.eq(".jpg").all():
        raise ValueError("external catalogue must contain only .jpg filenames")
    stems = filenames.map(lambda value: Path(value).stem)
    if not stems.str.fullmatch(r"\d+").all():
        raise ValueError("external catalogue filenames must use numeric stems")

    result = frame.copy()
    result["filename"] = filenames
    result.insert(0, "id", stems.astype(int))
    if result["id"].duplicated().any():
        duplicate_ids = sorted(result.loc[result["id"].duplicated(keep=False), "id"].unique())
        raise ValueError(f"external catalogue IDs must be unique: {duplicate_ids[:10]}")
    return result.sort_values("id").reset_index(drop=True)


def inventory_external_images(
    image_dir: str | Path,
    *,
    root: str | Path = ROOT,
) -> pd.DataFrame:
    """List V1 files without opening image pixels."""
    directory = Path(image_dir)
    root_path = Path(root)
    records: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in directory.iterdir() if candidate.is_file()):
        numeric = path.stem.isdigit()
        records.append(
            {
                "id": int(path.stem) if numeric else pd.NA,
                "filename": path.name,
                "path": _relative(path, root_path),
                "suffix": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "numeric_stem": numeric,
            }
        )
    frame = pd.DataFrame(
        records,
        columns=("id", "filename", "path", "suffix", "file_size_bytes", "numeric_stem"),
    )
    if frame.empty:
        return frame
    numeric_ids = frame.loc[frame["numeric_stem"], "id"].astype(int)
    if numeric_ids.duplicated().any():
        duplicate_ids = sorted(numeric_ids[numeric_ids.duplicated(keep=False)].unique())
        raise ValueError(f"external image IDs must be unique: {duplicate_ids[:10]}")
    frame["id"] = frame["id"].astype("Int64")
    return frame.sort_values(["numeric_stem", "id", "filename"], na_position="last").reset_index(
        drop=True
    )


def inventory_fingerprint(inventory: pd.DataFrame) -> str:
    """Hash ordered filenames and sizes so a cached decode audit cannot silently drift."""
    digest = hashlib.sha256()
    for row in inventory.sort_values("filename").itertuples(index=False):
        digest.update(f"{row.filename}\t{int(row.file_size_bytes)}\n".encode("utf-8"))
    return digest.hexdigest()


def reconcile_external_ids(
    catalogue_csv: str | Path,
    image_dir: str | Path,
    teacher_train_csv: str | Path,
    teacher_test_csv: str | Path,
    *,
    root: str | Path = ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare V1 catalogue/files with teacher roles without reading target values."""
    catalogue = read_external_catalogue(catalogue_csv)
    inventory = inventory_external_images(image_dir, root=root)
    train_ids = _read_ids(teacher_train_csv)
    test_ids = _read_ids(teacher_test_csv)
    if train_ids & test_ids:
        raise ValueError("teacher train and test IDs must not overlap")

    catalogue_ids = set(catalogue["id"])
    valid_files = inventory[
        inventory["numeric_stem"] & inventory["suffix"].eq(".jpg")
    ].copy()
    file_ids = set(valid_files["id"].astype(int))
    teacher_union = train_ids | test_ids
    missing_ids = sorted(catalogue_ids - file_ids)
    extra_ids = sorted(file_ids - catalogue_ids)
    missing = pd.DataFrame(
        {
            "id": missing_ids,
            "teacher_role": [
                "train" if item_id in train_ids else "test" if item_id in test_ids else "unknown"
                for item_id in missing_ids
            ],
            "issue": "catalogue_id_without_image",
        }
    )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "scope": "label-free structural audit; V1 is never independently split",
        "catalogue_csv_sha256": compute_sha256(catalogue_csv),
        "catalogue_csv_matches_expected": (
            compute_sha256(catalogue_csv) == EXPECTED_EXTERNAL_CSV_SHA256
        ),
        "catalogue_rows": len(catalogue),
        "catalogue_unique_ids": len(catalogue_ids),
        "image_files": len(inventory),
        "valid_jpg_files": len(valid_files),
        "non_numeric_files": int((~inventory["numeric_stem"]).sum()),
        "non_jpg_files": int((~inventory["suffix"].eq(".jpg")).sum()),
        "teacher_train_ids": len(train_ids),
        "teacher_test_ids": len(test_ids),
        "teacher_roles_overlap": len(train_ids & test_ids),
        "catalogue_train_overlap": len(catalogue_ids & train_ids),
        "catalogue_test_overlap": len(catalogue_ids & test_ids),
        "catalogue_equals_teacher_union": catalogue_ids == teacher_union,
        "catalogue_ids_outside_teacher": sorted(catalogue_ids - teacher_union),
        "teacher_ids_missing_from_catalogue": sorted(teacher_union - catalogue_ids),
        "catalogue_ids_without_image": missing_ids,
        "image_ids_outside_catalogue": extra_ids,
        "inventory_fingerprint": inventory_fingerprint(inventory),
        "safe_interpretation": (
            "V1 is an ID-keyed high-resolution image variant. It inherits the canonical "
            "teacher split and is not a new training or test population."
        ),
    }
    return summary, missing


def audit_external_images(
    image_dir: str | Path,
    *,
    root: str | Path = ROOT,
    workers: int | None = None,
) -> pd.DataFrame:
    """Hash and fully decode every V1 image without modifying raw files."""
    directory = Path(image_dir)
    paths = sorted(candidate for candidate in directory.iterdir() if candidate.is_file())
    worker_count = workers or min(8, os.cpu_count() or 4)
    root_path = Path(root)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        records = list(
            executor.map(
                lambda path: audit_image("external_v1", path, root=root_path),
                paths,
            )
        )
    return pd.DataFrame(records, columns=IMAGE_AUDIT_COLUMNS).sort_values(
        ["id", "filename"], na_position="last"
    )


def ensure_external_image_audit(
    image_dir: str | Path,
    audit_csv: str | Path,
    cache_json: str | Path,
    *,
    root: str | Path = ROOT,
    workers: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load a matching cached decode audit or rebuild it from raw V1 images."""
    inventory = inventory_external_images(image_dir, root=root)
    current_fingerprint = inventory_fingerprint(inventory)
    audit_path = Path(audit_csv)
    cache_path = Path(cache_json)
    if audit_path.is_file() and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cache.get("inventory_fingerprint") == current_fingerprint
            and int(cache.get("image_files", -1)) == len(inventory)
        ):
            frame = pd.read_csv(audit_path, keep_default_na=False)
            frame["decode_ok"] = (
                frame["decode_ok"].astype(str).str.strip().str.lower().isin({"true", "1"})
            )
            return frame, True

    frame = audit_external_images(image_dir, root=root, workers=workers)
    write_deterministic_csv(frame, audit_path, index=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "inventory_fingerprint": current_fingerprint,
                "image_files": len(inventory),
                "decoded_files": int(frame["decode_ok"].sum()),
                "audit_path": _relative(audit_path, Path(root)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame, False


def build_external_variant_index(
    splits: pd.DataFrame,
    image_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Attach V1 files to the canonical split while carrying no protected labels."""
    missing_split_columns = sorted(set(SAFE_SPLIT_COLUMNS) - set(splits.columns))
    if missing_split_columns:
        raise ValueError(f"split manifest is missing columns: {missing_split_columns}")
    if splits["id"].duplicated().any():
        raise ValueError("split IDs must be unique")
    if image_audit["id"].duplicated().any():
        raise ValueError("external image audit IDs must be unique")

    decoded = image_audit[image_audit["decode_ok"].astype(bool)].copy()
    decoded["id"] = pd.to_numeric(decoded["id"], errors="raise").astype(int)
    external_columns = [
        "id",
        "path",
        "width",
        "height",
        "aspect_ratio",
        "mode",
        "format",
        "file_size_bytes",
        "sha256",
    ]
    external = decoded[external_columns].rename(
        columns={
            "path": "external_path",
            "width": "external_width",
            "height": "external_height",
            "aspect_ratio": "external_aspect_ratio",
            "mode": "external_mode",
            "format": "external_format",
            "file_size_bytes": "external_file_size_bytes",
            "sha256": "external_sha256",
        }
    )
    safe_splits = splits[list(SAFE_SPLIT_COLUMNS)].rename(
        columns={
            "path": "teacher_path",
            "width": "teacher_width",
            "height": "teacher_height",
            "file_size_bytes": "teacher_file_size_bytes",
        }
    )
    joined = safe_splits.merge(external, on="id", how="inner", validate="one_to_one")
    joined["id"] = joined["id"].astype(int)
    joined["cv_fold"] = pd.to_numeric(joined["cv_fold"], errors="coerce").astype("Int64")
    return joined.sort_values("id").reset_index(drop=True)


def select_development_pairs(
    variant_index: pd.DataFrame,
    *,
    sample_size: int = 256,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Choose a deterministic development-only sample for paired image diagnostics."""
    development = variant_index[variant_index["partition"].eq("development")].copy()
    if development.empty:
        raise ValueError("external variant index contains no development rows")
    count = min(int(sample_size), len(development))
    if count <= 0:
        raise ValueError("sample_size must be positive")
    return development.sample(n=count, random_state=seed).sort_values("id").reset_index(drop=True)


def compare_variant_pairs(
    pairs: pd.DataFrame,
    *,
    root: str | Path = ROOT,
    workers: int | None = None,
) -> pd.DataFrame:
    """Measure visual agreement between teacher and V1 copies of the same development IDs."""
    if not pairs["partition"].eq("development").all():
        raise ValueError("paired diagnostics are development-only")
    root_path = Path(root)

    def resolve(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else root_path / path

    def compare(row: dict[str, Any]) -> dict[str, Any]:
        teacher_path = resolve(row["teacher_path"])
        external_path = resolve(row["external_path"])
        teacher_dhash, teacher_ahash = compute_image_hashes(teacher_path)
        external_dhash, external_ahash = compute_image_hashes(external_path)
        metrics = compute_pair_pixel_metrics(teacher_path, external_path)
        teacher_area = max(float(row["teacher_width"]) * float(row["teacher_height"]), 1.0)
        external_area = float(row["external_width"]) * float(row["external_height"])
        teacher_bytes = max(float(row["teacher_file_size_bytes"]), 1.0)
        return {
            "id": int(row["id"]),
            "teacher_path": str(row["teacher_path"]),
            "external_path": str(row["external_path"]),
            "teacher_width": int(float(row["teacher_width"])),
            "teacher_height": int(float(row["teacher_height"])),
            "external_width": int(float(row["external_width"])),
            "external_height": int(float(row["external_height"])),
            "pixel_area_ratio": external_area / teacher_area,
            "file_size_ratio": float(row["external_file_size_bytes"]) / teacher_bytes,
            "dhash_distance": hamming_distance(teacher_dhash, external_dhash),
            "ahash_distance": hamming_distance(teacher_ahash, external_ahash),
            **metrics,
        }

    worker_count = workers or min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        records = list(executor.map(compare, pairs.to_dict("records")))
    return pd.DataFrame(records).sort_values("id").reset_index(drop=True)
