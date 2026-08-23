"""Teacher-only orchestration for rebuildable shared data preparation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from fashion.config import ROOT, TARGET_COLUMNS
from fashion.data.audit import (
    audit_raw_data,
    refresh_structural_csv_summary,
    write_development_target_audit,
)
from fashion.data.dataset import load_splits
from fashion.data.families import build_product_families
from fashion.data.hashing import compute_sha256
from fashion.data.manifests import build_manifests
from fashion.data.metadata import write_label_maps_from_splits
from fashion.data.perceptual import run_perceptual_audit
from fashion.data.splits import make_splits, refresh_fixed_split_public_artifacts

CACHE_SCHEMA_VERSION = "5.0.0"
CACHE_FILENAME = "preparation_cache.json"
CACHE_SAMPLE_SIZE = 64

_BASE_ARTIFACTS = (
    "data/processed/audit/csv_summary.json",
    "data/processed/audit/exact_duplicate_groups.csv",
    "data/processed/audit/image_audit.csv.gz",
    "data/processed/audit/issues.csv",
    "data/processed/audit/missing_values.csv",
    "data/processed/audit/near_duplicate_candidates.csv.gz",
    "data/processed/audit/near_duplicate_summary.json",
    "data/processed/audit/perceptual_hashes.csv.gz",
    "data/processed/audit/product_family_groups.csv.gz",
    "data/processed/audit/product_family_summary.json",
    "data/processed/audit/target_class_counts.csv",
    "data/processed/cv_fold_summary.json",
    "data/processed/development_class_summary.csv",
    "data/processed/development_image_profile.json",
    "data/processed/label_maps.json",
    "data/processed/prediction_manifest.csv",
    "data/processed/split_summary.json",
    "data/processed/splits.csv",
    "data/processed/taxonomy.json",
)

# Kept temporarily as an empty compatibility symbol for downstream imports during migration.
_HIGH_RESOLUTION_ARTIFACTS: tuple[str, ...] = ()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_json_value(value: Any, root: Path) -> Any:
    """Remove volatile timings and make paths under the project root relative."""
    if isinstance(value, dict):
        return {
            key: _portable_json_value(item, root)
            for key, item in value.items()
            if key not in {"runtime_seconds"}
        }
    if isinstance(value, list):
        return [_portable_json_value(item, root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                portable = candidate.as_posix()
                for marker in ("/data/", "/docs/", "/notebooks/", "/results/"):
                    if marker in portable:
                        return f"{marker.strip('/')}/{portable.split(marker, 1)[1]}"
                return candidate.name
    return value


def _artifact_digest(path: Path, root: Path) -> str:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return _json_digest(_portable_json_value(value, root))
    return compute_sha256(path)


def _teacher_csv_fingerprint(path: Path, development_ids: set[int]) -> dict[str, Any]:
    """Hash header/IDs and permitted development targets, never protected targets."""
    structure = hashlib.sha256()
    development_targets = hashlib.sha256()
    row_count = 0
    development_row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"teacher CSV has no ID header: {path}")
        missing_targets = set(TARGET_COLUMNS).difference(reader.fieldnames)
        if missing_targets:
            raise ValueError(f"teacher CSV is missing targets {sorted(missing_targets)}: {path}")
        structure.update(json.dumps(reader.fieldnames).encode("utf-8"))
        for row in reader:
            item_id = int(row["id"])
            structure.update(f"{item_id}\n".encode("ascii"))
            row_count += 1
            if item_id in development_ids:
                values = [item_id, *(row[target] for target in TARGET_COLUMNS)]
                development_targets.update(
                    json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
                development_targets.update(b"\n")
                development_row_count += 1
    return {
        "rows": row_count,
        "header_and_id_sha256": structure.hexdigest(),
        "development_rows": development_row_count,
        "development_target_sha256": development_targets.hexdigest(),
        "protected_target_values_hashed": 0,
    }


def _id_only_csv_fingerprint(path: Path) -> dict[str, Any]:
    structure = hashlib.sha256()
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"CSV has no ID header: {path}")
        structure.update(json.dumps(reader.fieldnames).encode("utf-8"))
        for row in reader:
            structure.update(f"{int(row['id'])}\n".encode("ascii"))
            row_count += 1
    return {"rows": row_count, "header_and_id_sha256": structure.hexdigest()}


def _tree_fingerprint(path: Path, *, sample_size: int = CACHE_SAMPLE_SIZE) -> dict[str, Any]:
    """Check every relative name/size and hash a deterministic compact sample."""
    if not path.is_dir():
        raise FileNotFoundError(f"input image directory is missing: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    inventory = hashlib.sha256()
    total_bytes = 0
    for item in files:
        size = item.stat().st_size
        total_bytes += size
        inventory.update(f"{item.relative_to(path).as_posix()}\0{size}\n".encode("utf-8"))
    if not files:
        indexes: set[int] = set()
    elif len(files) <= sample_size:
        indexes = set(range(len(files)))
    elif sample_size == 1:
        indexes = {0}
    else:
        indexes = {
            round(index * (len(files) - 1) / (sample_size - 1))
            for index in range(sample_size)
        }
    samples = [
        {
            "path": files[index].relative_to(path).as_posix(),
            "sha256": compute_sha256(files[index]),
        }
        for index in sorted(indexes)
    ]
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "path_and_size_sha256": inventory.hexdigest(),
        "sample_size": len(samples),
        "sample_sha256": _json_digest(samples),
    }


def _input_fingerprints(root: Path, splits_path: Path) -> dict[str, Any]:
    splits = load_splits(splits_path)
    development_ids = set(
        splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    )
    return {
        "teacher_train": _teacher_csv_fingerprint(
            root / "data/raw/teacher/train/styles_train.csv", development_ids
        ),
        "teacher_prediction": _id_only_csv_fingerprint(
            root / "data/raw/teacher/test/styles_prediction.csv"
        ),
        "teacher_train_images": _tree_fingerprint(
            root / "data/raw/teacher/train/images_train"
        ),
        "teacher_prediction_images": _tree_fingerprint(
            root / "data/raw/teacher/test/images_test"
        ),
    }


def _artifact_fingerprints(root: Path) -> dict[str, str]:
    missing = [relative for relative in _BASE_ARTIFACTS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared-data cache is missing critical artifacts: {missing}")
    return {
        relative: _artifact_digest(root / relative, root) for relative in sorted(_BASE_ARTIFACTS)
    }


def write_development_image_profile(root: str | Path = ROOT) -> dict[str, Any]:
    """Write structural image facts that must not be used as fitted model statistics."""
    root = Path(root)
    splits = load_splits(root / "data/processed/splits.csv")
    development = splits[splits["partition"].eq("development")]
    profile = {
        "schema_version": "1.0.0",
        "source_scope": "development",
        "allowed_for_model_fit": False,
        "purpose": "descriptive image audit only",
        "rows": int(len(development)),
        "width": {
            "minimum": float(pd.to_numeric(development["width"]).min()),
            "median": float(pd.to_numeric(development["width"]).median()),
            "maximum": float(pd.to_numeric(development["width"]).max()),
        },
        "height": {
            "minimum": float(pd.to_numeric(development["height"]).min()),
            "median": float(pd.to_numeric(development["height"]).median()),
            "maximum": float(pd.to_numeric(development["height"]).max()),
        },
        "aspect_ratio": {
            "minimum": float(pd.to_numeric(development["aspect_ratio"]).min()),
            "median": float(pd.to_numeric(development["aspect_ratio"]).median()),
            "maximum": float(pd.to_numeric(development["aspect_ratio"]).max()),
        },
        "mode_counts": {
            str(key): int(value) for key, value in development["mode"].value_counts().items()
        },
        "format_counts": {
            str(key): int(value) for key, value in development["format"].value_counts().items()
        },
        "limitation": (
            "task owners must fit normalization and choose resize/crop/padding inside their own "
            "training-fold workflow"
        ),
    }
    output = root / "data/processed/development_image_profile.json"
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def write_preparation_cache(root: str | Path = ROOT) -> dict[str, Any]:
    """Write deterministic teacher-only fingerprints after a successful build."""
    root = Path(root)
    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mode": "full_teacher_only_rebuild_completed",
        "shared_source_policy": "teacher_only",
        "input_fingerprints": _input_fingerprints(
            root, root / "data/processed/splits.csv"
        ),
        "artifact_fingerprints": _artifact_fingerprints(root),
        "validation_policy": {
            "full_build_image_order": "raw_sha256_before_decode",
            "cached_image_inventory": (
                "all relative paths and sizes plus 64 deterministic SHA-256 samples per tree"
            ),
            "teacher_targets": "development only; protected target values are ignored",
            "critical_artifacts": "full protected-safe artifact hashes",
        },
    }
    path = root / "data/processed" / CACHE_FILENAME
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache


def validate_prepared_data_cache(root: str | Path = ROOT) -> dict[str, Any]:
    """Validate teacher inputs and protected-safe prepared artifacts."""
    root = Path(root)
    cache_path = root / "data/processed" / CACHE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(f"prepared-data cache is missing: {cache_path}")
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("prepared-data cache schema changed; run explicit full mode")
    if cached.get("shared_source_policy") != "teacher_only":
        raise ValueError("prepared-data cache does not use the teacher-only shared policy")
    current_inputs = _input_fingerprints(root, root / "data/processed/splits.csv")
    if current_inputs != cached.get("input_fingerprints"):
        raise ValueError("prepared-data inputs changed; run explicit full mode")
    current_artifacts = _artifact_fingerprints(root)
    if current_artifacts != cached.get("artifact_fingerprints"):
        raise ValueError("a critical prepared-data artifact changed; run explicit full mode")
    return {
        "cache": cache_path.relative_to(root).as_posix(),
        "status": "validated",
        "shared_source_policy": "teacher_only",
        "image_files_inventoried": sum(
            record["file_count"]
            for name, record in current_inputs.items()
            if name.endswith("_images")
        ),
        "image_files_content_sampled": sum(
            record["sample_size"]
            for name, record in current_inputs.items()
            if name.endswith("_images")
        ),
        "protected_target_values_hashed": 0,
    }


def refresh_protected_safe_tabular_artifacts(root: str | Path = ROOT) -> dict[str, Any]:
    """Refresh public tables on fixed membership without rescanning image pixels."""
    root = Path(root)
    processed = root / "data/processed"
    audit = processed / "audit"
    refresh_structural_csv_summary(
        root / "data/raw/teacher/train/styles_train.csv",
        root / "data/raw/teacher/test/styles_prediction.csv",
        audit / "image_audit.csv.gz",
        audit / "csv_summary.json",
        root,
    )
    refresh_fixed_split_public_artifacts(
        processed / "splits.csv",
        processed / "split_summary.json",
        processed / "cv_fold_summary.json",
        processed / "taxonomy.json",
        processed / "development_class_summary.csv",
    )
    write_label_maps_from_splits(
        processed / "splits.csv", processed / "label_maps.json", TARGET_COLUMNS
    )
    write_development_target_audit(
        processed / "splits.csv",
        audit / "csv_summary.json",
        audit / "missing_values.csv",
        audit / "target_class_counts.csv",
        audit / "issues.csv",
    )
    write_development_image_profile(root)
    cache = write_preparation_cache(root)
    return {
        "status": "protected-safe tabular artifacts refreshed",
        "protected_target_values_hashed": 0,
        "cache_schema_version": cache["schema_version"],
    }


def prepare_data(
    root: str | Path = ROOT,
    workers: int | None = None,
    *,
    initialize_split: bool = False,
) -> None:
    """Run the teacher-only forensic build while preserving canonical membership."""
    root = Path(root)
    train = root / "data/raw/teacher/train"
    prediction = root / "data/raw/teacher/test"
    processed = root / "data/processed"
    audit = processed / "audit"
    audit_raw_data(
        train_csv=train / "styles_train.csv",
        prediction_csv=prediction / "styles_prediction.csv",
        train_image_dir=train / "images_train",
        prediction_image_dir=prediction / "images_test",
        output_dir=audit,
        root=root,
        workers=workers,
    )
    with TemporaryDirectory(prefix="labelled-build-", dir=processed) as staging_dir:
        labelled = Path(staging_dir) / "labelled_manifest.csv"
        build_manifests(
            train_csv=train / "styles_train.csv",
            prediction_csv=prediction / "styles_prediction.csv",
            image_audit_csv=audit / "image_audit.csv.gz",
            train_output=labelled,
            prediction_output=processed / "prediction_manifest.csv",
        )
        run_perceptual_audit(
            train_manifest_csv=labelled,
            prediction_manifest_csv=processed / "prediction_manifest.csv",
            output_dir=audit,
            root=root,
            workers=workers,
        )
        build_product_families(
            train_manifest_csv=labelled,
            prediction_manifest_csv=processed / "prediction_manifest.csv",
            candidates_csv=audit / "near_duplicate_candidates.csv.gz",
            output_csv=audit / "product_family_groups.csv.gz",
            summary_output=audit / "product_family_summary.json",
            sealed_splits_csv=processed / "splits.csv",
        )
        make_splits(
            train_manifest_csv=labelled,
            duplicate_groups_csv=audit / "exact_duplicate_groups.csv",
            product_families_csv=audit / "product_family_groups.csv.gz",
            output_csv=processed / "splits.csv",
            summary_output=processed / "split_summary.json",
            cv_summary_output=processed / "cv_fold_summary.json",
            taxonomy_output=processed / "taxonomy.json",
            development_summary_output=processed / "development_class_summary.csv",
            initialize_split=initialize_split,
        )
    write_development_target_audit(
        processed / "splits.csv",
        audit / "csv_summary.json",
        audit / "missing_values.csv",
        audit / "target_class_counts.csv",
        audit / "issues.csv",
    )
    write_label_maps_from_splits(
        processed / "splits.csv", processed / "label_maps.json", TARGET_COLUMNS
    )
    write_development_image_profile(root)
    write_preparation_cache(root)
