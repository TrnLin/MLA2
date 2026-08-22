"""Orchestration for the rebuildable data preparation stages."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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
from fashion.data.statistics import (
    compute_normalization_stats,
    compute_paired_normalization_stats,
    refresh_paired_normalization_provenance,
)
from fashion.data.variants import (
    audit_all_variant_alignment,
    build_training_variant_manifest,
    catalogue_high_resolution_dataset,
)

CACHE_SCHEMA_VERSION = "3.0.0"
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
    "data/processed/development_class_summary.csv",
    "data/processed/label_maps.json",
    "data/processed/normalization_original_only.json",
    "data/processed/prediction_manifest.csv",
    "data/processed/split_summary.json",
    "data/processed/splits.csv",
    "data/processed/taxonomy.json",
)
_HIGH_RESOLUTION_ARTIFACTS = (
    "data/processed/high_resolution/alignment_summary.json",
    "data/processed/high_resolution/all_alignment_pairs.csv.gz",
    "data/processed/high_resolution/catalogue.json",
    "data/processed/high_resolution/image_catalogue.csv.gz",
    "data/processed/paired_normalization.json",
    "data/processed/training_image_variants.csv.gz",
    "data/processed/training_image_variants_summary.json",
)


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
                        suffix = portable.split(marker, maxsplit=1)[1]
                        return f"{marker.strip('/')}/{suffix}"
                return candidate.name
    return value


def _artifact_digest(path: Path, root: Path) -> str:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return _json_digest(_portable_json_value(value, root))
    return compute_sha256(path)


def _safe_splits_digest(path: Path) -> str:
    """Hash the normal redacted split view, never protected target values."""
    frame = load_splits(path).sort_values("id")
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _teacher_csv_fingerprint(path: Path, development_ids: set[int]) -> dict[str, Any]:
    """Fingerprint IDs and permitted train/validation targets only.

    Cache validation runs in a short-lived subprocess from the notebook. Protected target
    text is neither hashed nor returned, so changing it cannot affect normal EDA state.
    """
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
        relative = item.relative_to(path).as_posix()
        inventory.update(f"{relative}\0{size}\n".encode("utf-8"))
    if not files:
        sample_indexes: set[int] = set()
    elif len(files) <= sample_size:
        sample_indexes = set(range(len(files)))
    elif sample_size == 1:
        sample_indexes = {0}
    else:
        sample_indexes = {
            round(index * (len(files) - 1) / (sample_size - 1)) for index in range(sample_size)
        }
    samples = [
        {
            "path": files[index].relative_to(path).as_posix(),
            "sha256": compute_sha256(files[index]),
        }
        for index in sorted(sample_indexes)
    ]
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "path_and_size_sha256": inventory.hexdigest(),
        "sample_size": len(samples),
        "sample_sha256": _json_digest(samples),
    }


def _canonical_high_resolution_root(root: Path) -> Path:
    dataset_dir = root / "data/fashion-dataset"
    for candidate in (dataset_dir, dataset_dir / "fashion-dataset"):
        if (candidate / "images").is_dir() and (candidate / "images.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "the doubled image policy requires a complete data/fashion-dataset image tree"
    )


def _input_fingerprints(
    root: Path,
    splits_path: Path,
    *,
    include_high_resolution_variants: bool,
) -> dict[str, Any]:
    splits = load_splits(splits_path)
    development_ids = set(splits.loc[splits["partition"].isin({"train", "val"}), "id"].astype(int))
    review_paths = (
        root / "docs/reviews/cross_role_near_duplicate_review.csv",
        root / "docs/reviews/near_duplicate_policy_review.csv",
        root / "docs/reviews/product_name_policy_review.csv",
        root / "docs/reviews/product_name_pre_policy_triage.json",
    )
    fingerprints: dict[str, Any] = {
        "teacher_train": _teacher_csv_fingerprint(
            root / "data/raw/teacher/train/styles_train.csv", development_ids
        ),
        "teacher_prediction": _id_only_csv_fingerprint(
            root / "data/raw/teacher/test/styles_prediction.csv"
        ),
        "teacher_train_images": _tree_fingerprint(root / "data/raw/teacher/train/images_train"),
        "teacher_prediction_images": _tree_fingerprint(root / "data/raw/teacher/test/images_test"),
        "pending_review_inputs": {
            path.relative_to(root).as_posix(): compute_sha256(path)
            for path in review_paths
            if path.is_file()
        },
    }
    if include_high_resolution_variants:
        high_resolution_root = _canonical_high_resolution_root(root)
        fingerprints.update(
            {
                "high_resolution_images": _tree_fingerprint(high_resolution_root / "images"),
                "high_resolution_images_csv": {
                    "sha256": compute_sha256(high_resolution_root / "images.csv")
                },
            }
        )
    return fingerprints


def _artifact_fingerprints(root: Path, *, include_high_resolution_variants: bool) -> dict[str, str]:
    relative_paths = list(_BASE_ARTIFACTS)
    if include_high_resolution_variants:
        relative_paths.extend(_HIGH_RESOLUTION_ARTIFACTS)
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared-data cache is missing critical artifacts: {missing}")
    fingerprints = {
        relative: _artifact_digest(root / relative, root) for relative in relative_paths
    }
    return dict(sorted(fingerprints.items()))


def write_preparation_cache(
    root: str | Path = ROOT,
    *,
    include_high_resolution_variants: bool = True,
) -> dict[str, Any]:
    """Write deterministic fingerprints after a successful full forensic rebuild."""
    root = Path(root)
    splits_path = root / "data/processed/splits.csv"
    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "mode": "full_forensic_rebuild_completed",
        "include_high_resolution_variants": include_high_resolution_variants,
        "input_fingerprints": _input_fingerprints(
            root,
            splits_path,
            include_high_resolution_variants=include_high_resolution_variants,
        ),
        "artifact_fingerprints": _artifact_fingerprints(
            root,
            include_high_resolution_variants=include_high_resolution_variants,
        ),
        "validation_policy": {
            "image_inventory": (
                "all relative paths and sizes plus 64 deterministic SHA-256 samples; "
                "this is not full raw-image content assurance"
            ),
            "teacher_targets": "train and validation only; protected target values are ignored",
            "critical_artifacts": (
                "full file SHA-256 for protected-safe tables, including the persisted redacted "
                "split; semantic JSON SHA-256 ignores only runtime and machine-path noise"
            ),
        },
    }
    cache_path = root / "data/processed" / CACHE_FILENAME
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache


def validate_prepared_data_cache(
    root: str | Path = ROOT,
    *,
    include_high_resolution_variants: bool = True,
) -> dict[str, Any]:
    """Validate source inventory/sample guards and protected-safe prepared artifacts."""
    root = Path(root)
    cache_path = root / "data/processed" / CACHE_FILENAME
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"prepared-data cache is missing: {cache_path}; run explicit full forensic mode"
        )
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("prepared-data cache schema changed; run explicit full forensic mode")
    if cached.get("include_high_resolution_variants") != include_high_resolution_variants:
        raise ValueError("prepared-data cache does not match the requested image-variant policy")
    current_inputs = _input_fingerprints(
        root,
        root / "data/processed/splits.csv",
        include_high_resolution_variants=include_high_resolution_variants,
    )
    if current_inputs != cached.get("input_fingerprints"):
        raise ValueError("prepared-data inputs changed; run explicit full forensic mode")
    current_artifacts = _artifact_fingerprints(
        root,
        include_high_resolution_variants=include_high_resolution_variants,
    )
    if current_artifacts != cached.get("artifact_fingerprints"):
        raise ValueError(
            "a critical prepared-data artifact is stale or changed; rebuild in full mode"
        )
    return {
        "cache": cache_path.relative_to(root).as_posix(),
        "status": "validated",
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
        "raw_image_assurance": "all_paths_and_sizes_plus_64_content_samples_per_tree",
        "raw_image_content_fully_hashed": False,
        "critical_prepared_artifacts_fully_hashed": True,
        "critical_prepared_artifact_target_scope": (
            "full hashes of protected-safe artifacts; split holdout/quarantine targets redacted"
        ),
    }


def refresh_protected_safe_tabular_artifacts(
    root: str | Path = ROOT,
    *,
    include_high_resolution_variants: bool = True,
) -> dict[str, Any]:
    """Refresh small public tables on fixed partitions without rescanning images."""
    root = Path(root)
    processed_dir = root / "data/processed"
    audit_dir = processed_dir / "audit"
    train_csv = root / "data/raw/teacher/train/styles_train.csv"
    prediction_csv = root / "data/raw/teacher/test/styles_prediction.csv"
    refresh_structural_csv_summary(
        train_csv,
        prediction_csv,
        audit_dir / "image_audit.csv.gz",
        audit_dir / "csv_summary.json",
        root,
    )
    refresh_fixed_split_public_artifacts(
        processed_dir / "splits.csv",
        processed_dir / "split_summary.json",
        processed_dir / "taxonomy.json",
        processed_dir / "development_class_summary.csv",
    )
    write_label_maps_from_splits(
        processed_dir / "splits.csv",
        processed_dir / "label_maps.json",
        targets=TARGET_COLUMNS,
    )
    write_development_target_audit(
        processed_dir / "splits.csv",
        audit_dir / "csv_summary.json",
        audit_dir / "missing_values.csv",
        audit_dir / "target_class_counts.csv",
        audit_dir / "issues.csv",
    )
    if include_high_resolution_variants:
        high_dir = processed_dir / "high_resolution"
        catalogue_csv = high_dir / "image_catalogue.csv.gz"
        variants_csv = processed_dir / "training_image_variants.csv.gz"
        build_training_variant_manifest(
            splits_csv=processed_dir / "splits.csv",
            high_resolution_catalogue_csv=catalogue_csv,
            official_prediction_csv=None,
            trusted_prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            output_csv=variants_csv,
            summary_output=processed_dir / "training_image_variants_summary.json",
            root=root,
        )
        refresh_paired_normalization_provenance(
            variants_csv=variants_csv,
            splits_csv=processed_dir / "splits.csv",
            output_path=processed_dir / "paired_normalization.json",
        )
        catalogue_summary_path = high_dir / "catalogue.json"
        catalogue_summary = json.loads(catalogue_summary_path.read_text(encoding="utf-8"))
        catalogue_summary["inputs"]["original_image_audit_csv"] = {
            "path": "data/processed/audit/image_audit.csv.gz",
            "sha256": compute_sha256(audit_dir / "image_audit.csv.gz"),
        }
        catalogue_summary["inputs"]["splits_csv"] = {
            "path": "data/processed/splits.csv",
            "sha256": compute_sha256(processed_dir / "splits.csv"),
        }
        catalogue_summary["images"]["catalogue_sha256"] = compute_sha256(catalogue_csv)
        catalogue_summary_path.write_text(
            json.dumps(catalogue_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        alignment_path = high_dir / "all_alignment_pairs.csv.gz"
        alignment_summary_path = high_dir / "alignment_summary.json"
        alignment_summary = json.loads(alignment_summary_path.read_text(encoding="utf-8"))
        alignment_summary["original_hashes_csv"] = {
            "path": "data/processed/audit/perceptual_hashes.csv.gz",
            "sha256": compute_sha256(audit_dir / "perceptual_hashes.csv.gz"),
        }
        alignment_summary["high_resolution_catalogue_csv"] = {
            "path": "data/processed/high_resolution/image_catalogue.csv.gz",
            "sha256": compute_sha256(catalogue_csv),
        }
        alignment_summary["output_csv"] = (
            "data/processed/high_resolution/all_alignment_pairs.csv.gz"
        )
        alignment_summary["output_sha256"] = compute_sha256(alignment_path)
        alignment_summary_path.write_text(
            json.dumps(alignment_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    cache = write_preparation_cache(
        root=root,
        include_high_resolution_variants=include_high_resolution_variants,
    )
    return {
        "status": "protected-safe tabular artifacts refreshed",
        "protected_target_values_hashed": 0,
        "cache_schema_version": cache["schema_version"],
    }


def prepare_data(
    root: str | Path = ROOT,
    workers: int | None = None,
    *,
    include_high_resolution_variants: bool = True,
) -> None:
    """Build the split, train-only statistics, and paired image-variant manifest."""
    root = Path(root)
    train_dir = root / "data/raw/teacher/train"
    prediction_dir = root / "data/raw/teacher/test"
    processed_dir = root / "data/processed"
    audit_dir = processed_dir / "audit"

    audit_raw_data(
        train_csv=train_dir / "styles_train.csv",
        prediction_csv=prediction_dir / "styles_prediction.csv",
        train_image_dir=train_dir / "images_train",
        prediction_image_dir=prediction_dir / "images_test",
        output_dir=audit_dir,
        root=root,
        workers=workers,
    )
    with TemporaryDirectory(prefix="labelled-build-", dir=processed_dir) as staging_dir:
        labelled_staging = Path(staging_dir) / "labelled_manifest.csv"
        build_manifests(
            train_csv=train_dir / "styles_train.csv",
            prediction_csv=prediction_dir / "styles_prediction.csv",
            image_audit_csv=audit_dir / "image_audit.csv.gz",
            train_output=labelled_staging,
            prediction_output=processed_dir / "prediction_manifest.csv",
        )
        run_perceptual_audit(
            train_manifest_csv=labelled_staging,
            prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            output_dir=audit_dir,
            root=root,
            cross_role_review_csv=root / "docs/reviews/cross_role_near_duplicate_review.csv",
            policy_review_csv=root / "docs/reviews/near_duplicate_policy_review.csv",
            workers=workers,
        )
        build_product_families(
            train_manifest_csv=labelled_staging,
            prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            candidates_csv=audit_dir / "near_duplicate_candidates.csv.gz",
            output_csv=audit_dir / "product_family_groups.csv.gz",
            summary_output=audit_dir / "product_family_summary.json",
            product_name_review_csv=root / "docs/reviews/product_name_policy_review.csv",
            product_name_triage_json=root / "docs/reviews/product_name_pre_policy_triage.json",
        )
        make_splits(
            train_manifest_csv=labelled_staging,
            duplicate_groups_csv=audit_dir / "exact_duplicate_groups.csv",
            product_families_csv=audit_dir / "product_family_groups.csv.gz",
            output_csv=processed_dir / "splits.csv",
            summary_output=processed_dir / "split_summary.json",
            taxonomy_output=processed_dir / "taxonomy.json",
            development_summary_output=processed_dir / "development_class_summary.csv",
        )
    write_development_target_audit(
        processed_dir / "splits.csv",
        audit_dir / "csv_summary.json",
        audit_dir / "missing_values.csv",
        audit_dir / "target_class_counts.csv",
        audit_dir / "issues.csv",
    )
    write_label_maps_from_splits(
        processed_dir / "splits.csv",
        processed_dir / "label_maps.json",
        targets=("articleType", "season", "gender", "usage"),
    )
    compute_normalization_stats(
        splits_csv=processed_dir / "splits.csv",
        output_path=processed_dir / "normalization_original_only.json",
        root=root,
        workers=workers,
    )
    if include_high_resolution_variants:
        high_resolution_dir = root / "data/fashion-dataset"
        if not high_resolution_dir.is_dir():
            raise FileNotFoundError(
                "the doubled image policy requires data/fashion-dataset; "
                f"missing directory: {high_resolution_dir}"
            )
        catalogue = catalogue_high_resolution_dataset(
            dataset_dir=high_resolution_dir,
            splits_csv=processed_dir / "splits.csv",
            original_image_audit_csv=audit_dir / "image_audit.csv.gz",
            official_prediction_csv=None,
            trusted_prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            output_dir=processed_dir / "high_resolution",
            root=root,
            workers=workers,
        )
        build_training_variant_manifest(
            splits_csv=processed_dir / "splits.csv",
            high_resolution_catalogue_csv=catalogue["image_catalogue"],
            official_prediction_csv=None,
            trusted_prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            output_csv=processed_dir / "training_image_variants.csv.gz",
            summary_output=processed_dir / "training_image_variants_summary.json",
            root=root,
        )
        compute_paired_normalization_stats(
            variants_csv=processed_dir / "training_image_variants.csv.gz",
            variants_summary_json=processed_dir / "training_image_variants_summary.json",
            splits_csv=processed_dir / "splits.csv",
            official_prediction_csv=None,
            trusted_prediction_manifest_csv=processed_dir / "prediction_manifest.csv",
            output_path=processed_dir / "paired_normalization.json",
            root=root,
            workers=workers,
        )
        audit_all_variant_alignment(
            original_hashes_csv=audit_dir / "perceptual_hashes.csv.gz",
            high_resolution_catalogue_csv=catalogue["image_catalogue"],
            output_csv=processed_dir / "high_resolution/all_alignment_pairs.csv.gz",
            summary_output=processed_dir / "high_resolution/alignment_summary.json",
            root=root,
            workers=workers,
        )
    write_preparation_cache(
        root=root,
        include_high_resolution_variants=include_high_resolution_variants,
    )
