"""Catalogue same-ID image variants and build a leak-safe training manifest."""

from __future__ import annotations

import hashlib
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fashion.config import AUDIT_DIR, RANDOM_SEED, ROOT, SPLITS_CSV, TARGET_COLUMNS, TEST_CSV
from fashion.data.dataset import load_splits, load_splits_for_final_evaluation
from fashion.data.hashing import compute_sha256, write_deterministic_csv

HIGH_RES_DATA_DIR = ROOT / "data/fashion-dataset"
HIGH_RES_AUDIT_DIR = ROOT / "data/processed/high_resolution"
HIGH_RES_IMAGE_CATALOGUE_CSV = HIGH_RES_AUDIT_DIR / "image_catalogue.csv.gz"
HIGH_RES_CATALOGUE_JSON = HIGH_RES_AUDIT_DIR / "catalogue.json"
TRAINING_VARIANTS_CSV = ROOT / "data/processed/training_image_variants.csv.gz"
TRAINING_VARIANTS_SUMMARY_JSON = ROOT / "data/processed/training_image_variants_summary.json"
ALIGNMENT_AUDIT_CSV = HIGH_RES_AUDIT_DIR / "all_alignment_pairs.csv.gz"
ALIGNMENT_SUMMARY_JSON = HIGH_RES_AUDIT_DIR / "alignment_summary.json"
ALIGNMENT_WARNING_METRICS_CSV = HIGH_RES_AUDIT_DIR / "alignment_warning_metrics.csv"
ALIGNMENT_WARNING_SHEET = HIGH_RES_AUDIT_DIR / "alignment_warning_examples.png"

MODEL_PARTITIONS = ("train", "val", "holdout")
VARIANT_NAMES = frozenset({"original", "high_resolution"})
KAGGLE_SOURCE = {
    "title": "Fashion Product Images Dataset",
    "owner": "Param Aggarwal",
    "ref": "paramaggarwal/fashion-product-images-dataset",
    "dataset_id": 139630,
    "version": 1,
    "version_notes": "Initial release",
    "last_updated_utc": "2019-03-14T18:57:43.307Z",
    "dataset_file_bytes": 15711279132,
    "license": "MIT",
    "source_url": (
        "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data"
    ),
    "metadata_api_url": (
        "https://www.kaggle.com/api/v1/datasets/view/paramaggarwal/fashion-product-images-dataset"
    ),
    "download_api_url": (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "paramaggarwal/fashion-product-images-dataset?datasetVersionNumber=1"
    ),
    "archive_name": "archive.zip",
    "archive_sha256": None,
    "archive_digest_limitation": (
        "The downloaded archive is not present locally. Recovering its SHA-256 would require "
        "downloading another 15.7 GB bundle, so the unpacked canonical file catalogue digest "
        "is recorded instead."
    ),
    "verified_at": "2026-08-21",
}


def _compute_image_hashes(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        difference = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int32)
        difference_bits = (difference[:, 1:] > difference[:, :-1]).ravel()
        average = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
        average_bits = (average >= average.mean()).ravel()
    return (
        int.from_bytes(np.packbits(difference_bits).tobytes(), "big"),
        int.from_bytes(np.packbits(average_bits).tobytes(), "big"),
    )


def _hamming_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def _pair_pixel_metrics(first_path: str | Path, second_path: str | Path) -> dict[str, float]:
    arrays = []
    for path in (first_path, second_path):
        with Image.open(path) as image:
            rgb = (
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((64, 64), Image.Resampling.LANCZOS)
            )
            arrays.append(np.asarray(rgb, dtype=np.float32) / 255)
    difference = np.abs(arrays[0] - arrays[1])
    return {
        "mse": float(np.square(arrays[0] - arrays[1]).mean()),
        "mae": float(difference.mean()),
        "max_difference": float(difference.max()),
    }


def _relative(path: Path, root: Path = ROOT) -> str:
    """Keep lexical repository paths portable, even through symlink components."""
    path = Path(path)
    root = Path(root)
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def _files(path: Path) -> dict[str, tuple[int, int, int]]:
    return {
        item.name: (item.stat().st_size, item.stat().st_dev, item.stat().st_ino)
        for item in path.iterdir()
        if item.is_file()
    }


def _tree_relation(
    first: Path,
    second: Path,
    *,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    first_files = _files(first)
    second_files = _files(second)
    shared = sorted(set(first_files).intersection(second_files))
    name_or_size_mismatches = sorted(
        name
        for name in set(first_files).union(second_files)
        if name not in first_files
        or name not in second_files
        or first_files[name][0] != second_files[name][0]
    )
    generator = random.Random(seed)
    sample = sorted(generator.sample(shared, min(sample_size, len(shared))))
    sample_hash_mismatches = [
        name for name in sample if compute_sha256(first / name) != compute_sha256(second / name)
    ]
    same_physical_file_pairs = sum(
        first_files[name][1:] == second_files[name][1:] for name in shared
    )
    clean = not name_or_size_mismatches and not sample_hash_mismatches
    if first.is_symlink() or second.is_symlink():
        relation = "symlinked_tree"
    elif shared and same_physical_file_pairs == len(shared):
        relation = "hard_linked_files"
    elif clean:
        relation = "separate_probable_copy"
    else:
        relation = "different"
    return {
        "relation": relation,
        "first_count": len(first_files),
        "second_count": len(second_files),
        "shared_names": len(shared),
        "same_device_inode_pairs": same_physical_file_pairs,
        "name_or_size_mismatch_count": len(name_or_size_mismatches),
        "name_or_size_mismatches": name_or_size_mismatches,
        "hash_sample_seed": seed,
        "hash_sample_size": len(sample),
        "hash_sample_mismatch_count": len(sample_hash_mismatches),
        "hash_sample_mismatches": sample_hash_mismatches,
        "limitation": (
            "All names and sizes were compared. SHA-256 was compared for a seeded sample; "
            "un-sampled large files were not byte-hashed in both trees."
        ),
    }


def discover_high_resolution_root(
    dataset_dir: str | Path = HIGH_RES_DATA_DIR,
    *,
    root: str | Path = ROOT,
    seed: int = RANDOM_SEED,
    duplicate_sample_size: int = 256,
) -> tuple[Path, dict[str, Any]]:
    """Choose one physical tree after checking a possible duplicate nested extraction."""
    dataset_dir = Path(dataset_dir)
    candidates = [
        candidate
        for candidate in (dataset_dir, dataset_dir / "fashion-dataset")
        if (candidate / "images").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"no complete Fashion Product Images tree under {dataset_dir}")
    canonical = candidates[0]
    canonical_files = [item for item in (canonical / "images").iterdir() if item.is_file()]
    canonical_image_bytes = sum(item.stat().st_size for item in canonical_files)
    result: dict[str, Any] = {
        "dataset_dir": _relative(dataset_dir, Path(root)),
        "canonical_root": _relative(canonical, Path(root)),
        "candidate_roots": [_relative(candidate, Path(root)) for candidate in candidates],
        "canonical_file_count": len(canonical_files),
        "canonical_image_bytes": canonical_image_bytes,
        "duplicate_tree": None,
    }
    if len(candidates) > 1:
        first, second = candidates[:2]
        image_relation = _tree_relation(
            first / "images", second / "images", seed=seed, sample_size=duplicate_sample_size
        )
        safe_to_scan_once = (
            image_relation["name_or_size_mismatch_count"] == 0
            and image_relation["hash_sample_mismatch_count"] == 0
        )
        result["duplicate_tree"] = {
            "images": image_relation,
            "safe_to_scan_canonical_once": safe_to_scan_once,
        }
        if not safe_to_scan_once:
            raise ValueError("outer and nested high-resolution image trees do not match")
    return canonical, result


def _parse_image_links(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, usecols=["filename"], keep_default_na=False)
    ids = frame["filename"].str.removesuffix(".jpg").astype(int)
    return frame.assign(id=ids), {
        "columns_read": ["filename"],
        "rows": len(frame),
        "unique_ids": int(ids.nunique()),
        "duplicate_ids": int(ids.duplicated().sum()),
        "sha256": compute_sha256(path),
    }


def _scan_image(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": int(path.stem) if path.stem.isdigit() else "",
        "filename": path.name,
        "path": _relative(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": "",
        "decode_ok": False,
        "format": "",
        "mode": "",
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "error": "",
    }
    try:
        payload = path.read_bytes()
        row["sha256"] = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            row.update(
                {
                    "decode_ok": True,
                    "format": image.format or "",
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                }
            )
    except Exception as error:  # Pillow raises several format-specific exception types.
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _distribution(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    quantiles = numeric.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {
        "count": len(numeric),
        "mean": float(numeric.mean()),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }


def _set_comparison(reference: set[int], observed: set[int]) -> dict[str, Any]:
    return {
        "reference_count": len(reference),
        "observed_count": len(observed),
        "missing_count": len(reference - observed),
        "missing_ids": sorted(reference - observed),
        "orphan_count": len(observed - reference),
        "orphan_ids": sorted(observed - reference),
    }


def _official_prediction_ids(
    official_prediction_csv: str | Path | None,
    trusted_prediction_manifest_csv: str | Path | None,
) -> tuple[set[int], Path, str]:
    official = Path(official_prediction_csv) if official_prediction_csv else None
    trusted = Path(trusted_prediction_manifest_csv) if trusted_prediction_manifest_csv else None
    if official and official.is_file():
        frame = pd.read_csv(official, usecols=["id"])
        return set(frame["id"].astype(int)), official, "official_styles_prediction"
    if trusted and trusted.is_file():
        frame = pd.read_csv(trusted, usecols=["id"])
        return set(frame["id"].astype(int)), trusted, "trusted_processed_prediction_manifest"
    raise FileNotFoundError("no trusted official prediction-ID source exists; merge fails closed")


def catalogue_high_resolution_dataset(
    *,
    dataset_dir: str | Path = HIGH_RES_DATA_DIR,
    splits_csv: str | Path = SPLITS_CSV,
    original_image_audit_csv: str | Path | None = None,
    official_prediction_csv: str | Path | None = TEST_CSV,
    trusted_prediction_manifest_csv: str | Path | None = None,
    output_dir: str | Path = HIGH_RES_AUDIT_DIR,
    root: str | Path = ROOT,
    seed: int = RANDOM_SEED,
    duplicate_sample_size: int = 256,
    workers: int | None = None,
) -> dict[str, Path]:
    """Scan one canonical tree and record reproducible catalogue evidence."""
    started = time.monotonic()
    canonical, layout = discover_high_resolution_root(
        dataset_dir, root=root, seed=seed, duplicate_sample_size=duplicate_sample_size
    )
    if original_image_audit_csv is None:
        original_image_audit_csv = Path(splits_csv).parent / "audit/image_audit.csv.gz"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_csv = canonical / "images.csv"
    image_links: pd.DataFrame | None = None
    links_summary: dict[str, Any] | None = None
    if images_csv.is_file():
        image_links, links_summary = _parse_image_links(images_csv)
    prediction_ids, prediction_source, prediction_source_kind = _official_prediction_ids(
        official_prediction_csv, trusted_prediction_manifest_csv
    )
    splits = load_splits(splits_csv)
    split_ids = set(splits["id"].astype(int))

    image_items = sorted(
        (item for item in (canonical / "images").iterdir() if item.is_file()),
        key=lambda value: value.name,
    )
    worker_count = workers or min(32, max(1, len(image_items)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        image_rows = list(pool.map(_scan_image, image_items))
    image_catalogue = pd.DataFrame(image_rows)
    image_catalogue["path"] = [_relative(item, Path(root)) for item in image_items]
    image_catalogue.sort_values(["id", "filename"], inplace=True, ignore_index=True)
    image_catalogue_path = output_dir / "image_catalogue.csv.gz"
    write_deterministic_csv(
        image_catalogue,
        image_catalogue_path,
        index=False,
    )

    valid_images = image_catalogue[image_catalogue["decode_ok"]].copy()
    original_geometry = pd.read_csv(
        original_image_audit_csv,
        usecols=["id", "decode_ok", "width", "height", "aspect_ratio"],
        keep_default_na=False,
    )
    original_geometry = original_geometry[
        original_geometry["decode_ok"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    original_geometry["id"] = pd.to_numeric(original_geometry["id"], errors="raise").astype(int)
    for column in ("width", "height", "aspect_ratio"):
        original_geometry[column] = pd.to_numeric(original_geometry[column], errors="raise")
    original_geometry.rename(
        columns={
            "width": "original_width",
            "height": "original_height",
            "aspect_ratio": "original_aspect_ratio",
        },
        inplace=True,
    )
    geometry = original_geometry.merge(
        valid_images[["id", "width", "height", "aspect_ratio"]],
        on="id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    geometry.rename(
        columns={
            "width": "high_resolution_width",
            "height": "high_resolution_height",
            "aspect_ratio": "high_resolution_aspect_ratio",
        },
        inplace=True,
    )
    comparable_geometry = geometry[geometry["_merge"].eq("both")].copy()
    aspect_changed = comparable_geometry[
        ~np.isclose(
            comparable_geometry["original_aspect_ratio"].astype(float),
            comparable_geometry["high_resolution_aspect_ratio"].astype(float),
            rtol=0,
            atol=1e-12,
        )
    ].copy()
    aspect_changed["relative_aspect_difference_percent"] = (
        (
            aspect_changed["high_resolution_aspect_ratio"] - aspect_changed["original_aspect_ratio"]
        ).abs()
        / aspect_changed["original_aspect_ratio"]
        * 100
    )
    not_larger = comparable_geometry[
        (comparable_geometry["high_resolution_width"] <= comparable_geometry["original_width"])
        | (comparable_geometry["high_resolution_height"] <= comparable_geometry["original_height"])
    ]
    image_ids = set(pd.to_numeric(valid_images["id"], errors="coerce").dropna().astype(int))
    link_ids = set(image_links["id"].astype(int)) if image_links is not None else None
    teacher_union = split_ids.union(prediction_ids)
    non_image_files = image_catalogue[
        ~image_catalogue["filename"].str.lower().str.endswith((".jpg", ".jpeg"))
    ]["filename"].tolist()
    corrupt = image_catalogue[~image_catalogue["decode_ok"]][["filename", "error"]].to_dict(
        "records"
    )
    summary: dict[str, Any] = {
        "schema_version": 2,
        "seed": seed,
        "runtime_seconds": time.monotonic() - started,
        "source_provenance": KAGGLE_SOURCE,
        "inputs": {
            "layout": layout,
            "splits_csv": {
                "path": _relative(Path(splits_csv), Path(root)),
                "sha256": compute_sha256(splits_csv),
            },
            "original_image_audit_csv": {
                "path": _relative(Path(original_image_audit_csv), Path(root)),
                "sha256": compute_sha256(original_image_audit_csv),
            },
            "prediction_id_source": {
                "path": _relative(prediction_source, Path(root)),
                "kind": prediction_source_kind,
                "sha256": compute_sha256(prediction_source),
            },
        },
        "source_scope": {
            "policy": "image_only",
            "images_csv": links_summary,
            "target_bearing_metadata_opened": False,
            "excluded_paths": ["styles.csv", "styles/*.json"],
            "claim": (
                "The high-resolution source contributes image bytes and safe image IDs only. "
                "No Kaggle target-bearing metadata file is opened."
            ),
        },
        "images": {
            "file_count": len(image_catalogue),
            "decoded_count": int(image_catalogue["decode_ok"].sum()),
            "corrupt_count": len(corrupt),
            "corrupt_files": corrupt,
            "non_image_file_count": len(non_image_files),
            "non_image_files": non_image_files,
            "formats": valid_images["format"].value_counts().sort_index().to_dict(),
            "modes": valid_images["mode"].value_counts().sort_index().to_dict(),
            "width": _distribution(valid_images["width"]),
            "height": _distribution(valid_images["height"]),
            "aspect_ratio": _distribution(valid_images["aspect_ratio"]),
            "file_size_bytes": _distribution(image_catalogue["file_size_bytes"]),
            "total_file_size_bytes": int(image_catalogue["file_size_bytes"].sum()),
            "full_sha256_count": int(image_catalogue["sha256"].astype(str).ne("").sum()),
            "unique_sha256_count": int(image_catalogue["sha256"].replace("", np.nan).nunique()),
            "catalogue_sha256": compute_sha256(image_catalogue_path),
            "same_id_geometry": {
                "pair_count": len(comparable_geometry),
                "original_only_count": int(geometry["_merge"].eq("left_only").sum()),
                "high_resolution_only_count": int(geometry["_merge"].eq("right_only").sum()),
                "not_larger_count": len(not_larger),
                "not_larger_ids": not_larger["id"].astype(int).tolist(),
                "aspect_changed_count": len(aspect_changed),
                "maximum_relative_aspect_difference_percent": float(
                    aspect_changed["relative_aspect_difference_percent"].max()
                    if len(aspect_changed)
                    else 0
                ),
                "geometry_interpretation": (
                    "Observed aspect changes are below 1% and are consistent with minor pixel "
                    "rounding during resizing. Geometry alone cannot prove that product content "
                    "matches."
                ),
                "aspect_changed": aspect_changed[
                    [
                        "id",
                        "original_width",
                        "original_height",
                        "original_aspect_ratio",
                        "high_resolution_width",
                        "high_resolution_height",
                        "high_resolution_aspect_ratio",
                        "relative_aspect_difference_percent",
                    ]
                ].to_dict("records"),
            },
        },
        "ids": {
            "images_csv_vs_images": (
                _set_comparison(link_ids, image_ids) if link_ids is not None else None
            ),
            "teacher_available_union_vs_images": _set_comparison(teacher_union, image_ids),
            "canonical_splits_vs_labelled_images": _set_comparison(
                split_ids, image_ids - prediction_ids
            ),
            "official_prediction_vs_images": _set_comparison(
                prediction_ids, image_ids.intersection(prediction_ids)
            ),
            "split_prediction_intersection": len(split_ids.intersection(prediction_ids)),
        },
        "limitations": [
            "The duplicate outer tree was fully compared by name and byte size, then byte-hashed "
            "with a deterministic sample instead of hashing both 15 GB copies in full.",
            "The canonical image tree was fully decoded and SHA-256 hashed once.",
            "Kaggle styles.csv and styles/*.json were not opened; metadata equality is an external "
            "source assertion, not a catalogue finding.",
        ],
    }
    summary_path = output_dir / "catalogue.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "image_catalogue": image_catalogue_path,
        "catalogue": summary_path,
    }


def build_training_variant_manifest(
    *,
    splits_csv: str | Path = SPLITS_CSV,
    high_resolution_catalogue_csv: str | Path = HIGH_RES_IMAGE_CATALOGUE_CSV,
    official_prediction_csv: str | Path | None = TEST_CSV,
    trusted_prediction_manifest_csv: str | Path | None = None,
    output_csv: str | Path = TRAINING_VARIANTS_CSV,
    summary_output: str | Path = TRAINING_VARIANTS_SUMMARY_JSON,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Pair both resolutions for every non-quarantine labelled product."""
    splits = load_splits(splits_csv)
    prediction_ids, prediction_source, prediction_source_kind = _official_prediction_ids(
        official_prediction_csv, trusted_prediction_manifest_csv
    )
    high = pd.read_csv(high_resolution_catalogue_csv, keep_default_na=False)
    high = high[high["decode_ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    high["id"] = high["id"].astype(int)
    if high["id"].duplicated().any():
        raise ValueError("high-resolution image catalogue has duplicate IDs")

    structural = [
        "id",
        "path",
        "partition",
        "sha256",
        "duplicate_group",
        "product_family_group",
        "is_cross_role_duplicate",
        "quarantine_reason",
    ]
    missing = set(structural).difference(splits.columns)
    if missing:
        raise ValueError(f"splits.csv lacks variant safety columns: {sorted(missing)}")
    eligible = splits[splits["partition"].isin(MODEL_PARTITIONS)].copy()
    original = eligible[structural].copy()
    original.rename(columns={"sha256": "canonical_sha256"}, inplace=True)
    original.insert(0, "variant_key", original["id"].astype(str) + ":original")
    original.insert(2, "variant", "original")
    original["variant_sha256"] = original["canonical_sha256"]

    high_variants = eligible[structural].merge(
        high[["id", "path", "sha256"]],
        on="id",
        how="left",
        suffixes=("_canonical", "_variant"),
        validate="one_to_one",
    )
    missing_high = high_variants["path_variant"].eq("") | high_variants["path_variant"].isna()
    if missing_high.any():
        missing_rows = high_variants.loc[missing_high, ["id", "partition"]].to_dict("records")
        raise ValueError(
            f"high-resolution variants are missing for eligible labelled IDs: {missing_rows[:20]}"
        )
    high_variants = high_variants.rename(
        columns={
            "path_variant": "path",
            "sha256_canonical": "canonical_sha256",
            "sha256_variant": "variant_sha256",
        }
    ).drop(columns=["path_canonical"])
    high_variants.insert(0, "variant_key", high_variants["id"].astype(str) + ":high_resolution")
    high_variants.insert(2, "variant", "high_resolution")

    variants = pd.concat([original, high_variants], ignore_index=True, sort=False)
    variants["sample_group"] = variants["id"].map(lambda value: f"product_{int(value)}")
    variants["independence_group"] = variants["product_family_group"]
    variants["is_training_eligible"] = variants["partition"].eq("train")
    variants["is_evaluation_eligible"] = variants["partition"].isin({"val", "holdout"})
    counts = variants.groupby("id")["variant_key"].transform("count")
    variants["per_product_weight"] = 1.0 / counts
    variants.sort_values(["partition", "id", "variant"], inplace=True, ignore_index=True)

    if variants["variant_key"].duplicated().any():
        raise ValueError("variant keys must be unique")
    if set(variants["id"]).intersection(prediction_ids):
        raise ValueError("official prediction IDs entered the image variant manifest")
    expected_variants = variants.groupby("id")["variant"].agg(set)
    incomplete_ids = expected_variants[expected_variants.ne(VARIANT_NAMES)].index.tolist()
    if incomplete_ids:
        raise ValueError(f"eligible labelled IDs lack a complete image pair: {incomplete_ids[:20]}")

    quarantine_rows = splits[splits["partition"].eq("quarantine")]
    high_quarantine_sha = set(
        high.loc[high["id"].isin(quarantine_rows["id"].astype(int)), "sha256"]
    )
    high_prediction_sha = set(high.loc[high["id"].isin(prediction_ids), "sha256"])
    high_model = variants[variants["variant"].eq("high_resolution")]
    high_model_sha = set(high_model["variant_sha256"])
    high_sha_partition_crossings = int(
        (high_model.groupby("variant_sha256")["partition"].nunique() > 1).sum()
    )
    high_duplicate_memberships = {
        frozenset(group["id"].astype(int))
        for _, group in high_model.groupby("variant_sha256")
        if len(group) > 1
    }
    canonical_duplicate_memberships = {
        frozenset(group["id"].astype(int))
        for _, group in high_model.groupby("duplicate_group")
        if len(group) > 1
    }
    duplicate_membership_mismatches = high_duplicate_memberships.symmetric_difference(
        canonical_duplicate_memberships
    )
    partition_coverage: dict[str, dict[str, int]] = {}
    for partition in MODEL_PARTITIONS:
        rows = variants[variants["partition"].eq(partition)]
        product_count = int(rows["id"].nunique())
        complete_pairs = int(rows.groupby("id")["variant"].agg(set).eq(VARIANT_NAMES).sum())
        partition_coverage[partition] = {
            "product_count": product_count,
            "variant_count": len(rows),
            "original_variant_count": int(rows["variant"].eq("original").sum()),
            "high_resolution_variant_count": int(rows["variant"].eq("high_resolution").sum()),
            "complete_pair_count": complete_pairs,
            "incomplete_pair_count": product_count - complete_pairs,
        }

    proofs: dict[str, Any] = {
        "policy": "complete_low_high_pairs_in_train_val_holdout",
        "model_product_count": int(variants["id"].nunique()),
        "model_variant_count": len(variants),
        "partition_coverage": partition_coverage,
        "official_prediction_id_intersection": len(
            set(variants["id"]).intersection(prediction_ids)
        ),
        "quarantine_id_intersection": len(set(variants["id"]).intersection(quarantine_rows["id"])),
        "quarantine_product_family_intersection": len(
            set(variants["product_family_group"]).intersection(
                quarantine_rows["product_family_group"]
            )
        ),
        "quarantine_canonical_sha256_intersection": len(
            set(variants["canonical_sha256"]).intersection(quarantine_rows["sha256"])
        ),
        "quarantine_duplicate_group_intersection": len(
            set(variants["duplicate_group"]).intersection(quarantine_rows["duplicate_group"])
        ),
        "high_resolution_sha_partition_crossing_count": high_sha_partition_crossings,
        "high_resolution_duplicate_membership_mismatch_count": len(duplicate_membership_mismatches),
        "high_resolution_sha_vs_quarantine_intersection": len(
            high_model_sha.intersection(high_quarantine_sha)
        ),
        "high_resolution_sha_vs_prediction_intersection": len(
            high_model_sha.intersection(high_prediction_sha)
        ),
        "excluded_prediction_quarantine_high_resolution_sha_group_count": len(
            high_prediction_sha.intersection(high_quarantine_sha)
        ),
        "cross_role_flagged_model_variants": int(
            variants["is_cross_role_duplicate"].astype(str).str.lower().isin({"true", "1"}).sum()
        ),
        "target_columns_in_variant_manifest": len(set(TARGET_COLUMNS).intersection(variants)),
    }
    if any(value for key, value in proofs.items() if key.endswith("intersection")):
        raise ValueError(f"variant leakage proof failed: {proofs}")
    if high_sha_partition_crossings:
        raise ValueError("a high-resolution SHA-256 group crosses model partitions")
    if duplicate_membership_mismatches:
        raise ValueError("high-resolution exact duplicate membership differs from canonical groups")
    if proofs["cross_role_flagged_model_variants"]:
        raise ValueError("cross-role forbidden variants entered modelling")
    if proofs["target_columns_in_variant_manifest"]:
        raise ValueError("protected targets entered the structural variant manifest")
    if any(values["incomplete_pair_count"] for values in partition_coverage.values()):
        raise ValueError(f"paired coverage is incomplete: {partition_coverage}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        variants,
        output_csv,
        index=False,
    )
    proofs.update(
        {
            "manifest_path": _relative(output_csv, Path(root)),
            "manifest_sha256": compute_sha256(output_csv),
            "splits_sha256": compute_sha256(splits_csv),
            "high_resolution_catalogue_path": _relative(
                Path(high_resolution_catalogue_csv), Path(root)
            ),
            "high_resolution_catalogue_sha256": compute_sha256(high_resolution_catalogue_csv),
            "prediction_id_source_kind": prediction_source_kind,
            "prediction_id_source_path": _relative(prediction_source, Path(root)),
            "prediction_id_source_sha256": compute_sha256(prediction_source),
        }
    )
    summary_output = Path(summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(proofs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proofs


def _safe_existing_project_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"variant path must be project-relative without traversal: {value}")
    full_path = root / path
    if not full_path.is_file():
        raise ValueError(f"variant image file does not exist: {path.as_posix()}")
    return full_path


def validate_image_variant_manifest(
    *,
    variants_csv: str | Path = TRAINING_VARIANTS_CSV,
    summary_json: str | Path = TRAINING_VARIANTS_SUMMARY_JSON,
    splits_csv: str | Path = SPLITS_CSV,
    official_prediction_csv: str | Path | None = TEST_CSV,
    trusted_prediction_manifest_csv: str | Path | None = None,
    root: str | Path = ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail closed unless a saved variant manifest exactly matches trusted inputs."""
    variants_csv = Path(variants_csv)
    summary_json = Path(summary_json)
    root = Path(root)
    if not variants_csv.is_file() or not summary_json.is_file():
        raise FileNotFoundError("variant manifest and its trusted summary are both required")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    if summary.get("manifest_sha256") != compute_sha256(variants_csv):
        raise ValueError("variant manifest SHA-256 does not match its trusted summary")
    if summary.get("splits_sha256") != compute_sha256(splits_csv):
        raise ValueError("splits.csv SHA-256 does not match the variant summary")
    catalogue_path_value = summary.get("high_resolution_catalogue_path")
    if not isinstance(catalogue_path_value, str):
        raise ValueError("variant summary has no trusted high-resolution catalogue path")
    catalogue_path = _safe_existing_project_path(catalogue_path_value, root)
    if summary.get("high_resolution_catalogue_sha256") != compute_sha256(catalogue_path):
        raise ValueError("high-resolution catalogue SHA-256 does not match the variant summary")
    prediction_ids, prediction_source, prediction_source_kind = _official_prediction_ids(
        official_prediction_csv, trusted_prediction_manifest_csv
    )
    if summary.get("prediction_id_source_kind") != prediction_source_kind or summary.get(
        "prediction_id_source_sha256"
    ) != compute_sha256(prediction_source):
        raise ValueError("official prediction-ID source does not match the variant summary")
    required_zero_proofs = (
        "official_prediction_id_intersection",
        "quarantine_id_intersection",
        "quarantine_product_family_intersection",
        "quarantine_canonical_sha256_intersection",
        "quarantine_duplicate_group_intersection",
        "high_resolution_sha_partition_crossing_count",
        "high_resolution_duplicate_membership_mismatch_count",
        "high_resolution_sha_vs_quarantine_intersection",
        "high_resolution_sha_vs_prediction_intersection",
        "cross_role_flagged_model_variants",
        "target_columns_in_variant_manifest",
    )
    if summary.get("policy") != "complete_low_high_pairs_in_train_val_holdout" or any(
        summary.get(key) != 0 for key in required_zero_proofs
    ):
        raise ValueError("variant summary does not contain the required zero-leakage proofs")

    variants = pd.read_csv(variants_csv, keep_default_na=False)
    required = {
        "variant_key",
        "id",
        "variant",
        "path",
        "partition",
        "canonical_sha256",
        "duplicate_group",
        "product_family_group",
        "is_cross_role_duplicate",
        "quarantine_reason",
        "variant_sha256",
        "sample_group",
        "independence_group",
        "is_training_eligible",
        "is_evaluation_eligible",
        "per_product_weight",
    }
    if missing := required.difference(variants.columns):
        raise ValueError(f"variant manifest lacks required columns: {sorted(missing)}")
    if unexpected := set(variants.columns).difference(required):
        raise ValueError(f"variant manifest has unexpected columns: {sorted(unexpected)}")
    variants["id"] = pd.to_numeric(variants["id"], errors="raise").astype(int)
    weights = pd.to_numeric(variants["per_product_weight"], errors="raise")
    splits = load_splits(splits_csv)
    structural = [
        "id",
        "partition",
        "path",
        "sha256",
        "duplicate_group",
        "product_family_group",
        "is_cross_role_duplicate",
        "quarantine_reason",
    ]
    if missing := set(structural).difference(splits.columns):
        raise ValueError(f"splits.csv lacks variant safety columns: {sorted(missing)}")
    expected = splits[splits["partition"].isin(MODEL_PARTITIONS)][structural].copy()
    expected_ids = set(expected["id"].astype(int))
    actual_ids = set(variants["id"])
    if actual_ids != expected_ids:
        raise ValueError("variant IDs do not exactly cover every eligible split ID")
    if set(variants["partition"]) - set(MODEL_PARTITIONS):
        raise ValueError("variant manifest contains a forbidden partition")
    if actual_ids.intersection(prediction_ids):
        raise ValueError("official prediction IDs entered the variant manifest")
    if summary.get("model_product_count") != len(expected_ids) or summary.get(
        "model_variant_count"
    ) != len(variants):
        raise ValueError("variant summary counts do not match the trusted split")
    if variants["variant_key"].duplicated().any():
        raise ValueError("variant keys must be unique")
    counts = variants.groupby("id", sort=False).size()
    if not counts.eq(2).all():
        raise ValueError("every eligible product must have exactly two variant rows")
    pair_sets = variants.groupby("id", sort=False)["variant"].agg(set)
    if not pair_sets.eq(VARIANT_NAMES).all():
        raise ValueError("every product must have one original and one high-resolution variant")
    expected_keys = variants["id"].astype(str) + ":" + variants["variant"].astype(str)
    if not variants["variant_key"].astype(str).eq(expected_keys).all():
        raise ValueError("variant keys do not match ID and variant")
    if not weights.eq(0.5).all() or not weights.groupby(variants["id"]).sum().eq(1.0).all():
        raise ValueError("variant weights must be exactly 0.5 and sum to one per product")

    expected_check = expected.rename(
        columns={"path": "expected_original_path", "sha256": "expected_canonical_sha256"}
    )
    checked = variants.merge(expected_check, on="id", how="left", validate="many_to_one")
    for column in (
        "partition",
        "duplicate_group",
        "product_family_group",
        "is_cross_role_duplicate",
        "quarantine_reason",
    ):
        left = f"{column}_x"
        right = f"{column}_y"
        if not checked[left].astype(str).eq(checked[right].astype(str)).all():
            raise ValueError(f"variant {column} differs from splits.csv")
        checked.drop(columns=[right], inplace=True)
        checked.rename(columns={left: column}, inplace=True)
    if (
        not checked["canonical_sha256"]
        .astype(str)
        .eq(checked["expected_canonical_sha256"].astype(str))
        .all()
    ):
        raise ValueError("variant canonical SHA-256 differs from splits.csv")
    originals = checked[checked["variant"].eq("original")]
    if not originals["path"].astype(str).eq(originals["expected_original_path"].astype(str)).all():
        raise ValueError("original variant paths differ from splits.csv")
    if (
        not originals["variant_sha256"]
        .astype(str)
        .eq(originals["expected_canonical_sha256"].astype(str))
        .all()
    ):
        raise ValueError("original variant SHA-256 differs from splits.csv")
    if (
        not checked["sample_group"]
        .astype(str)
        .eq(checked["id"].map(lambda value: f"product_{int(value)}"))
        .all()
    ):
        raise ValueError("sample groups do not match product IDs")
    if (
        not checked["independence_group"]
        .astype(str)
        .eq(checked["product_family_group"].astype(str))
        .all()
    ):
        raise ValueError("independence groups do not match product-family groups")
    training_flags = checked["is_training_eligible"].astype(str).str.lower().isin({"true", "1"})
    evaluation_flags = checked["is_evaluation_eligible"].astype(str).str.lower().isin({"true", "1"})
    if (
        not training_flags.eq(checked["partition"].eq("train")).all()
        or not evaluation_flags.eq(checked["partition"].isin({"val", "holdout"})).all()
    ):
        raise ValueError("variant eligibility flags do not match partitions")
    for value in checked["path"]:
        _safe_existing_project_path(value, root)
    high_catalogue = pd.read_csv(catalogue_path, keep_default_na=False)
    needed_catalogue = {"id", "path", "sha256", "decode_ok"}
    if missing := needed_catalogue.difference(high_catalogue.columns):
        raise ValueError(f"high-resolution catalogue lacks columns: {sorted(missing)}")
    high_catalogue = high_catalogue[
        high_catalogue["decode_ok"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    high_catalogue["id"] = pd.to_numeric(high_catalogue["id"], errors="raise").astype(int)
    high_rows = checked[checked["variant"].eq("high_resolution")].merge(
        high_catalogue[["id", "path", "sha256"]],
        on="id",
        how="left",
        suffixes=("_manifest", "_catalogue"),
        validate="one_to_one",
    )
    if (
        not high_rows["path_manifest"].astype(str).eq(high_rows["path_catalogue"].astype(str)).all()
        or not high_rows["variant_sha256"].astype(str).eq(high_rows["sha256"].astype(str)).all()
    ):
        raise ValueError(
            "high-resolution variant paths or hashes differ from the trusted catalogue"
        )
    return variants, splits


def load_image_variants(
    *,
    partition: Literal["train", "val", "holdout"] = "train",
    variants_csv: str | Path = TRAINING_VARIANTS_CSV,
    summary_json: str | Path = TRAINING_VARIANTS_SUMMARY_JSON,
    splits_csv: str | Path = SPLITS_CSV,
    official_prediction_csv: str | Path | None = TEST_CSV,
    trusted_prediction_manifest_csv: str | Path | None = None,
    root: str | Path = ROOT,
    evaluation_unlocked: bool = False,
) -> pd.DataFrame:
    """Load both same-product variants while preserving protected-target redaction."""
    if evaluation_unlocked and partition != "holdout":
        raise ValueError("the explicit evaluation unlock is only valid for holdout")
    variants, redacted_splits = validate_image_variant_manifest(
        variants_csv=variants_csv,
        summary_json=summary_json,
        splits_csv=splits_csv,
        official_prediction_csv=official_prediction_csv,
        trusted_prediction_manifest_csv=trusted_prediction_manifest_csv,
        root=root,
    )
    selected = variants[variants["partition"].eq(partition)].copy()
    splits = (
        load_splits_for_final_evaluation(splits_csv, evaluation_unlocked=True)
        if evaluation_unlocked
        else redacted_splits
    )
    split_rows = splits[splits["partition"].eq(partition)].copy()
    payload_columns = ["id", *[column for column in split_rows if column not in selected]]
    selected = selected.merge(
        split_rows[payload_columns],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if selected.empty:
        raise ValueError(f"no paired variants exist for {partition}")
    return selected.sort_values(["id", "variant"], ignore_index=True)


def audit_all_variant_alignment(
    *,
    original_hashes_csv: str | Path = AUDIT_DIR / "perceptual_hashes.csv.gz",
    high_resolution_catalogue_csv: str | Path = HIGH_RES_IMAGE_CATALOGUE_CSV,
    output_csv: str | Path = ALIGNMENT_AUDIT_CSV,
    summary_output: str | Path = ALIGNMENT_SUMMARY_JSON,
    root: str | Path = ROOT,
    workers: int | None = None,
    exception_distance: int = 4,
) -> dict[str, Any]:
    """Fingerprint every same-ID pair and report possible content mismatches."""
    started = time.monotonic()
    original = pd.read_csv(original_hashes_csv, keep_default_na=False)
    required_original = {"id", "dhash_u64", "ahash_u64"}
    if missing := required_original.difference(original.columns):
        raise ValueError(f"original hash audit lacks columns: {sorted(missing)}")
    original["id"] = original["id"].astype(int)
    if original["id"].duplicated().any():
        raise ValueError("original perceptual hash audit has duplicate IDs")

    high = pd.read_csv(high_resolution_catalogue_csv, keep_default_na=False)
    high = high[high["decode_ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    high["id"] = high["id"].astype(int)
    if high["id"].duplicated().any():
        raise ValueError("high-resolution image catalogue has duplicate IDs")
    root = Path(root)
    paths = [root / value for value in high["path"]]
    worker_count = workers or min(32, max(1, len(paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        hashes = list(pool.map(_compute_image_hashes, paths))
    high["high_dhash_u64"] = [value[0] for value in hashes]
    high["high_ahash_u64"] = [value[1] for value in hashes]

    pairs = original[["id", "dhash_u64", "ahash_u64"]].merge(
        high[
            [
                "id",
                "path",
                "width",
                "height",
                "aspect_ratio",
                "high_dhash_u64",
                "high_ahash_u64",
            ]
        ],
        on="id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    comparable = pairs[pairs["_merge"].eq("both")].copy()
    comparable["dhash_distance"] = [
        _hamming_distance(first, second)
        for first, second in zip(comparable["dhash_u64"], comparable["high_dhash_u64"], strict=True)
    ]
    comparable["ahash_distance"] = [
        _hamming_distance(first, second)
        for first, second in zip(comparable["ahash_u64"], comparable["high_ahash_u64"], strict=True)
    ]
    comparable["possible_content_mismatch"] = (
        comparable[["dhash_distance", "ahash_distance"]].max(axis=1) > exception_distance
    )
    comparable.sort_values("id", inplace=True, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        comparable,
        output_csv,
        index=False,
    )

    def distance_counts(column: str) -> dict[str, int]:
        counts = comparable[column].value_counts().sort_index()
        return {str(int(key)): int(value) for key, value in counts.items()}

    exceptions = comparable[comparable["possible_content_mismatch"]]
    summary: dict[str, Any] = {
        "runtime_seconds": time.monotonic() - started,
        "original_hashes_csv": {
            "path": _relative(Path(original_hashes_csv), root),
            "sha256": compute_sha256(original_hashes_csv),
        },
        "high_resolution_catalogue_csv": {
            "path": _relative(Path(high_resolution_catalogue_csv), root),
            "sha256": compute_sha256(high_resolution_catalogue_csv),
        },
        "pair_count": len(comparable),
        "original_only_count": int(pairs["_merge"].eq("left_only").sum()),
        "high_resolution_only_count": int(pairs["_merge"].eq("right_only").sum()),
        "dhash_distance_counts": distance_counts("dhash_distance"),
        "ahash_distance_counts": distance_counts("ahash_distance"),
        "exception_distance": exception_distance,
        "possible_content_mismatch_count": len(exceptions),
        "possible_content_mismatch_ids": exceptions["id"].astype(int).tolist(),
        "output_csv": _relative(output_csv, root),
        "output_sha256": compute_sha256(output_csv),
        "limitation": (
            "Perceptual hashes strongly test visual identity but are not a mathematical proof. "
            "High-distance same-ID pairs are non-blocking warnings for later inspection."
        ),
    }
    summary_output = Path(summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def audit_alignment_pairs(
    *,
    variants_csv: str | Path = TRAINING_VARIANTS_CSV,
    output_csv: str | Path,
    contact_sheet_path: str | Path,
    root: str | Path = ROOT,
    seed: int = RANDOM_SEED,
    sample_size: int = 12,
) -> pd.DataFrame:
    """Measure and render a seeded, label-free low/high alignment sample."""
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    train = variants[variants["partition"].eq("train")]
    paired = train.pivot(index="id", columns="variant", values="path").dropna()
    if not {"original", "high_resolution"}.issubset(paired.columns):
        raise ValueError("training variant manifest has no complete low/high pairs")
    generator = random.Random(seed)
    ids = sorted(generator.sample(sorted(paired.index.astype(int)), min(sample_size, len(paired))))
    root = Path(root)
    records: list[dict[str, Any]] = []
    thumbs: list[tuple[int, Image.Image, Image.Image]] = []
    for item_id in ids:
        low_path = root / paired.loc[item_id, "original"]
        high_path = root / paired.loc[item_id, "high_resolution"]
        low_hashes = _compute_image_hashes(low_path)
        high_hashes = _compute_image_hashes(high_path)
        metrics = _pair_pixel_metrics(low_path, high_path)
        with Image.open(low_path) as low_image, Image.open(high_path) as high_image:
            low_size = low_image.size
            high_size = high_image.size
            thumbs.append(
                (
                    item_id,
                    ImageOps.contain(ImageOps.exif_transpose(low_image).convert("RGB"), (180, 240)),
                    ImageOps.contain(
                        ImageOps.exif_transpose(high_image).convert("RGB"), (180, 240)
                    ),
                )
            )
        records.append(
            {
                "id": item_id,
                "low_path": _relative(low_path),
                "high_path": _relative(high_path),
                "low_width": low_size[0],
                "low_height": low_size[1],
                "high_width": high_size[0],
                "high_height": high_size[1],
                "dhash_distance": _hamming_distance(low_hashes[0], high_hashes[0]),
                "ahash_distance": _hamming_distance(low_hashes[1], high_hashes[1]),
                **metrics,
            }
        )
    result = pd.DataFrame(records)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        result,
        output_csv,
        index=False,
    )

    columns = 4
    cell_width, cell_height = 390, 300
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (item_id, low, high) in enumerate(thumbs):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(low, (x + 5, y + 25))
        sheet.paste(high, (x + 200, y + 25))
        draw.text(
            (x + 5, y + 5),
            f"ID {item_id}: original | high resolution",
            fill="black",
            font=font,
        )
    contact_sheet_path = Path(contact_sheet_path)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet_path)
    return result


def measure_alignment_warnings(
    *,
    alignment_csv: str | Path = ALIGNMENT_AUDIT_CSV,
    original_hashes_csv: str | Path = AUDIT_DIR / "perceptual_hashes.csv.gz",
    high_resolution_catalogue_csv: str | Path = HIGH_RES_IMAGE_CATALOGUE_CSV,
    output_csv: str | Path = ALIGNMENT_WARNING_METRICS_CSV,
    contact_sheet_path: str | Path = ALIGNMENT_WARNING_SHEET,
    root: str | Path = ROOT,
) -> pd.DataFrame:
    """Measure and render every non-blocking warning from the full fingerprint audit."""
    alignment = pd.read_csv(alignment_csv, keep_default_na=False)
    flagged = alignment[
        alignment["possible_content_mismatch"].astype(str).str.lower().isin({"true", "1"})
    ][["id", "dhash_distance", "ahash_distance"]].copy()
    original = pd.read_csv(original_hashes_csv, usecols=["id", "path"])
    original.rename(columns={"path": "original_path"}, inplace=True)
    high = pd.read_csv(high_resolution_catalogue_csv, usecols=["id", "path"])
    high.rename(columns={"path": "high_resolution_path"}, inplace=True)
    flagged = flagged.merge(original, on="id", validate="one_to_one").merge(
        high, on="id", validate="one_to_one"
    )
    root = Path(root)
    metrics = [
        _pair_pixel_metrics(root / row.original_path, root / row.high_resolution_path)
        for row in flagged.itertuples(index=False)
    ]
    result = pd.concat([flagged.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    result["maximum_hash_distance"] = result[["dhash_distance", "ahash_distance"]].max(axis=1)
    result.sort_values(
        ["maximum_hash_distance", "mse", "id"],
        ascending=[False, False, True],
        inplace=True,
        ignore_index=True,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(
        result,
        output_csv,
        index=False,
    )

    columns = 4
    cell_width, cell_height = 390, 300
    rows = (len(result) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, max(1, rows) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(result.itertuples(index=False)):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        with Image.open(root / row.original_path) as low_image:
            low = ImageOps.contain(ImageOps.exif_transpose(low_image).convert("RGB"), (180, 240))
        with Image.open(root / row.high_resolution_path) as high_image:
            high_thumb = ImageOps.contain(
                ImageOps.exif_transpose(high_image).convert("RGB"), (180, 240)
            )
        sheet.paste(low, (x + 5, y + 25))
        sheet.paste(high_thumb, (x + 200, y + 25))
        label = (
            f"ID {int(row.id)} | dH {int(row.dhash_distance)} | "
            f"aH {int(row.ahash_distance)} | MSE {row.mse:.4f}"
        )
        draw.text((x + 5, y + 5), label, fill="black", font=font)
    contact_sheet_path = Path(contact_sheet_path)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet_path)
    return result
