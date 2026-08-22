"""Train-only image normalization statistics."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fashion.config import (
    IMAGE_SIZE,
    ORIGINAL_ONLY_NORMALIZATION_JSON,
    PAD_COLOR,
    RANDOM_SEED,
    ROOT,
    SPLITS_CSV,
    TEST_CSV,
)
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.images import ImageSize, resolve_image_size, transform_image_with_mask


def train_ids_digest(ids: list[int]) -> str:
    """Hash a sorted train-ID list for normalization provenance."""
    encoded = ",".join(map(str, sorted(ids))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _image_moments(
    task: tuple[Path, ImageSize, tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    path, image_size, pad_color = task
    with Image.open(path) as source:
        image, content_mask = transform_image_with_mask(
            source, image_size=image_size, pad_color=pad_color
        )
    pixels = image[content_mask].reshape(-1, 3).astype(np.float64)
    return (
        pixels.sum(axis=0),
        np.square(pixels).sum(axis=0),
        len(pixels),
        int(content_mask.size),
    )


def compute_normalization_stats(
    splits_csv: str | Path = SPLITS_CSV,
    output_path: str | Path = ORIGINAL_ONLY_NORMALIZATION_JSON,
    root: str | Path = ROOT,
    image_size: ImageSize = IMAGE_SIZE,
    pad_color: tuple[int, int, int] = PAD_COLOR,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
) -> dict[str, object]:
    """Fit the original-image-only comparison baseline on train."""
    splits = load_splits(splits_csv)
    train = splits[splits["partition"].eq("train")].copy()
    if train.empty:
        raise ValueError("splits.csv contains no training rows")
    if not train["partition"].eq("train").all():
        raise ValueError("normalization scope contains a non-training row")
    tasks = [
        (Path(root) / Path(path), image_size, pad_color) for path in train.sort_values("id")["path"]
    ]
    workers = workers or min(32, (os.cpu_count() or 4) * 4)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    total_output_pixels = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for image_sum, image_sum_sq, pixel_count, output_pixels in executor.map(
            _image_moments, tasks
        ):
            channel_sum += image_sum
            channel_sum_sq += image_sum_sq
            total_pixels += pixel_count
            total_output_pixels += output_pixels
    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sum_sq / total_pixels - np.square(mean), 0.0)
    output_height, output_width = resolve_image_size(image_size)
    padding_pixels = total_output_pixels - total_pixels
    stats: dict[str, object] = {
        "schema_version": "2.0.0",
        "policy": "original_resolution_only_baseline",
        "image_size": [output_height, output_width],
        "pad_color": list(pad_color),
        "mean": [round(float(value), 6) for value in mean],
        "std": [round(float(value), 6) for value in np.sqrt(variance)],
        "num_images": len(train),
        "total_content_pixels": total_pixels,
        "total_output_pixels": total_output_pixels,
        "padding_pixels": padding_pixels,
        "padding_share": float(padding_pixels / total_output_pixels),
        "pixel_scope": "resized image content only; letterbox padding excluded",
        "padding_after_standardization": 0.0,
        "source_partition": "train",
        "seed": seed,
        "train_ids_digest": train_ids_digest(train["id"].astype(int).tolist()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def compute_paired_normalization_stats(
    *,
    variants_csv: str | Path,
    variants_summary_json: str | Path,
    splits_csv: str | Path = SPLITS_CSV,
    official_prediction_csv: str | Path | None = TEST_CSV,
    trusted_prediction_manifest_csv: str | Path | None = None,
    output_path: str | Path,
    root: str | Path = ROOT,
    image_size: ImageSize = IMAGE_SIZE,
    pad_color: tuple[int, int, int] = PAD_COLOR,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
) -> dict[str, object]:
    """Fit train-only RGB statistics with each resolution weighted 0.5 per product."""
    from fashion.data.variants import load_image_variants

    train = load_image_variants(
        partition="train",
        variants_csv=variants_csv,
        summary_json=variants_summary_json,
        splits_csv=splits_csv,
        official_prediction_csv=official_prediction_csv,
        trusted_prediction_manifest_csv=trusted_prediction_manifest_csv,
        root=root,
    )
    if train.empty or not train["partition"].eq("train").all():
        raise ValueError("paired normalization scope must contain training rows only")
    weights = pd.to_numeric(train["per_product_weight"], errors="raise")
    if not weights.eq(0.5).all() or not weights.groupby(train["id"]).sum().eq(1.0).all():
        raise ValueError("paired normalization requires two half-weight variants per product")
    ordered = train.sort_values(["id", "variant"])
    tasks = [
        (Path(root) / Path(path), image_size, pad_color) for path in ordered["path"].astype(str)
    ]
    ordered_weights = ordered["per_product_weight"].astype(float).tolist()
    workers = workers or min(32, (os.cpu_count() or 4) * 4)
    weighted_image_mean = np.zeros(3, dtype=np.float64)
    weighted_image_second_moment = np.zeros(3, dtype=np.float64)
    total_effective_weight = 0.0
    weighted_content_pixels = 0.0
    weighted_output_pixels = 0.0
    raw_content_pixels = 0
    raw_output_pixels = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        moments = executor.map(_image_moments, tasks)
        for weight, (image_sum, image_sum_sq, pixel_count, output_pixels) in zip(
            ordered_weights, moments, strict=True
        ):
            weighted_image_mean += weight * image_sum / pixel_count
            weighted_image_second_moment += weight * image_sum_sq / pixel_count
            total_effective_weight += weight
            weighted_content_pixels += weight * pixel_count
            weighted_output_pixels += weight * output_pixels
            raw_content_pixels += pixel_count
            raw_output_pixels += output_pixels
    mean = weighted_image_mean / total_effective_weight
    second_moment = weighted_image_second_moment / total_effective_weight
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    output_height, output_width = resolve_image_size(image_size)
    weighted_padding = weighted_output_pixels - weighted_content_pixels
    train_ids = ordered["id"].drop_duplicates().astype(int).tolist()
    stats: dict[str, object] = {
        "schema_version": "3.0.0",
        "policy": "pair_weighted_original_and_high_resolution",
        "source_partition": "train",
        "source_variants": ["original", "high_resolution"],
        "variant_weight": 0.5,
        "product_count": len(train_ids),
        "variant_count": len(ordered),
        "total_effective_product_weight": total_effective_weight,
        "image_size": [output_height, output_width],
        "pad_color": list(pad_color),
        "mean": [round(float(value), 6) for value in mean],
        "std": [round(float(value), 6) for value in np.sqrt(variance)],
        "weighted_content_pixels": weighted_content_pixels,
        "weighted_output_pixels": weighted_output_pixels,
        "weighted_padding_pixels": weighted_padding,
        "weighted_padding_share": float(weighted_padding / weighted_output_pixels),
        "raw_content_pixels": raw_content_pixels,
        "raw_output_pixels": raw_output_pixels,
        "pixel_scope": (
            "train variants only; each image's content moments receive its 0.5 variant weight; "
            "letterbox padding excluded"
        ),
        "padding_after_standardization": 0.0,
        "seed": seed,
        "train_ids_digest": train_ids_digest(train_ids),
        "variant_manifest_sha256": compute_sha256(variants_csv),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def refresh_paired_normalization_provenance(
    *,
    variants_csv: str | Path,
    splits_csv: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Refresh manifest provenance without recomputing unchanged train pixels."""
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    splits = load_splits(splits_csv)
    train_ids = sorted(splits.loc[splits["partition"].eq("train"), "id"].astype(int))
    train = variants[variants["partition"].eq("train")].copy()
    if sorted(train["id"].drop_duplicates().astype(int)) != train_ids:
        raise ValueError("paired-normalization train IDs differ from the fixed split")
    if not train.groupby("id")["variant"].agg(set).eq({"original", "high_resolution"}).all():
        raise ValueError("paired-normalization provenance requires complete variant pairs")
    weights = pd.to_numeric(train["per_product_weight"], errors="raise")
    if not weights.eq(0.5).all() or not weights.groupby(train["id"]).sum().eq(1.0).all():
        raise ValueError("paired-normalization provenance requires half-weight pairs")
    output = Path(output_path)
    stats = json.loads(output.read_text(encoding="utf-8"))
    if stats.get("policy") != "pair_weighted_original_and_high_resolution":
        raise ValueError("paired-normalization policy changed; recompute pixels in full mode")
    if stats.get("product_count") != len(train_ids) or stats.get("variant_count") != len(train):
        raise ValueError("paired-normalization row counts changed; recompute pixels in full mode")
    if stats.get("train_ids_digest") != train_ids_digest(train_ids):
        raise ValueError("paired-normalization train IDs changed; recompute pixels in full mode")
    stats["variant_manifest_sha256"] = compute_sha256(variants_csv)
    output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats
