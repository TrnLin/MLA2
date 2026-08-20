"""Train-only image normalization statistics."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fashion.config import (
    IMAGE_SIZE,
    NORMALIZATION_JSON,
    PAD_COLOR,
    RANDOM_SEED,
    ROOT,
    SPLITS_CSV,
)
from fashion.data.dataset import load_splits
from fashion.data.images import load_and_transform_image


def train_ids_digest(ids: list[int]) -> str:
    """Hash a sorted train-ID list for normalization provenance."""
    encoded = ",".join(map(str, sorted(ids))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _image_moments(
    task: tuple[Path, int, tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    path, image_size, pad_color = task
    image = load_and_transform_image(path, image_size=image_size, pad_color=pad_color)
    pixels = image.reshape(-1, 3).astype(np.float64)
    return pixels.sum(axis=0), np.square(pixels).sum(axis=0), len(pixels)


def compute_normalization_stats(
    splits_csv: str | Path = SPLITS_CSV,
    output_path: str | Path = NORMALIZATION_JSON,
    root: str | Path = ROOT,
    image_size: int = IMAGE_SIZE,
    pad_color: tuple[int, int, int] = PAD_COLOR,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
) -> dict[str, object]:
    """Fit RGB mean/std on the training partition only."""
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
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for image_sum, image_sum_sq, pixel_count in executor.map(_image_moments, tasks):
            channel_sum += image_sum
            channel_sum_sq += image_sum_sq
            total_pixels += pixel_count
    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sum_sq / total_pixels - np.square(mean), 0.0)
    stats: dict[str, object] = {
        "schema_version": "1.0.0",
        "image_size": image_size,
        "pad_color": list(pad_color),
        "mean": [round(float(value), 6) for value in mean],
        "std": [round(float(value), 6) for value in np.sqrt(variance)],
        "num_images": len(train),
        "total_pixels": total_pixels,
        "source_partition": "train",
        "seed": seed,
        "train_ids_digest": train_ids_digest(train["id"].astype(int).tolist()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
