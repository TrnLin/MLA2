"""Shared label-free perceptual duplicate triage for canonical data preparation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.config import (
    AUDIT_DIR,
    PREDICTION_MANIFEST_CSV,
    ROOT,
)
from fashion.data.hashing import write_deterministic_csv

PIXEL_METRIC_SIZE = 64
FOREGROUND_WHITE_THRESHOLD = 245
NEAR_DUPLICATE_RULE = {
    "maximum_dhash_distance": 2,
    "maximum_ahash_distance": 1,
    "maximum_canvas_mse": 0.0005,
    "maximum_canvas_mae": 0.01,
    "maximum_crop_mse": 0.005,
    "maximum_crop_mae": 0.05,
    "minimum_foreground_ratio": 0.8,
}


def compute_image_hashes(path: str | Path) -> tuple[int, int]:
    """Compute 64-bit difference and average hashes."""
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        difference_array = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int32)
        difference_bits = (difference_array[:, 1:] > difference_array[:, :-1]).ravel()
        average_array = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
        average_bits = (average_array >= average_array.mean()).ravel()
    return (
        int.from_bytes(np.packbits(difference_bits).tobytes(), "big"),
        int.from_bytes(np.packbits(average_bits).tobytes(), "big"),
    )


def hamming_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def find_hamming_candidates_multi_index(
    hashes: list[int] | np.ndarray,
    max_distance: int = 3,
    chunks: int = 4,
) -> dict[int, list[tuple[int, int]]]:
    """Find all close 64-bit pairs with exact pigeonhole coverage."""
    if chunks < max_distance + 1 or 64 % chunks:
        raise ValueError("chunks must divide 64 and be greater than max_distance")
    chunk_bits = 64 // chunks
    mask = (1 << chunk_bits) - 1
    tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(chunks)]
    for index, hash_value in enumerate(hashes):
        for chunk in range(chunks):
            shift = (chunks - chunk - 1) * chunk_bits
            tables[chunk][(int(hash_value) >> shift) & mask].append(index)

    checked: set[tuple[int, int]] = set()
    matches: dict[int, list[tuple[int, int]]] = {
        distance: [] for distance in range(max_distance + 1)
    }
    for table in tables:
        for indices in table.values():
            for position, first in enumerate(indices):
                for second in indices[position + 1 :]:
                    pair = (min(first, second), max(first, second))
                    if pair in checked:
                        continue
                    checked.add(pair)
                    distance = hamming_distance(hashes[pair[0]], hashes[pair[1]])
                    if distance <= max_distance:
                        matches[distance].append(pair)
    return matches


def _foreground_crop(array: np.ndarray) -> tuple[np.ndarray, float]:
    foreground = np.any(array < FOREGROUND_WHITE_THRESHOLD, axis=2)
    fraction = float(foreground.mean())
    if not foreground.any():
        return array, fraction
    rows, columns = np.where(foreground)
    return (
        array[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1],
        fraction,
    )


def compute_pair_pixel_metrics(
    first_path: str | Path,
    second_path: str | Path,
    size: int = PIXEL_METRIC_SIZE,
) -> dict[str, float]:
    """Measure full-canvas and foreground-crop differences for one diagnostic pair."""
    arrays: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    foreground_fractions: list[float] = []
    for path in (first_path, second_path):
        with Image.open(path) as image:
            source = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        crop, foreground_fraction = _foreground_crop(source)
        arrays.append(
            np.asarray(
                Image.fromarray(source).resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            / 255
        )
        crops.append(
            np.asarray(
                Image.fromarray(crop).resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            / 255
        )
        foreground_fractions.append(foreground_fraction)

    canvas_difference = np.abs(arrays[0] - arrays[1])
    crop_difference = np.abs(crops[0] - crops[1])
    maximum_fraction = max(foreground_fractions)
    foreground_ratio = min(foreground_fractions) / maximum_fraction if maximum_fraction > 0 else 1.0
    return {
        "mse": float(np.square(arrays[0] - arrays[1]).mean()),
        "mae": float(canvas_difference.mean()),
        "max_difference": float(canvas_difference.max()),
        "crop_mse": float(np.square(crops[0] - crops[1]).mean()),
        "crop_mae": float(crop_difference.mean()),
        "foreground_fraction_1": foreground_fractions[0],
        "foreground_fraction_2": foreground_fractions[1],
        "foreground_ratio": foreground_ratio,
    }


def meets_near_duplicate_rule(row: pd.Series | Any) -> bool:
    """Apply the frozen automatic pixel-and-hash acceptance rule."""
    return bool(
        int(row.dhash_distance) <= NEAR_DUPLICATE_RULE["maximum_dhash_distance"]
        and int(row.ahash_distance) <= NEAR_DUPLICATE_RULE["maximum_ahash_distance"]
        and float(row.mse) <= NEAR_DUPLICATE_RULE["maximum_canvas_mse"]
        and float(row.mae) <= NEAR_DUPLICATE_RULE["maximum_canvas_mae"]
        and float(row.crop_mse) <= NEAR_DUPLICATE_RULE["maximum_crop_mse"]
        and float(row.crop_mae) <= NEAR_DUPLICATE_RULE["maximum_crop_mae"]
        and float(row.foreground_ratio) >= NEAR_DUPLICATE_RULE["minimum_foreground_ratio"]
    )


def run_perceptual_audit(
    train_manifest_csv: str | Path | None = None,
    prediction_manifest_csv: str | Path = PREDICTION_MANIFEST_CSV,
    output_dir: str | Path = AUDIT_DIR,
    root: str | Path = ROOT,
    max_distance: int = 2,
    workers: int | None = None,
) -> dict[str, Any]:
    """Generate pixel-scored candidates before the sole split is built."""
    if train_manifest_csv is None:
        raise ValueError("the labelled manifest must be an explicit temporary build path")
    labelled = pd.read_csv(train_manifest_csv, keep_default_na=False)[
        ["id", "path", "sha256"]
    ].copy()
    labelled["role"] = "labelled"
    prediction = pd.read_csv(prediction_manifest_csv, keep_default_na=False)[
        ["id", "path", "sha256"]
    ].copy()
    prediction["role"] = "prediction"
    inventory = pd.concat([labelled, prediction], ignore_index=True).sort_values(["role", "id"])
    inventory.reset_index(drop=True, inplace=True)
    root = Path(root)

    def hash_row(row: dict[str, Any]) -> dict[str, Any]:
        difference_hash, average_hash = compute_image_hashes(root / row["path"])
        return {
            **row,
            "dhash_u64": difference_hash,
            "ahash_u64": average_hash,
            "dhash_hex": f"{difference_hash:016x}",
            "ahash_hex": f"{average_hash:016x}",
        }

    worker_count = workers or min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        hashed = list(executor.map(hash_row, inventory.to_dict("records")))
    hashes = pd.DataFrame(hashed)
    pairs_by_distance = find_hamming_candidates_multi_index(
        hashes["dhash_u64"].to_numpy(), max_distance=max_distance
    )
    pair_rows: list[dict[str, Any]] = []
    pair_paths: list[tuple[Path, Path]] = []
    for distance, pairs in pairs_by_distance.items():
        for first_index, second_index in pairs:
            first = hashes.iloc[first_index]
            second = hashes.iloc[second_index]
            pair_rows.append(
                {
                    "id_1": int(first["id"]),
                    "id_2": int(second["id"]),
                    "role_1": str(first["role"]),
                    "role_2": str(second["role"]),
                    "dhash_distance": distance,
                    "ahash_distance": hamming_distance(first["ahash_u64"], second["ahash_u64"]),
                    "is_exact_sha256": first["sha256"] == second["sha256"],
                    "cross_role": first["role"] != second["role"],
                }
            )
            pair_paths.append((root / first["path"], root / second["path"]))

    def pixel_metrics(paths: tuple[Path, Path]) -> dict[str, float]:
        return compute_pair_pixel_metrics(paths[0], paths[1])

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        metrics = list(executor.map(pixel_metrics, pair_paths))
    candidates = pd.concat([pd.DataFrame(pair_rows), pd.DataFrame(metrics)], axis=1)
    candidates["meets_automatic_rule"] = candidates.apply(meets_near_duplicate_rule, axis=1)

    decisions: list[str] = []
    for row in candidates.itertuples(index=False):
        if row.is_exact_sha256:
            decisions.append("accepted_exact_sha256")
            continue
        if not row.meets_automatic_rule:
            decisions.append("rejected_metrics")
            continue
        if not row.cross_role:
            decisions.append("accepted_automatic")
            continue
        decisions.append("quarantine_cross_role_automatic_match")
    candidates["decision"] = decisions
    candidates["accepted_near_duplicate"] = candidates["decision"].isin(
        {
            "accepted_exact_sha256",
            "accepted_automatic",
            "quarantine_cross_role_automatic_match",
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = output_dir / "perceptual_hashes.csv.gz"
    candidates_path = output_dir / "near_duplicate_candidates.csv.gz"
    write_deterministic_csv(
        hashes,
        hashes_path,
        index=False,
    )
    write_deterministic_csv(
        candidates,
        candidates_path,
        index=False,
        float_format="%.8f",
    )
    accepted = candidates[candidates["accepted_near_duplicate"]]
    cross_auto = candidates[
        candidates["cross_role"]
        & candidates["meets_automatic_rule"]
        & ~candidates["is_exact_sha256"]
    ]
    summary = {
        "schema_version": "5.0.0",
        "scope": "label-free audit over labelled and official prediction images before splitting",
        "total_images": len(hashes),
        "candidate_counts_by_dhash_distance": {
            str(distance): len(pairs) for distance, pairs in pairs_by_distance.items()
        },
        "candidate_pairs": len(candidates),
        "pixel_metric_size": PIXEL_METRIC_SIZE,
        "foreground_white_threshold": FOREGROUND_WHITE_THRESHOLD,
        "automatic_rule": NEAR_DUPLICATE_RULE,
        "accepted_pairs": len(accepted),
        "accepted_non_exact_pairs": int((~accepted["is_exact_sha256"]).sum()),
        "accepted_cross_role_pairs": int(accepted["cross_role"].sum()),
        "cross_role_automatic_matches": {
            "pair_count": len(cross_auto),
            "labelled_ids_conservatively_quarantined": int(cross_auto["id_1"].nunique()),
            "prediction_ids_never_entering_development": int(cross_auto["id_2"].nunique()),
            "safety_policy": (
                "Every non-exact cross-role automatic match is quarantined. Human input is "
                "not read and cannot block preparation, cache validation, or testing."
            ),
        },
        "warning": (
            "The automatic rule favours precision over recall. Rejected candidates remain "
            "triage signals, not proof that two products differ."
        ),
    }
    (output_dir / "near_duplicate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
