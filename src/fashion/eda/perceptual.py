"""Label-free perceptual duplicate diagnostics."""

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

from fashion.config import AUDIT_DIR, PREDICTION_MANIFEST_CSV, RANDOM_SEED, ROOT, SPLITS_CSV
from fashion.data.dataset import load_splits


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


def compute_pair_pixel_metrics(
    first_path: str | Path,
    second_path: str | Path,
    size: int = 128,
) -> dict[str, float]:
    """Return normalized RGB differences for a diagnostic pair."""
    with Image.open(first_path) as first, Image.open(second_path) as second:
        first_array = (
            np.asarray(
                ImageOps.exif_transpose(first)
                .convert("RGB")
                .resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            / 255
        )
        second_array = (
            np.asarray(
                ImageOps.exif_transpose(second)
                .convert("RGB")
                .resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            / 255
        )
    difference = np.abs(first_array - second_array)
    return {
        "mse": float(np.square(first_array - second_array).mean()),
        "mae": float(difference.mean()),
        "max_difference": float(difference.max()),
    }


def run_perceptual_audit(
    splits_csv: str | Path = SPLITS_CSV,
    prediction_manifest_csv: str | Path = PREDICTION_MANIFEST_CSV,
    output_dir: str | Path = AUDIT_DIR,
    root: str | Path = ROOT,
    max_distance: int = 2,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build label-free candidate evidence without changing the shared split."""
    labelled = load_splits(splits_csv)[["id", "partition", "path", "sha256"]].copy()
    labelled["role"] = "labelled"
    prediction = pd.read_csv(prediction_manifest_csv)[["id", "path", "sha256"]].copy()
    prediction["partition"] = "prediction"
    prediction["role"] = "prediction"
    inventory = pd.concat([labelled, prediction], ignore_index=True)
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

    records = inventory.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers or min(32, (os.cpu_count() or 4) * 4)) as executor:
        hashed = list(executor.map(hash_row, records))
    hashes = pd.DataFrame(hashed).sort_values(["role", "id"]).reset_index(drop=True)
    candidates = find_hamming_candidates_multi_index(
        hashes["dhash_u64"].to_numpy(), max_distance=max_distance
    )
    candidate_rows: list[dict[str, Any]] = []
    for distance, pairs in candidates.items():
        for first_index, second_index in pairs:
            first = hashes.iloc[first_index]
            second = hashes.iloc[second_index]
            candidate_rows.append(
                {
                    "id_1": int(first["id"]),
                    "id_2": int(second["id"]),
                    "partition_1": first["partition"],
                    "partition_2": second["partition"],
                    "dhash_distance": distance,
                    "ahash_distance": hamming_distance(first["ahash_u64"], second["ahash_u64"]),
                    "is_exact_sha256": first["sha256"] == second["sha256"],
                    "crosses_partition": first["partition"] != second["partition"],
                }
            )
    candidate_frame = pd.DataFrame(candidate_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes.to_csv(output_dir / "perceptual_hashes.csv", index=False)
    candidate_frame.to_csv(output_dir / "near_duplicate_candidates.csv", index=False)
    summary = {
        "schema_version": "1.0.0",
        "seed": seed,
        "scope": "label-free image diagnostics across labelled and official prediction roles",
        "total_images": len(hashes),
        "candidate_counts_by_dhash_distance": {
            str(distance): len(pairs) for distance, pairs in candidates.items()
        },
        "cross_partition_candidates": int(candidate_frame["crosses_partition"].sum())
        if not candidate_frame.empty
        else 0,
        "split_changed": False,
        "warning": "perceptual collisions are candidates, not automatic duplicate groups",
    }
    (output_dir / "near_duplicate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
