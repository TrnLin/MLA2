"""Image-only overlap checks and deterministic source-family links for external data."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.data.external_usage import decoded_sha256, file_sha256
from fashion.data.families import normalize_product_name
from fashion.data.images import transform_image_with_mask
from fashion.data.perceptual import (
    FOREGROUND_WHITE_THRESHOLD,
    compute_pair_pixel_metrics,
    meets_near_duplicate_rule,
)


def _perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    """Same dHash/aHash arithmetic as the canonical file-based implementation."""
    gray = ImageOps.exif_transpose(image).convert("L")
    diff = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int32)
    average = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    return (
        np.packbits((diff[:, 1:] > diff[:, :-1]).ravel()).tobytes().hex(),
        np.packbits((average >= average.mean()).ravel()).tobytes().hex(),
    )


def fingerprint_image(path: str | Path) -> dict[str, Any]:
    """Fingerprint raw pixels and the exact final CNN view, without reading labels."""
    with Image.open(path) as im:
        im.load()
        original = ImageOps.exif_transpose(im).convert("RGB")
        array, _ = transform_image_with_mask(original, image_size=(80, 60), normalize_range=False)
        view = Image.fromarray(array.astype(np.uint8))
        crop = _foreground_image(original)
        dhash, ahash = _perceptual_hashes(original)
        view_dhash, view_ahash = _perceptual_hashes(view)
        crop_dhash, crop_ahash = _perceptual_hashes(crop)
        return {
            "file_sha256": file_sha256(path),
            "pixel_sha256": decoded_sha256(original),
            "view_pixel_sha256": decoded_sha256(view),
            "crop_pixel_sha256": decoded_sha256(crop),
            "dhash_hex": dhash,
            "ahash_hex": ahash,
            "view_dhash_hex": view_dhash,
            "view_ahash_hex": view_ahash,
            "crop_dhash_hex": crop_dhash,
            "crop_ahash_hex": crop_ahash,
            "original_width": original.width,
            "original_height": original.height,
        }


def _foreground_image(image: Image.Image) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    values = np.asarray(rgb)
    foreground = np.any(values < FOREGROUND_WHITE_THRESHOLD, axis=2)
    if not foreground.any():
        return rgb
    ys, xs = np.where(foreground)
    return rgb.crop((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))


def _cropped_path(path: Path, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (file_sha256(path) + "_foreground.png")
    if not dest.exists():
        with Image.open(path) as image:
            _foreground_image(image).save(dest)
    return dest


def _normalized_path(path: Path, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (file_sha256(path) + ".png")
    if not dest.exists():
        with Image.open(path) as im:
            array, _ = transform_image_with_mask(im, image_size=(80, 60), normalize_range=False)
        Image.fromarray(array.astype(np.uint8)).save(dest)
    return dest


def find_teacher_overlaps(
    candidates: pd.DataFrame, references: pd.DataFrame, *, root: str | Path, cache: str | Path
) -> pd.DataFrame:
    """Compare every external view to every supplied teacher role; preserve rejected signals."""
    root, cache = Path(root), Path(cache)
    rows = []
    hashes = {
        key: np.array([int(v, 16) for v in references[key]], dtype=np.uint64)
        for prefix in ("", "view_", "crop_")
        for key in (prefix + "dhash_hex", prefix + "ahash_hex")
    }
    identity_maps = {}
    for key in ("file_sha256", "pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256"):
        values = defaultdict(set)
        for pos, value in enumerate(references[key]):
            values[value].add(pos)
        identity_maps[key] = values
    for _, candidate in candidates.iterrows():
        identities = defaultdict(list)
        for source_key, ref_keys in (
            ("file_sha256", ("file_sha256",)),
            ("pixel_sha256", ("pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256")),
            ("view_pixel_sha256", ("pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256")),
            ("crop_pixel_sha256", ("pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256")),
        ):
            for ref_key in ref_keys:
                for pos in identity_maps[ref_key].get(candidate[source_key], set()):
                    identities[pos].append(f"{source_key}={ref_key}")
        signals = {}
        for source_prefix in ("", "view_", "crop_"):
            for ref_prefix in ("", "view_", "crop_"):
                distances = np.bitwise_count(
                    hashes[ref_prefix + "dhash_hex"]
                    ^ np.uint64(int(candidate[source_prefix + "dhash_hex"], 16))
                )
                for pos in np.flatnonzero(distances <= 2):
                    pos = int(pos)
                    distance = int(distances[pos])
                    average_distance = (
                        int(candidate[source_prefix + "ahash_hex"], 16)
                        ^ int(references.iloc[pos][ref_prefix + "ahash_hex"], 16)
                    ).bit_count()
                    signals[(pos, source_prefix, ref_prefix)] = (distance, average_distance)
        seen_identical = set()
        for (pos, source_prefix, ref_prefix), (distance, average_distance) in signals.items():
            ref = references.iloc[pos]
            exact = bool(identities.get(pos))
            first, second = root / candidate.original_path, root / ref.path
            if source_prefix == "view_":
                first = _normalized_path(first, cache)
            elif source_prefix == "crop_":
                first = _cropped_path(first, cache)
            if ref_prefix == "view_":
                second = _normalized_path(second, cache)
            elif ref_prefix == "crop_":
                second = _cropped_path(second, cache)
            metrics = compute_pair_pixel_metrics(first, second)
            accepted = exact or meets_near_duplicate_rule(
                SimpleNamespace(dhash_distance=distance, ahash_distance=average_distance, **metrics)
            )
            rows.append(
                {
                    "external_id": candidate.external_id,
                    "teacher_id": str(ref.id),
                    "teacher_role": ref.audit_role,
                    "teacher_path": ref.path,
                    "external_view": source_prefix or "original",
                    "teacher_view": ref_prefix or "original",
                    "dhash_distance": distance,
                    "ahash_distance": average_distance,
                    "identity_evidence": ";".join(identities.get(pos, [])),
                    "accepted_overlap": bool(accepted),
                    **metrics,
                }
            )
            seen_identical.add(pos)
        for pos in set(identities) - seen_identical:
            ref = references.iloc[pos]
            rows.append(
                {
                    "external_id": candidate.external_id,
                    "teacher_id": str(ref.id),
                    "teacher_role": ref.audit_role,
                    "teacher_path": ref.path,
                    "external_view": "identity",
                    "teacher_view": "identity",
                    "identity_evidence": ";".join(identities[pos]),
                    "accepted_overlap": True,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "external_id",
            "teacher_id",
            "teacher_role",
            "teacher_path",
            "external_view",
            "teacher_view",
            "dhash_distance",
            "ahash_distance",
            "identity_evidence",
            "accepted_overlap",
            "mse",
            "mae",
            "max_difference",
            "crop_mse",
            "crop_mae",
            "foreground_fraction_1",
            "foreground_fraction_2",
            "foreground_ratio",
        ],
    )


def link_external_families(
    frame: pd.DataFrame, *, root: str | Path, cache: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link source identities, normalized names, pixels and accepted near matches."""
    result = frame.copy().reset_index(drop=True)
    cache = Path(cache) if cache else Path(root) / "data/external/.comparison_views"
    parent = list(range(len(result)))
    edges = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b, reason):
        first, second = find(a), find(b)
        if first != second:
            parent[max(first, second)] = min(first, second)
        edges.append(
            {
                "first_id": result.iloc[a].external_id,
                "second_id": result.iloc[b].external_id,
                "reason": reason,
            }
        )

    buckets = defaultdict(list)
    for index, row in result.iterrows():
        title = normalize_product_name(str(row.source_title))
        url = urlparse(str(row.image_url))
        image_key = (
            url.path
            if url.hostname and url.hostname.endswith("flixcart.com")
            else str(row.image_url)
        )
        keys = [
            ("source_group", str(row.source_group_id)),
            ("image_url", image_key),
            ("source_product", str(row.source) + ":" + str(row.source_id)),
        ]
        if len(title) >= 12:
            keys.append(("normalized_title", title))
        if "amazon" in str(row.source).lower() or "abo" in str(row.source).lower():
            keys.append(("amazon_asin", str(row.source_id)))
        for key in ("file_sha256", "pixel_sha256", "view_pixel_sha256", "crop_pixel_sha256"):
            if str(row.get(key, "")):
                keys.append((key, str(row[key])))
        for key in keys:
            if key[1]:
                buckets[key].append(index)
    for (kind, _), indices in buckets.items():
        for index in indices[1:]:
            union(indices[0], index, kind)
    for first in range(len(result)):
        a = result.iloc[first]
        for second in range(first + 1, len(result)):
            b = result.iloc[second]
            if find(first) == find(second):
                continue
            dh = (int(a.dhash_hex, 16) ^ int(b.dhash_hex, 16)).bit_count()
            ah = (int(a.ahash_hex, 16) ^ int(b.ahash_hex, 16)).bit_count()
            if dh <= 2 and ah <= 1:
                metrics = compute_pair_pixel_metrics(
                    Path(root) / a.original_path, Path(root) / b.original_path
                )
                if meets_near_duplicate_rule(
                    SimpleNamespace(dhash_distance=dh, ahash_distance=ah, **metrics)
                ):
                    union(first, second, "accepted_original_near_duplicate")
            if (
                find(first) != find(second)
                and (int(a.view_dhash_hex, 16) ^ int(b.view_dhash_hex, 16)).bit_count() <= 2
            ):
                # Conservative final-view grouping prevents close variants crossing the split.
                view_ah = (int(a.view_ahash_hex, 16) ^ int(b.view_ahash_hex, 16)).bit_count()
                if view_ah <= 1:
                    union(first, second, "conservative_final_view_hash_group")
            crop_dh = (int(a.crop_dhash_hex, 16) ^ int(b.crop_dhash_hex, 16)).bit_count()
            crop_ah = (int(a.crop_ahash_hex, 16) ^ int(b.crop_ahash_hex, 16)).bit_count()
            if find(first) != find(second) and crop_dh <= 2 and crop_ah <= 1:
                metrics = compute_pair_pixel_metrics(
                    _cropped_path(Path(root) / a.original_path, cache),
                    _cropped_path(Path(root) / b.original_path, cache),
                )
                if meets_near_duplicate_rule(
                    SimpleNamespace(dhash_distance=crop_dh, ahash_distance=crop_ah, **metrics)
                ):
                    union(first, second, "accepted_foreground_near_duplicate")
    groups = defaultdict(list)
    for i, row in result.iterrows():
        groups[find(i)].append(row.external_id)
    names = {
        root_index: "external_family_"
        + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:20]
        for root_index, ids in groups.items()
    }
    result["external_group_id"] = [names[find(i)] for i in range(len(result))]
    return result, pd.DataFrame(edges, columns=["first_id", "second_id", "reason"])
