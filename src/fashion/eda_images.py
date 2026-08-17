"""Deterministic image measurements and duplicate-review candidates for EDA."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from fashion.data_audit import dhash, hamming_distance


DEFAULT_SAMPLE_SEED = 2753
MAX_NEAR_DUPLICATE_ITEMS = 2048
IMAGE_METRICS = (
    "brightness",
    "contrast",
    "colorfulness",
    "saturation",
    "edge_sharpness",
)
MEASUREMENT_COLUMNS = (
    "path",
    "width",
    "height",
    "aspect_ratio",
    "mode",
    "file_bytes",
    "file_sha256",
    "byte_sha256",
    "pixel_sha256",
    "dhash",
    *IMAGE_METRICS,
    "error",
)


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, Integral) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")


def _integer_ids(values: pd.Series) -> pd.Series:
    """Normalize product IDs without silently changing fractional identifiers."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("id values must be integers")
    return numeric.astype("int64")


def stratified_sample(
    frame: pd.DataFrame,
    column: str,
    limit: int,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> pd.DataFrame:
    """Return an ID-sorted, seeded proportional sample with rare-class coverage."""
    _validate_limit(limit)
    if column not in frame:
        raise ValueError(f"Column not found: {column}")
    if "id" not in frame:
        raise ValueError("Column not found: id")
    result = frame.copy()
    result["id"] = _integer_ids(result["id"])
    if result.empty or limit == 0:
        return result.iloc[0:0].sort_values("id", ignore_index=True)

    result["_stratum"] = result[column].astype("string").fillna("")
    result = result.sort_values(["_stratum", "id"], kind="stable").reset_index(drop=True)
    counts = result["_stratum"].value_counts(sort=False).sort_index()
    target_size = min(int(limit), len(result))
    if target_size == len(result):
        return result.drop(columns="_stratum").sort_values("id", ignore_index=True)
    class_count = len(counts)
    if target_size >= class_count:
        allocations = pd.Series(1, index=counts.index, dtype="int64")
        remaining = target_size - class_count
    else:
        allocations = pd.Series(0, index=counts.index, dtype="int64")
        remaining = target_size

    if remaining:
        shares = counts / counts.sum() * remaining
        capacity = counts - allocations
        whole = np.minimum(np.floor(shares).astype("int64"), capacity)
        allocations = allocations.add(whole, fill_value=0).astype("int64")
        leftovers = remaining - int(whole.sum())
        while leftovers:
            eligible = capacity.index[allocations.lt(counts)]
            for stratum in sorted(
                eligible,
                key=lambda name: (-(shares[name] - whole[name]), str(name)),
            )[:leftovers]:
                allocations.loc[stratum] += 1
                leftovers -= 1

    generator = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for stratum in counts.index:
        group = result.loc[result["_stratum"].eq(stratum)]
        size = int(allocations.loc[stratum])
        if size:
            positions = np.sort(generator.choice(len(group), size=size, replace=False))
            selected.append(group.iloc[positions])
    if not selected:
        return result.drop(columns="_stratum").iloc[0:0].sort_values(
            "id", ignore_index=True
        )
    return (
        pd.concat(selected)
        .drop(columns="_stratum")
        .sort_values("id", ignore_index=True)
    )


def _empty_measurement(path: str, *, file_bytes: int | None = None) -> dict[str, object]:
    return {
        "path": path,
        "width": None,
        "height": None,
        "aspect_ratio": np.nan,
        "mode": None,
        "file_bytes": file_bytes,
        "file_sha256": None,
        "byte_sha256": None,
        "pixel_sha256": None,
        "dhash": None,
        **{metric: np.nan for metric in IMAGE_METRICS},
        "error": None,
    }


def _image_metrics(rgb: np.ndarray) -> dict[str, float]:
    grayscale = np.asarray(
        Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.float64
    )
    rgb_float = rgb.astype(np.float64)
    red_green = rgb_float[:, :, 0] - rgb_float[:, :, 1]
    yellow_blue = (
        (rgb_float[:, :, 0] + rgb_float[:, :, 1]) / 2 - rgb_float[:, :, 2]
    )
    colorfulness = np.sqrt(
        np.std(red_green) ** 2
        + np.std(yellow_blue) ** 2
        + 0.3 * (np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2)
    )
    maximum = rgb_float.max(axis=2)
    minimum = rgb_float.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    horizontal = np.diff(grayscale, axis=1)
    vertical = np.diff(grayscale, axis=0)
    edge_values = np.concatenate((horizontal.ravel(), vertical.ravel()))
    return {
        "brightness": float(grayscale.mean()),
        "contrast": float(grayscale.std()),
        "colorfulness": float(colorfulness),
        "saturation": float(saturation.mean()),
        "edge_sharpness": float(np.mean(np.square(edge_values)))
        if edge_values.size
        else 0.0,
    }


def measure_image(path: Path) -> dict[str, object]:
    """Measure one image using Pillow and NumPy, recording failures as data."""
    try:
        image_path = Path(path)
    except TypeError:
        record = _empty_measurement(str(path))
        record["error"] = "Invalid image path"
        return record
    record = _empty_measurement(str(image_path))
    try:
        content = image_path.read_bytes()
        file_hash = sha256(content).hexdigest()
        record["file_bytes"] = len(content)
        record["file_sha256"] = file_hash
        record["byte_sha256"] = file_hash
        with Image.open(image_path) as image:
            image.load()
            rgb_image = image.convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
            width, height = image.size
            record.update(
                {
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height else np.nan,
                    "mode": image.mode,
                    "pixel_sha256": sha256(
                        f"{width}x{height}:RGB:".encode("ascii") + rgb.tobytes()
                    ).hexdigest(),
                    "dhash": dhash(rgb_image),
                    **_image_metrics(rgb),
                }
            )
    except (OSError, UnidentifiedImageError, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def measure_images(frame: pd.DataFrame, path_column: str) -> pd.DataFrame:
    """Measure paths in deterministic ID/path order while retaining integer IDs."""
    missing = {"id", path_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = frame.loc[:, ["id", path_column]].copy()
    ordered["id"] = _integer_ids(ordered["id"])
    ordered["_path_text"] = ordered[path_column].map(str)
    ordered = ordered.sort_values(["id", "_path_text"], kind="stable")
    records = []
    for row in ordered.itertuples(index=False):
        record = measure_image(getattr(row, path_column))
        record["id"] = int(row.id)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=["id", *MEASUREMENT_COLUMNS])


def _valid_measurements(measurements: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    missing = required.difference(measurements.columns)
    if missing:
        raise ValueError(f"Measurements are missing columns: {sorted(missing)}")
    valid = measurements.copy()
    valid["id"] = _integer_ids(valid["id"])
    return valid.loc[valid["error"].isna() & valid[list(required - {"id", "error"})].notna().all(axis=1)]


def exact_duplicate_groups(measurements: pd.DataFrame) -> pd.DataFrame:
    """Return bounded groups of byte-identical files, excluding unreadable images."""
    columns = ["file_sha256", "pixel_sha256", "ids", "count"]
    valid = _valid_measurements(
        measurements, {"id", "error", "file_sha256", "pixel_sha256"}
    )
    records: list[dict[str, object]] = []
    for file_hash, group in valid.groupby("file_sha256", sort=True):
        ordered = group.sort_values("id", kind="stable")
        if len(ordered) > 1:
            records.append(
                {
                    "file_sha256": file_hash,
                    "pixel_sha256": ordered["pixel_sha256"].iloc[0],
                    "ids": tuple(ordered["id"].tolist()),
                    "count": len(ordered),
                }
            )
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["file_sha256"], ignore_index=True
    )


def near_duplicate_candidates(
    measurements: pd.DataFrame, max_distance: int = 6
) -> pd.DataFrame:
    """Return bounded, deterministic dHash review pairs excluding exact file bytes."""
    if (
        not isinstance(max_distance, Integral)
        or isinstance(max_distance, bool)
        or not 0 <= max_distance <= 64
    ):
        raise ValueError("max_distance must be an integer from 0 to 64")
    columns = ["id_left", "id_right", "distance", "dhash_left", "dhash_right"]
    valid = _valid_measurements(
        measurements, {"id", "error", "file_sha256", "dhash"}
    )
    if len(valid) > MAX_NEAR_DUPLICATE_ITEMS:
        raise ValueError(
            f"near-duplicate analysis supports at most {MAX_NEAR_DUPLICATE_ITEMS} items"
        )
    valid = valid.loc[
        valid["dhash"].astype(str).str.fullmatch(r"[0-9a-fA-F]{16}")
    ].sort_values("id", kind="stable")
    records: list[dict[str, object]] = []
    for left, right in combinations(valid.itertuples(index=False), 2):
        if left.file_sha256 == right.file_sha256:
            continue
        distance = hamming_distance(left.dhash, right.dhash)
        if distance <= max_distance:
            records.append(
                {
                    "id_left": int(left.id),
                    "id_right": int(right.id),
                    "distance": distance,
                    "dhash_left": left.dhash,
                    "dhash_right": right.dhash,
                }
            )
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["distance", "id_left", "id_right"], ignore_index=True
    )


def paired_image_comparison(low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    """Join matched product views once and retain validity and error evidence."""
    required = {"id", "path", "error", *IMAGE_METRICS}
    missing_low = required.difference(low.columns)
    missing_high = required.difference(high.columns)
    if missing_low or missing_high:
        raise ValueError(
            "Measurements are missing columns: "
            f"{sorted(missing_low or missing_high)}"
        )
    left = low.loc[:, ["id", "path", "error", *IMAGE_METRICS]].copy()
    right = high.loc[:, ["id", "path", "error", *IMAGE_METRICS]].copy()
    left["id"] = _integer_ids(left["id"])
    right["id"] = _integer_ids(right["id"])
    if left["id"].duplicated().any() or right["id"].duplicated().any():
        raise ValueError("paired measurements must have one row per id")
    paired = left.merge(right, on="id", how="inner", suffixes=("_low", "_high"))
    paired = paired.rename(
        columns={
            "path_low": "low_path",
            "path_high": "high_path",
            "error_low": "low_error",
            "error_high": "high_error",
        }
    )
    paired["valid_pair"] = paired["low_error"].isna() & paired["high_error"].isna()
    for metric in IMAGE_METRICS:
        low_column = f"{metric}_low"
        high_column = f"{metric}_high"
        paired[f"{metric}_delta"] = paired[high_column] - paired[low_column]
        paired[f"{metric}_ratio"] = np.divide(
            paired[high_column],
            paired[low_column],
            out=np.full(len(paired), np.nan),
            where=paired[low_column].to_numpy(dtype=float) != 0,
        )
        paired.loc[~paired["valid_pair"], f"{metric}_delta"] = np.nan
        paired.loc[~paired["valid_pair"], f"{metric}_ratio"] = np.nan
    return paired.sort_values("id", ignore_index=True)
