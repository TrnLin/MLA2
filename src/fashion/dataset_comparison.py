from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from fashion.config import (
    ORIGINAL_CSV,
    ORIGINAL_IMAGE_DIR,
    ORIGINAL_IMAGE_LINKS_CSV,
    ORIGINAL_STYLE_JSON_DIR,
    RANDOM_SEED,
    TARGET_COLUMNS,
    TEACHER_TRAIN_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
    TEST_CSV,
    TEST_IMAGE_DIR,
)
from fashion.eda import _dhash, audit_csv, hierarchy_conflicts, hamming_distance


LABEL_COLUMNS = (
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
)
NEAR_DUPLICATE_THRESHOLDS = (0, 2, 4, 6)


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _display(value: object) -> str:
    cleaned = _clean(value)
    return cleaned if cleaned else "<BLANK>"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _sorted_jpgs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))


def _inventory(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in _sorted_jpgs(directory)}


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, object]]:
    selected = frame if limit is None else frame.head(limit)
    return _json_safe(selected.to_dict("records"))  # type: ignore[return-value]


def _distribution(frame: pd.DataFrame, target: str) -> list[dict[str, object]]:
    values = frame[target].map(_display)
    counts = values.value_counts(dropna=False)
    total = len(frame)
    return [
        {"label": str(label), "count": int(count), "share": float(count / total)}
        for label, count in counts.items()
    ]


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((count / total) * math.log2(count / total) for count in counter.values()))


def _split_label_audit(
    official_train: pd.DataFrame,
    quarantined_test: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in TARGET_COLUMNS:
        train = Counter(official_train[target].map(_display))
        test = Counter(quarantined_test[target].map(_display))
        labels = sorted(set(train) | set(test))
        train_total = sum(train.values())
        test_total = sum(test.values())
        total_variation = 0.5 * sum(
            abs(train[label] / train_total - test[label] / test_total)
            for label in labels
        )
        unseen = sorted(set(test) - set(train), key=lambda label: (-test[label], label))
        train_only = sorted(set(train) - set(test), key=lambda label: (-train[label], label))
        result[target] = {
            "train_classes": len(train),
            "test_classes": len(test),
            "train_entropy_bits": _entropy(train),
            "test_entropy_bits": _entropy(test),
            "total_variation": float(total_variation),
            "unseen_test_labels": [
                {"label": label, "test_rows": int(test[label])} for label in unseen
            ],
            "unseen_test_rows": int(sum(test[label] for label in unseen)),
            "train_only_labels": [
                {"label": label, "train_rows": int(train[label])}
                for label in train_only
            ],
            "train_distribution": [
                {"label": label, "count": int(count)}
                for label, count in train.most_common()
            ],
            "test_distribution": [
                {"label": label, "count": int(count)}
                for label, count in test.most_common()
            ],
        }
    return result


def _extract_json_labels(data: dict[str, object]) -> dict[str, str]:
    payload = data.get("data", {})
    if not isinstance(payload, dict):
        return {}

    def nested_type(name: str) -> str:
        value = payload.get(name, {})
        return _clean(value.get("typeName")) if isinstance(value, dict) else ""

    return {
        "gender": _clean(payload.get("gender")),
        "masterCategory": nested_type("masterCategory"),
        "subCategory": nested_type("subCategory"),
        "articleType": nested_type("articleType"),
        "baseColour": _clean(payload.get("baseColour")),
        "season": _clean(payload.get("season")),
        "year": _clean(payload.get("year")),
        "usage": _clean(payload.get("usage")),
        "productDisplayName": _clean(payload.get("productDisplayName")),
    }


def _audit_json_labels(
    original_by_id: pd.DataFrame,
    json_directory: Path,
    verbose: bool,
) -> dict[str, object]:
    csv_lookup = original_by_id.set_index("id")
    mismatch_counts = Counter()
    mismatch_examples: list[dict[str, object]] = []
    parse_errors: list[dict[str, str]] = []
    json_ids: set[str] = set()

    paths = sorted(json_directory.glob("*.json"), key=lambda path: int(path.stem))
    for index, path in enumerate(paths, start=1):
        image_id = path.stem
        json_ids.add(image_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = _extract_json_labels(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parse_errors.append({"id": image_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if image_id not in csv_lookup.index:
            continue
        csv_row = csv_lookup.loc[image_id]
        for column in LABEL_COLUMNS:
            left = _clean(csv_row[column])
            right = _clean(labels.get(column, ""))
            if left != right:
                mismatch_counts[column] += 1
                if len(mismatch_examples) < 40:
                    mismatch_examples.append(
                        {
                            "id": image_id,
                            "column": column,
                            "csv": left,
                            "json": right,
                        }
                    )
        if verbose and index % 15000 == 0:
            print(f"  JSON records checked: {index:,}/{len(paths):,}")

    csv_ids = set(original_by_id["id"].astype(str))
    return {
        "files": len(paths),
        "parse_errors": parse_errors,
        "csv_without_json": sorted(csv_ids - json_ids, key=int),
        "json_without_csv": sorted(json_ids - csv_ids, key=int),
        "mismatch_counts": dict(mismatch_counts),
        "mismatch_examples": mismatch_examples,
    }


def _inspect_image_collection(
    paths: Sequence[Path],
    decode_limit: int | None,
    seed: int,
    verbose: bool,
) -> dict[str, object]:
    dimensions: Counter[tuple[int, int, str]] = Counter()
    header_errors: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                dimensions[(image.width, image.height, image.mode)] += 1
        except (OSError, UnidentifiedImageError) as exc:
            header_errors.append({"id": path.stem, "error": f"{type(exc).__name__}: {exc}"})
        if verbose and index % 20000 == 0:
            print(f"  image headers checked: {index:,}/{len(paths):,}")

    selected = list(paths)
    if decode_limit is not None and len(selected) > decode_limit:
        selected = Random(seed).sample(selected, decode_limit)
    decode_errors: list[dict[str, str]] = []
    for path in selected:
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, UnidentifiedImageError) as exc:
            decode_errors.append({"id": path.stem, "error": f"{type(exc).__name__}: {exc}"})

    byte_sizes = np.array([path.stat().st_size for path in paths], dtype=np.int64)
    pixel_counts = np.array(
        [width * height for (width, height, _), count in dimensions.items() for _ in range(count)],
        dtype=np.int64,
    )
    return {
        "files": len(paths),
        "bytes_total": int(byte_sizes.sum()) if len(byte_sizes) else 0,
        "bytes_median": float(np.median(byte_sizes)) if len(byte_sizes) else 0.0,
        "megapixels_median": (
            float(np.median(pixel_counts) / 1_000_000) if len(pixel_counts) else 0.0
        ),
        "dimensions": [
            {"width": width, "height": height, "mode": mode, "files": count}
            for (width, height, mode), count in dimensions.most_common()
        ],
        "header_errors": header_errors,
        "decoded_files": len(selected),
        "decode_errors": decode_errors,
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_original_images(
    inventory: dict[str, Path],
    labels_by_id: pd.DataFrame,
    test_ids: set[str],
    verbose: bool,
) -> tuple[dict[str, str], dict[str, object]]:
    digest_to_ids: dict[str, list[str]] = defaultdict(list)
    id_to_digest: dict[str, str] = {}
    paths = sorted(inventory.items(), key=lambda item: int(item[0]))
    for index, (image_id, path) in enumerate(paths, start=1):
        digest = _sha256_file(path)
        digest_to_ids[digest].append(image_id)
        id_to_digest[image_id] = digest
        if verbose and index % 10000 == 0:
            print(f"  original images hashed: {index:,}/{len(paths):,}")

    duplicate_groups = [
        sorted(ids, key=int) for ids in digest_to_ids.values() if len(ids) > 1
    ]
    duplicate_groups.sort(key=lambda ids: (-len(ids), [int(value) for value in ids]))
    label_lookup = labels_by_id.set_index("id")
    conflict_counts = Counter()
    conflict_examples: list[dict[str, object]] = []
    cross_split_groups = 0

    for ids in duplicate_groups:
        splits = {"test" if image_id in test_ids else "train" for image_id in ids}
        if len(splits) > 1:
            cross_split_groups += 1
        differences: dict[str, list[str]] = {}
        for target in TARGET_COLUMNS:
            values = sorted({_display(label_lookup.loc[image_id, target]) for image_id in ids})
            if len(values) > 1:
                differences[target] = values
                conflict_counts[target] += 1
        if differences and len(conflict_examples) < 30:
            conflict_examples.append(
                {
                    "ids": ids,
                    "splits": sorted(splits),
                    "differences": differences,
                }
            )

    return id_to_digest, {
        "groups": len(duplicate_groups),
        "images": int(sum(len(group) for group in duplicate_groups)),
        "largest_group": max((len(group) for group in duplicate_groups), default=0),
        "cross_split_groups": cross_split_groups,
        "label_conflict_groups": int(sum(1 for group in duplicate_groups if any(
            len({_display(label_lookup.loc[image_id, target]) for image_id in group}) > 1
            for target in TARGET_COLUMNS
        ))),
        "conflict_counts_by_target": dict(conflict_counts),
        "conflict_examples": conflict_examples,
        "largest_groups": duplicate_groups[:20],
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]

    def summary(self) -> tuple[int, int, int]:
        sizes = Counter(self.find(index) for index in range(len(self.parent)))
        component_sizes = [count for count in sizes.values() if count > 1]
        return (
            len(component_sizes),
            sum(component_sizes),
            max(component_sizes, default=1),
        )


def _dhash_with_draft(path: Path) -> str:
    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", (144, 192))
        return _dhash(image)


def _near_duplicate_sweep(
    inventory: dict[str, Path],
    labels_by_id: pd.DataFrame,
    test_ids: set[str],
    id_to_digest: dict[str, str],
    sample_size: int,
    seed: int,
    verbose: bool,
) -> dict[str, object]:
    train_pool = sorted(set(inventory) - test_ids, key=int)
    test_pool = sorted(set(inventory) & test_ids, key=int)
    rng = Random(seed)
    desired_test = min(len(test_pool), max(1, sample_size // 4))
    desired_train = min(len(train_pool), sample_size - desired_test)
    sampled_ids = sorted(
        rng.sample(train_pool, desired_train) + rng.sample(test_pool, desired_test),
        key=int,
    )
    hashes: list[int] = []
    readable_ids: list[str] = []
    errors: list[dict[str, str]] = []
    for index, image_id in enumerate(sampled_ids, start=1):
        try:
            hashes.append(int(_dhash_with_draft(inventory[image_id]), 16))
            readable_ids.append(image_id)
        except (OSError, UnidentifiedImageError) as exc:
            errors.append({"id": image_id, "error": f"{type(exc).__name__}: {exc}"})
        if verbose and index % 500 == 0:
            print(f"  perceptual sample hashed: {index:,}/{len(sampled_ids):,}")

    sets = {threshold: _DisjointSet(len(readable_ids)) for threshold in NEAR_DUPLICATE_THRESHOLDS}
    edge_counts = Counter()
    conflicting_pairs: list[dict[str, object]] = []
    label_lookup = labels_by_id.set_index("id")
    for left in range(len(readable_ids)):
        for right in range(left + 1, len(readable_ids)):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance > max(NEAR_DUPLICATE_THRESHOLDS):
                continue
            left_id = readable_ids[left]
            right_id = readable_ids[right]
            if id_to_digest.get(left_id) == id_to_digest.get(right_id):
                continue
            for threshold in NEAR_DUPLICATE_THRESHOLDS:
                if distance <= threshold:
                    edge_counts[threshold] += 1
                    sets[threshold].union(left, right)
            differences = {
                target: [
                    _display(label_lookup.loc[left_id, target]),
                    _display(label_lookup.loc[right_id, target]),
                ]
                for target in TARGET_COLUMNS
                if _display(label_lookup.loc[left_id, target])
                != _display(label_lookup.loc[right_id, target])
            }
            if differences and len(conflicting_pairs) < 50:
                conflicting_pairs.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "distance": distance,
                        "cross_split": (left_id in test_ids) != (right_id in test_ids),
                        "differences": differences,
                    }
                )

    conflicting_pairs.sort(key=lambda row: (int(row["distance"]), int(row["left_id"])))
    sweep = []
    for threshold in NEAR_DUPLICATE_THRESHOLDS:
        groups, images, largest = sets[threshold].summary()
        sweep.append(
            {
                "threshold": threshold,
                "candidate_pairs": int(edge_counts[threshold]),
                "components": groups,
                "images_in_components": images,
                "largest_component": largest,
            }
        )
    return {
        "sample_size": len(readable_ids),
        "train_sample": sum(image_id not in test_ids for image_id in readable_ids),
        "test_sample": sum(image_id in test_ids for image_id in readable_ids),
        "errors": errors,
        "sweep": sweep,
        "conflicting_pair_examples": conflicting_pairs[:30],
    }


def _same_id_image_comparison(
    original: dict[str, Path],
    teacher_train: dict[str, Path],
    teacher_test: dict[str, Path],
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    teacher = {**teacher_train, **teacher_test}
    shared = sorted(set(original) & set(teacher), key=int)
    selected = Random(seed).sample(shared, min(sample_size, len(shared)))
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for image_id in selected:
        try:
            with Image.open(teacher[image_id]) as small_image:
                small = np.asarray(small_image.convert("RGB"), dtype=np.float32)
                small_size = small_image.size
            with Image.open(original[image_id]) as full_image:
                full_size = full_image.size
                resized = np.asarray(
                    full_image.convert("RGB").resize(small_size, Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
                full_hash = _dhash(full_image)
            with Image.open(teacher[image_id]) as small_image:
                small_hash = _dhash(small_image)
            mse = float(np.mean((small - resized) ** 2))
            psnr = 99.0 if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))
            records.append(
                {
                    "id": image_id,
                    "teacher_width": small_size[0],
                    "teacher_height": small_size[1],
                    "original_width": full_size[0],
                    "original_height": full_size[1],
                    "pixel_ratio": (full_size[0] * full_size[1]) / (small_size[0] * small_size[1]),
                    "psnr_db": psnr,
                    "dhash_distance": hamming_distance(small_hash, full_hash),
                    "teacher_bytes": teacher[image_id].stat().st_size,
                    "original_bytes": original[image_id].stat().st_size,
                }
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            errors.append({"id": image_id, "error": f"{type(exc).__name__}: {exc}"})

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return {"sample_size": 0, "errors": errors}
    return {
        "sample_size": len(frame),
        "errors": errors,
        "median_pixel_ratio": float(frame["pixel_ratio"].median()),
        "median_psnr_db": float(frame["psnr_db"].median()),
        "median_dhash_distance": float(frame["dhash_distance"].median()),
        "share_dhash_at_most_6": float(frame["dhash_distance"].le(6).mean()),
        "median_file_size_ratio": float(
            (frame["original_bytes"] / frame["teacher_bytes"]).median()
        ),
        "teacher_dimensions": [
            {
                "width": int(width),
                "height": int(height),
                "files": int(count),
            }
            for (width, height), count in frame.groupby(
                ["teacher_width", "teacher_height"]
            ).size().sort_values(ascending=False).items()
        ],
        "examples": _frame_records(
            frame.sort_values(["dhash_distance", "psnr_db"], ascending=[False, True]),
            20,
        ),
    }


def _name_gender_contradictions(frame: pd.DataFrame) -> dict[str, object]:
    explicit_patterns = {
        "Men": re.compile(r"\b(men|mens|man)\b", re.IGNORECASE),
        "Women": re.compile(r"\b(women|womens|woman|ladies)\b", re.IGNORECASE),
        "Boys": re.compile(r"\b(boys|boy)\b", re.IGNORECASE),
        "Girls": re.compile(r"\b(girls|girl)\b", re.IGNORECASE),
    }
    candidates: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        name = _clean(row.productDisplayName)
        mentioned = [
            label for label, pattern in explicit_patterns.items() if pattern.search(name)
        ]
        if len(mentioned) == 1 and row.gender not in {mentioned[0], "Unisex"}:
            candidates.append(
                {
                    "id": str(row.id),
                    "csv_gender": _clean(row.gender),
                    "name_gender": mentioned[0],
                    "articleType": _clean(row.articleType),
                    "productDisplayName": name,
                }
            )
    return {"count": len(candidates), "examples": candidates[:50]}


def _known_candidate_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    candidates = frame.loc[
        frame["id"].isin(["38223", "45824"]),
        ["id", "gender", "masterCategory", "subCategory", "articleType", "season", "usage", "productDisplayName"],
    ]
    return _frame_records(candidates)


def _rare_combinations(frame: pd.DataFrame) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(list(TARGET_COLUMNS), dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", *TARGET_COLUMNS], ignore_index=True)
    )
    return _frame_records(grouped.loc[grouped["rows"].le(2)], 40)


def _render_figures(
    summary: dict[str, object],
    official_train: pd.DataFrame,
    quarantined_test: pd.DataFrame,
    labels_by_id: pd.DataFrame,
    output_directory: Path,
) -> list[str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []

    populations = summary["populations"]
    names = ["Original", "Teacher train", "Teacher test"]
    metadata = [
        populations["original_metadata_rows"],
        populations["teacher_train_metadata_rows"],
        populations["teacher_test_rows"],
    ]
    images = [
        populations["original_image_files"],
        populations["teacher_train_image_files"],
        populations["teacher_test_image_files"],
    ]
    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(x - width / 2, metadata, width, label="Metadata rows")
    axis.bar(x + width / 2, images, width, label="JPG files")
    axis.set(title="Metadata and image coverage by source", ylabel="Files or rows", xticks=x, xticklabels=names)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_directory / "population-coverage.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(str(path))

    label_audit = summary["labels"]["split_comparison"]
    targets = list(TARGET_COLUMNS)
    shifts = [label_audit[target]["total_variation"] for target in targets]
    unseen_rows = [label_audit[target]["unseen_test_rows"] for target in targets]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(targets, shifts)
    axes[0].set(title="Official train-to-test label shift", ylabel="Total variation distance", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(targets, unseen_rows)
    axes[1].set(title="Test rows in labels unseen during training", ylabel="Quarantined test rows")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_directory / "label-shift.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(str(path))

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for frame, label in (
        (official_train, "Official teacher train"),
        (quarantined_test, "Quarantined teacher test"),
    ):
        counts = frame["articleType"].map(_display).value_counts().sort_values(ascending=False)
        axis.plot(np.arange(1, len(counts) + 1), counts.to_numpy(), marker=".", markersize=3, label=label)
    axis.set_yscale("log")
    axis.set(
        title="articleType support is long-tailed and split-shifted",
        xlabel="Class rank",
        ylabel="Rows per class (log scale)",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_directory / "article-type-long-tail.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(str(path))

    anomaly_ids = ["38223", "45824"]
    duplicate_examples = summary["labels"]["exact_duplicate_conflicts"]
    for example in duplicate_examples:
        anomaly_ids.extend(example["ids"])
    contradiction_examples = summary["labels"]["name_gender_contradictions"]["examples"]
    anomaly_ids.extend(example["id"] for example in contradiction_examples)
    anomaly_ids = list(dict.fromkeys(anomaly_ids))[:12]
    lookup = labels_by_id.set_index("id")
    columns = 4
    rows = max(1, math.ceil(len(anomaly_ids) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows), squeeze=False)
    for axis, image_id in zip(axes.flat, anomaly_ids):
        path = ORIGINAL_IMAGE_DIR / f"{image_id}.jpg"
        try:
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            row = lookup.loc[image_id]
            axis.set_title(
                f"{image_id} · {_display(row['articleType'])}\n"
                f"{_display(row['gender'])} · {_display(row['season'])} · {_display(row['usage'])}",
                fontsize=8,
            )
        except (OSError, UnidentifiedImageError, KeyError):
            axis.text(0.5, 0.5, f"{image_id}\nunavailable", ha="center", va="center")
        axis.axis("off")
    for axis in axes.flat[len(anomaly_ids):]:
        axis.axis("off")
    figure.suptitle("Highest-priority label review candidates")
    figure.tight_layout()
    path = output_directory / "label-review-candidates.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(str(path))
    return figure_paths


def run_dataset_comparison(
    output_directory: Path,
    *,
    same_id_sample_size: int = 512,
    near_duplicate_sample_size: int = 2048,
    original_decode_sample_size: int = 512,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Reading metadata and reconciling official IDs...")
    original, original_schema = audit_csv(ORIGINAL_CSV)
    teacher_train, teacher_schema = audit_csv(TEACHER_TRAIN_CSV)
    template = pd.read_csv(TEST_CSV, keep_default_na=False, dtype="string")
    original["id"] = original["id"].astype("string")
    teacher_train["id"] = teacher_train["id"].astype("string")
    template["id"] = template["id"].astype("string")

    test_ids = set(template["id"].astype(str))
    original_ids = set(original["id"].astype(str))
    teacher_train_ids = set(teacher_train["id"].astype(str))
    quarantined_test = original.loc[original["id"].isin(test_ids)].copy()
    official_train_from_original = original.loc[original["id"].isin(teacher_train_ids)].copy()

    shared = teacher_train.merge(
        official_train_from_original,
        on="id",
        suffixes=("_teacher", "_original"),
        validate="one_to_one",
    )
    shared_mismatch_counts = {}
    shared_mismatch_examples: list[dict[str, object]] = []
    for column in LABEL_COLUMNS:
        left = shared[f"{column}_teacher"].map(_clean)
        right = shared[f"{column}_original"].map(_clean)
        mismatch = left.ne(right)
        shared_mismatch_counts[column] = int(mismatch.sum())
        if mismatch.any() and len(shared_mismatch_examples) < 30:
            for row in shared.loc[mismatch, ["id", f"{column}_teacher", f"{column}_original"]].head(5).itertuples(index=False):
                shared_mismatch_examples.append(
                    {
                        "id": str(row[0]),
                        "column": column,
                        "teacher": _clean(row[1]),
                        "original": _clean(row[2]),
                    }
                )

    original_images = _inventory(ORIGINAL_IMAGE_DIR)
    teacher_train_images = _inventory(TEACHER_TRAIN_IMAGE_DIR)
    teacher_test_images = _inventory(TEST_IMAGE_DIR)

    if verbose:
        print("Checking the original JSON labels against styles.csv...")
    json_audit = _audit_json_labels(original, ORIGINAL_STYLE_JSON_DIR, verbose)
    links = pd.read_csv(
        ORIGINAL_IMAGE_LINKS_CSV,
        keep_default_na=False,
        dtype="string",
    )
    link_ids = set(links["filename"].str.replace(r"\.jpg$", "", regex=True).astype(str))

    if verbose:
        print("Inspecting image headers and decode integrity...")
    image_audits = {
        "original": _inspect_image_collection(
            list(original_images.values()),
            original_decode_sample_size,
            seed,
            verbose,
        ),
        "teacher_train_present": _inspect_image_collection(
            list(teacher_train_images.values()),
            None,
            seed,
            verbose,
        ),
        "teacher_test": _inspect_image_collection(
            list(teacher_test_images.values()),
            None,
            seed,
            verbose,
        ),
    }

    if verbose:
        print("Hashing original images for exact duplicates...")
    id_to_digest, exact_duplicates = _hash_original_images(
        original_images,
        original,
        test_ids,
        verbose,
    )
    if verbose:
        print("Comparing same-ID original and teacher images...")
    same_id = _same_id_image_comparison(
        original_images,
        teacher_train_images,
        teacher_test_images,
        same_id_sample_size,
        seed,
    )
    if verbose:
        print("Sweeping perceptual near-duplicate thresholds...")
    near_duplicates = _near_duplicate_sweep(
        original_images,
        original,
        test_ids,
        id_to_digest,
        near_duplicate_sample_size,
        seed,
        verbose,
    )

    split_comparison = _split_label_audit(
        official_train_from_original,
        quarantined_test,
    )
    article_counts = official_train_from_original["articleType"].map(_display).value_counts()
    hierarchy = hierarchy_conflicts(official_train_from_original)
    name_contradictions = _name_gender_contradictions(official_train_from_original)

    missing_teacher_images = teacher_train_ids - set(teacher_train_images)
    recoverable_from_original = missing_teacher_images & set(original_images)
    unrecoverable = missing_teacher_images - set(original_images)
    exact_conflicts = exact_duplicates["conflict_examples"]

    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "populations": {
            "original_metadata_rows": len(original),
            "teacher_train_metadata_rows": len(teacher_train),
            "teacher_test_rows": len(template),
            "original_image_files": len(original_images),
            "teacher_train_image_files": len(teacher_train_images),
            "teacher_test_image_files": len(teacher_test_images),
            "original_json_files": json_audit["files"],
            "original_link_rows": len(links),
        },
        "id_reconciliation": {
            "original_equals_teacher_union": original_ids == teacher_train_ids | test_ids,
            "teacher_train_test_overlap": len(teacher_train_ids & test_ids),
            "original_not_in_teacher_union": sorted(
                original_ids - teacher_train_ids - test_ids, key=int
            ),
            "teacher_not_in_original": sorted(
                (teacher_train_ids | test_ids) - original_ids, key=int
            ),
            "shared_label_mismatch_counts": shared_mismatch_counts,
            "shared_label_mismatch_examples": shared_mismatch_examples,
            "test_template_nonblank_cells": {
                column: int(template[column].str.strip().ne("").sum())
                for column in template.columns
                if column != "id"
            },
        },
        "schema": {
            "original": {
                "row_count": original_schema.row_count,
                "physical_columns": list(original_schema.physical_columns),
                "phantom_columns": list(original_schema.phantom_columns),
                "phantom_nonempty_counts": original_schema.phantom_nonempty_counts,
                "duplicate_ids": list(original_schema.duplicate_ids),
                "blank_counts": original_schema.blank_counts,
                "literal_na_usage_count": original_schema.literal_na_usage_count,
            },
            "teacher_train": {
                "row_count": teacher_schema.row_count,
                "physical_columns": list(teacher_schema.physical_columns),
                "phantom_columns": list(teacher_schema.phantom_columns),
                "phantom_nonempty_counts": teacher_schema.phantom_nonempty_counts,
                "duplicate_ids": list(teacher_schema.duplicate_ids),
                "blank_counts": teacher_schema.blank_counts,
                "literal_na_usage_count": teacher_schema.literal_na_usage_count,
            },
        },
        "artifact_consistency": {
            "json": json_audit,
            "csv_without_link": sorted(original_ids - link_ids, key=int),
            "link_without_csv": sorted(link_ids - original_ids, key=int),
            "csv_without_original_image": sorted(original_ids - set(original_images), key=int),
            "original_image_without_csv": sorted(set(original_images) - original_ids, key=int),
        },
        "teacher_train_image_loss": {
            "metadata_rows": len(teacher_train_ids),
            "present_images": len(teacher_train_images),
            "missing_images": len(missing_teacher_images),
            "recoverable_from_original": len(recoverable_from_original),
            "unrecoverable_known_orphans": sorted(unrecoverable, key=int),
            "first_missing_ids": sorted(missing_teacher_images, key=int)[:30],
            "largest_present_id": max(map(int, teacher_train_images), default=None),
        },
        "labels": {
            "split_comparison": split_comparison,
            "official_train_distributions": {
                target: _distribution(official_train_from_original, target)
                for target in TARGET_COLUMNS
            },
            "article_type_classes": int(article_counts.size),
            "article_type_singletons": int(article_counts.eq(1).sum()),
            "article_type_under_10": int(article_counts.lt(10).sum()),
            "hierarchy_conflicts": len(hierarchy),
            "hierarchy_conflict_examples": _frame_records(hierarchy, 30),
            "name_gender_contradictions": name_contradictions,
            "rare_target_combinations": _rare_combinations(official_train_from_original),
            "known_review_candidates": _known_candidate_rows(original),
            "exact_duplicate_conflicts": exact_conflicts,
        },
        "images": {
            "audits": image_audits,
            "same_id_comparison": same_id,
            "exact_duplicates": exact_duplicates,
            "near_duplicate_sample": near_duplicates,
        },
        "decision": {
            "label_source": (
                "Labels are identical for every shared teacher-train ID; the original JSON "
                "is a useful cross-check, not a cleaner independent annotation source."
            ),
            "image_source": (
                "Use original high-resolution images for official teacher-train IDs when "
                "the extra detail is useful; the restored teacher thumbnails are complete "
                "for every source-available train image but heavily downsampled."
            ),
            "test_policy": (
                "Keep all 5,829 teacher-test IDs and their recovered original labels out of "
                "training, tuning, feature selection, and model comparison."
            ),
        },
    }
    summary["figures"] = _render_figures(
        summary,
        official_train_from_original,
        quarantined_test,
        original,
        output_directory,
    )
    safe_summary = _json_safe(summary)
    summary_path = output_directory / "comparison-summary.json"
    summary_path.write_text(json.dumps(safe_summary, indent=2), encoding="utf-8")
    if verbose:
        print(f"Comparison summary written to {summary_path}")
    return safe_summary  # type: ignore[return-value]


def load_comparison_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
