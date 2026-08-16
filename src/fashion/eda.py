from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class SchemaAudit:
    row_count: int
    physical_columns: tuple[str, ...]
    phantom_columns: tuple[str, ...]
    phantom_nonempty_counts: dict[str, int]
    duplicate_ids: tuple[str, ...]
    blank_counts: dict[str, int]
    literal_na_usage_count: int


@dataclass(frozen=True)
class IdReconciliation:
    csv_only_ids: tuple[str, ...]
    image_only_ids: tuple[str, ...]
    matched_count: int


def audit_csv(path: Path) -> tuple[pd.DataFrame, SchemaAudit]:
    if not path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {path}")

    frame = pd.read_csv(path, keep_default_na=False, dtype={"id": "string"})
    required = {
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    phantom = tuple(column for column in frame.columns if column.startswith("Unnamed:"))
    nonempty_phantoms = {
        column: int(frame[column].astype("string").str.strip().ne("").sum())
        for column in phantom
    }
    display_columns = ["productDisplayName", *phantom]
    frame["productDisplayName"] = frame[display_columns].apply(
        lambda row: ",".join(
            str(value) for value in row if str(value).strip()
        ),
        axis=1,
    )

    duplicate_ids = tuple(
        sorted(frame.loc[frame["id"].duplicated(keep=False), "id"].astype(str).unique())
    )
    named = frame.drop(columns=list(phantom))
    blank_counts = {
        column: int(named[column].astype("string").str.strip().eq("").sum())
        for column in named.columns
    }
    audit = SchemaAudit(
        row_count=len(frame),
        physical_columns=tuple(frame.columns),
        phantom_columns=phantom,
        phantom_nonempty_counts=nonempty_phantoms,
        duplicate_ids=duplicate_ids,
        blank_counts=blank_counts,
        literal_na_usage_count=int(named["usage"].eq("NA").sum()),
    )
    return named, audit


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_pixels(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _dhash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def audit_images(image_dir: Path) -> pd.DataFrame:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")

    records: list[dict[str, object]] = []
    for path in sorted(image_dir.glob("*.jpg"), key=lambda item: item.stem):
        record: dict[str, object] = {
            "id": path.stem,
            "path": str(path),
            "width": pd.NA,
            "height": pd.NA,
            "mode": pd.NA,
            "byte_sha256": pd.NA,
            "pixel_sha256": pd.NA,
            "dhash": pd.NA,
            "error": "",
        }
        try:
            record["byte_sha256"] = _sha256_file(path)
            with Image.open(path) as image:
                image.load()
                record.update(
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    pixel_sha256=_sha256_pixels(image),
                    dhash=_dhash(image),
                )
        except (OSError, UnidentifiedImageError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    return pd.DataFrame.from_records(records).astype({"id": "string"})


def reconcile_ids(styles: pd.DataFrame, images: pd.DataFrame) -> IdReconciliation:
    csv_ids = set(styles["id"].astype(str))
    image_ids = set(images["id"].astype(str))
    return IdReconciliation(
        csv_only_ids=tuple(sorted(csv_ids - image_ids)),
        image_only_ids=tuple(sorted(image_ids - csv_ids)),
        matched_count=len(csv_ids & image_ids),
    )


def exact_duplicate_groups(
    images: pd.DataFrame,
    hash_column: str = "pixel_sha256",
) -> tuple[tuple[str, ...], ...]:
    usable = images.dropna(subset=[hash_column])
    groups = (
        tuple(sorted(group["id"].astype(str)))
        for _, group in usable.groupby(hash_column, sort=True)
        if len(group) > 1
    )
    return tuple(sorted(groups))


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass
class _BKNode:
    value: int
    children: dict[int, "_BKNode"]


class _BKTree:
    def __init__(self) -> None:
        self._root: _BKNode | None = None

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self._root is None:
            self._root = _BKNode(value=value, children={})
            return

        node = self._root
        while True:
            distance = self._distance(value, node.value)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value, children={})
                return
            node = child

    def query(self, value: int, radius: int) -> tuple[int, ...]:
        if self._root is None:
            return ()

        matches: list[int] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            distance = self._distance(value, node.value)
            if distance <= radius:
                matches.append(node.value)
            lower = distance - radius
            upper = distance + radius
            stack.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return tuple(sorted(matches))


def near_duplicate_groups(
    images: pd.DataFrame,
    max_distance: int = 6,
) -> tuple[tuple[str, ...], ...]:
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be between 0 and 64")

    usable = images.dropna(subset=["dhash", "pixel_sha256"])[
        ["id", "dhash", "pixel_sha256"]
    ].copy()
    records_by_hash: dict[int, list[tuple[str, str]]] = {}
    for row in usable.itertuples(index=False):
        records_by_hash.setdefault(int(row.dhash, 16), []).append(
            (str(row.id), str(row.pixel_sha256))
        )

    parent = {image_id: image_id for image_id in usable["id"].astype(str)}

    def find(image_id: str) -> str:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    def connect(
        left_records: list[tuple[str, str]],
        right_records: list[tuple[str, str]],
        same_bucket: bool = False,
    ) -> None:
        for left_index, (left_id, left_pixels) in enumerate(left_records):
            start = left_index + 1 if same_bucket else 0
            for right_id, right_pixels in right_records[start:]:
                if left_pixels != right_pixels:
                    union(left_id, right_id)

    tree = _BKTree()
    for hash_value in sorted(records_by_hash):
        current_records = records_by_hash[hash_value]
        connect(current_records, current_records, same_bucket=True)
        for match in tree.query(hash_value, max_distance):
            connect(current_records, records_by_hash[match])
        tree.add(hash_value)

    components: dict[str, list[str]] = {}
    for image_id in sorted(parent):
        components.setdefault(find(image_id), []).append(image_id)
    return tuple(
        sorted(tuple(group) for group in components.values() if len(group) > 1)
    )
