from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

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
    """Return candidate near-duplicate components grouped by union-find over dHash.

    Each returned tuple lists image IDs transitively linked through dHash pairs
    within ``max_distance`` whose ``pixel_sha256`` values differ. Members of a
    component are **not** guaranteed to be pairwise within ``max_distance``;
    a chain A–B–C can place A and C in the same group even when their direct
    Hamming distance exceeds ``max_distance``.

    Pairs with identical ``pixel_sha256`` never receive a near-duplicate edge,
    but two exact-pixel duplicates can still appear together when a third image
    bridges them. Treat every group as a review queue—confirm visually before
    treating members as near duplicates.
    """
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


def _display_labels(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<BLANK>").replace("", "<BLANK>")


def target_distribution(styles: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in styles:
        raise KeyError(f"Unknown target: {target}")
    counts = (
        _display_labels(styles[target])
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values(["count", "label"], ascending=[False, True], ignore_index=True)
    )
    counts["share"] = counts["count"] / len(styles) if len(styles) else 0.0
    counts["rank"] = np.arange(1, len(counts) + 1)
    return counts[["label", "count", "share", "rank"]]


def target_summary(
    styles: pd.DataFrame,
    targets: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in targets:
        distribution = target_distribution(styles, target)
        nonblank = distribution.loc[distribution["label"].ne("<BLANK>")]
        majority = nonblank.iloc[0] if not nonblank.empty else None
        minority_count = int(nonblank["count"].min()) if not nonblank.empty else 0
        majority_count = int(majority["count"]) if majority is not None else 0
        rows.append(
            {
                "target": target,
                "rows": len(styles),
                "classes_including_blank": len(distribution),
                "blank_count": int(
                    distribution.loc[
                        distribution["label"].eq("<BLANK>"), "count"
                    ].sum()
                ),
                "majority_label": majority["label"] if majority is not None else "",
                "majority_count": majority_count,
                "majority_accuracy": majority_count / len(styles) if len(styles) else 0.0,
                "minority_count": minority_count,
                "imbalance_ratio": (
                    majority_count / minority_count if minority_count else np.inf
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def hierarchy_conflicts(styles: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        styles.groupby("articleType", dropna=False)
        .agg(
            master_categories=(
                "masterCategory",
                lambda values: tuple(sorted(set(_display_labels(values)))),
            ),
            subcategories=(
                "subCategory",
                lambda values: tuple(sorted(set(_display_labels(values)))),
            ),
            rows=("id", "size"),
        )
        .reset_index()
    )
    grouped["master_category_count"] = grouped["master_categories"].str.len()
    grouped["subcategory_count"] = grouped["subcategories"].str.len()
    return grouped.loc[
        grouped["master_category_count"].gt(1) | grouped["subcategory_count"].gt(1)
    ].sort_values("articleType", ignore_index=True)


def cooccurrence_table(
    styles: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    if left not in styles or right not in styles:
        raise KeyError(f"Unknown columns: {left}, {right}")
    return pd.crosstab(
        _display_labels(styles[left]),
        _display_labels(styles[right]),
        dropna=False,
    )


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(_display_labels(left), _display_labels(right), dropna=False)
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    row_count, column_count = observed.shape
    if total == 0 or row_count < 2 or column_count < 2:
        return 0.0

    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    chi_squared = np.divide(
        (observed - expected) ** 2,
        expected,
        out=np.zeros_like(expected),
        where=expected > 0,
    ).sum()
    phi_squared = chi_squared / total
    correction = ((column_count - 1) * (row_count - 1)) / (total - 1)
    corrected_phi = max(0.0, phi_squared - correction)
    corrected_rows = row_count - ((row_count - 1) ** 2) / (total - 1)
    corrected_columns = column_count - ((column_count - 1) ** 2) / (total - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.sqrt(corrected_phi / denominator), 0.0, 1.0))


def sample_ids(
    styles: pd.DataFrame,
    target: str,
    labels: Sequence[str],
    per_label: int,
    seed: int,
) -> tuple[str, ...]:
    if target not in styles:
        raise KeyError(f"Unknown target: {target}")
    if per_label < 1:
        raise ValueError("per_label must be at least 1")

    normalized = styles.assign(_label=_display_labels(styles[target]))
    selected: list[str] = []
    for offset, label in enumerate(labels):
        candidates = normalized.loc[normalized["_label"].eq(label), "id"]
        count = min(per_label, len(candidates))
        if count:
            selected.extend(
                candidates.sample(n=count, random_state=seed + offset).astype(str)
            )
    return tuple(selected)


def plot_image_grid(
    ids: Sequence[str],
    image_dir: Path,
    title: str,
    columns: int = 5,
) -> plt.Figure:
    """Build an image grid; the caller owns and must close the returned figure."""

    if columns < 1:
        raise ValueError("columns must be at least 1")
    rows = max(1, int(np.ceil(len(ids) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.2 * columns, 2.7 * rows),
        squeeze=False,
    )
    for axis, image_id in zip(axes.flat, ids):
        path = image_dir / f"{image_id}.jpg"
        try:
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            axis.text(0.5, 0.5, type(exc).__name__, ha="center", va="center")
        axis.set_title(str(image_id), fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(ids):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def build_dataset_overview(
    styles: pd.DataFrame,
    images: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """Save the dataset overview; the caller owns and must close the figure."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    article = target_distribution(styles, "articleType")
    axes[0, 0].plot(article["rank"], article["count"], color="#b33b24")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="articleType long tail", xlabel="Class rank", ylabel="Count")

    for axis, target in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        ("season", "gender", "usage"),
    ):
        distribution = target_distribution(styles, target)
        axis.bar(distribution["label"], distribution["count"], color="#345995")
        axis.set_title(target)
        axis.set_ylabel("Count")
        axis.tick_params(axis="x", rotation=35)
        if target in {"season", "usage"}:
            axis.set_yscale("log")
        for index, count in enumerate(distribution["count"]):
            axis.annotate(
                f"{int(count):,}",
                (index, count),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    readable_count = int(images["error"].eq("").sum())
    style_ids = set(styles["id"].astype(str))
    image_ids = set(images["id"].astype(str))
    unmatched_count = len(style_ids.symmetric_difference(image_ids))
    figure.suptitle(
        "Fashion dataset overview\n"
        f"{len(styles):,} metadata rows · {readable_count:,} readable images · "
        f"{unmatched_count:,} unmatched IDs"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure
