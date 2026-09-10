"""Source-image intake checks and the legacy source-only audit adapter.

The combined training dataset uses ``expanded_usage`` and the normal FashionDataset.
This module retains the original source-record checks for reproducible image admission.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.config import ROOT
from fashion.data.images import StreamingStats, transform_image_with_mask

RARE_CLASSES = ("Home", "Party", "Smart Casual", "Travel", "NA")
ACCEPTED_EVIDENCE = {"source_exact_occasion", "source_exact_style"}
IMAGE_SIZE = (80, 60)  # Height, width; Pillow files are 60 by 80.


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded_sha256(image: Image.Image) -> str:
    """Hash oriented RGB pixels and geometry, independent of image encoding."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    prefix = f"RGB:{rgb.width}:{rgb.height}:".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def prepare_external_image(original: str | Path, destination: str | Path) -> dict[str, Any]:
    """Save deterministic RGB PNG using the shared CNN letterbox transform."""
    with Image.open(original) as image:
        original_decoded = decoded_sha256(image)
        array, mask = transform_image_with_mask(image, image_size=IMAGE_SIZE, normalize_range=False)
    rendered = Image.fromarray(array.astype(np.uint8))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination, format="PNG", optimize=False, compress_level=9)
    ys, xs = np.where(mask)
    return {
        "sha256": file_sha256(destination),
        "decoded_sha256": decoded_sha256(rendered),
        "original_decoded_sha256": original_decoded,
        "width": 60,
        "height": 80,
        "content_left": int(xs.min()),
        "content_top": int(ys.min()),
        "content_width": int(xs.max() - xs.min() + 1),
        "content_height": int(ys.max() - ys.min() + 1),
    }


def _bools(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false", "1", "0"}).all():
        raise ValueError("boolean manifest fields must be explicit")
    return values.isin({"true", "1"})


def _image_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("image path escapes the dataset root")
    return path


def _mask(row: pd.Series) -> np.ndarray:
    values = np.asarray(
        [row[key] for key in ("content_left", "content_top", "content_width", "content_height")],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("content rectangle coordinates must be finite integers")
    left, top, width, height = map(int, values)
    if min(left, top) < 0 or min(width, height) < 1 or left + width > 60 or top + height > 80:
        raise ValueError("invalid content rectangle")
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def assign_external_partitions(
    frame: pd.DataFrame, *, seed: int = 2753, validation_fraction: float = 0.2
) -> pd.DataFrame:
    """Allocate complete source families; a singleton stratum stays train-only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    result = frame.copy()
    result["external_partition"] = "train"
    if result.empty:
        return result
    for _, group in result.groupby("external_group_id"):
        if group["source_label"].nunique() != 1:
            raise ValueError("a source family has conflicting labels")
    families = result.groupby("external_group_id", sort=True).agg(
        source_label=("source_label", "first"),
        source_stratum=("source", lambda x: "+".join(sorted(set(x)))),
    )
    for _, stratum in families.groupby(["source_label", "source_stratum"], sort=True):
        ids = sorted(
            stratum.index,
            key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(),
        )
        if len(ids) < 2:
            continue
        count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
        result.loc[result["external_group_id"].isin(ids[:count]), "external_partition"] = (
            "validation"
        )
    return result


def validate_external_manifest(
    frame: pd.DataFrame, *, root: str | Path = ROOT, check_files: bool = False
) -> None:
    """Fail closed on weak labels, role mixing, group leaks, or corrupt prepared files."""
    required = {
        "external_id",
        "source",
        "source_id",
        "source_label",
        "label_strength",
        "label_evidence",
        "source_url",
        "image_url",
        "rights_basis",
        "external_group_id",
        "external_partition",
        "path",
        "sha256",
        "decoded_sha256",
        "original_sha256",
        "original_decoded_sha256",
        "width",
        "height",
        "content_left",
        "content_top",
        "content_width",
        "content_height",
        "teacher_overlap",
        "teacher_usage_compatible",
        "visual_decision",
        "training_target",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"external manifest missing columns: {sorted(missing)}")
    if frame.empty:
        return
    for column in required - {"teacher_overlap", "teacher_usage_compatible"}:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"blank external field: {column}")
    if frame.external_id.duplicated().any():
        raise ValueError("external IDs must be unique")
    if not frame.source_label.isin(RARE_CLASSES).all():
        raise ValueError("only selected rare source classes are allowed")
    if frame.source_label.eq("NA").any():
        raise ValueError("no source-compatible NA definition has been established")
    if not frame.label_strength.isin(ACCEPTED_EVIDENCE).all():
        raise ValueError("weak text cannot enter the accepted source-label set")
    if _bools(frame.teacher_overlap).any():
        raise ValueError("teacher overlap must be excluded")
    if (
        _bools(frame.teacher_usage_compatible).any()
        or not frame.training_target.eq("source_usage").all()
    ):
        raise ValueError("external rows are source-label supervision, not teacher Usage targets")
    if not frame.visual_decision.eq("keep").all():
        raise ValueError("all admitted images need a visual keep decision")
    if not frame.external_partition.isin({"train", "validation"}).all():
        raise ValueError("unknown external partition")
    for column in ("external_group_id", "sha256", "decoded_sha256", "original_decoded_sha256"):
        if (frame.groupby(column).external_partition.nunique() > 1).any():
            raise ValueError(f"{column} crosses external partitions")
    if (frame.groupby("external_group_id").source_label.nunique() > 1).any():
        raise ValueError("conflicting source labels in a family")
    for column, expected in (("width", 60), ("height", 80)):
        if not pd.to_numeric(frame[column], errors="raise").eq(expected).all():
            raise ValueError("prepared image dimensions must be 60 by 80")
    root = Path(root)
    for _, row in frame.iterrows():
        _mask(row)
        path = _image_path(root, str(row.path))
        if check_files:
            if file_sha256(path) != row.sha256:
                raise ValueError(f"image hash mismatch: {row.external_id}")
            with Image.open(path) as im:
                im.load()
                if im.size != (60, 80) or im.mode != "RGB" or im.format != "PNG":
                    raise ValueError("prepared images must be RGB 60 by 80 PNG")
                if decoded_sha256(im) != row.decoded_sha256:
                    raise ValueError("decoded image hash mismatch")


def load_external_manifest(path: str | Path, *, root: str | Path = ROOT) -> pd.DataFrame:
    """Load the explicit auxiliary manifest without interpreting literal NA as missing."""
    frame = pd.read_csv(path, keep_default_na=False)
    validate_external_manifest(frame, root=root, check_files=True)
    return frame


def fit_external_statistics(frame: pd.DataFrame, *, root: str | Path = ROOT) -> dict[str, Any]:
    """Fit optional source-only RGB statistics on training content pixels only."""
    validate_external_manifest(frame, root=root, check_files=True)
    if frame.empty or not frame.external_partition.eq("train").all():
        raise ValueError("statistics require a nonempty training-only frame")
    stats = StreamingStats()
    for _, row in frame.iterrows():
        with Image.open(_image_path(Path(root), str(row.path))) as im:
            array = np.asarray(im, dtype=np.float32) / 255.0
        stats.update(array, _mask(row))
    return {
        "mean": stats.mean,
        "std": stats.std,
        "content_pixels": stats.total_pixels,
        "scope": "external train only",
        "training_ids": sorted(frame.external_id.tolist()),
    }


class ExternalUsageDataset:
    """Numpy CHW samples for a separate source-label head; torch DataLoader compatible."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        partition: str,
        label_to_index: dict[str, int],
        mean: list[float] | tuple[float, ...],
        std: list[float] | tuple[float, ...],
        root: str | Path = ROOT,
        target: str = "source_usage",
    ) -> None:
        if target != "source_usage":
            raise ValueError("this dataset only supports the source_usage auxiliary target")
        validate_external_manifest(frame, root=root, check_files=True)
        if partition not in {"train", "validation"}:
            raise ValueError("select external train or validation explicitly")
        self.frame = frame[frame.external_partition.eq(partition)].reset_index(drop=True)
        self.root = Path(root)
        self.label_to_index = dict(label_to_index)
        indices = list(self.label_to_index.values())
        if any(type(value) is not int or value < 0 for value in indices) or len(
            set(indices)
        ) != len(indices):
            raise ValueError("source label indices must be unique nonnegative integers")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("normalization requires finite mean and positive std")
        if unknown := set(self.frame.source_label) - set(self.label_to_index):
            raise ValueError(f"unknown source labels: {sorted(unknown)}")
        self.group_sizes = self.frame.external_group_id.value_counts()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(_image_path(self.root, str(row.path))) as im:
            array = np.asarray(im, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        array[~_mask(row)] = 0.0
        return {
            "image": np.transpose(array, (2, 0, 1)).copy(),
            "label": int(self.label_to_index[str(row.source_label)]),
            "external_id": str(row.external_id),
            "source_label": str(row.source_label),
            "external_group_id": str(row.external_group_id),
            "sample_weight": 1.0 / int(self.group_sizes[row.external_group_id]),
            "path": str(row.path),
            "training_target": "source_usage",
        }
