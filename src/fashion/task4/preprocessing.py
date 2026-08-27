"""Deterministic, fold-safe image preprocessing for Task 4 retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.config import ROOT
from fashion.data.images import StreamingStats, transform_image_with_mask

__all__ = (
    "PreprocessedImage",
    "PreprocessingContract",
    "fit_fold_rgb_statistics",
    "load_preprocessed_image",
    "normalize_for_model",
    "preprocess_image",
)


@dataclass(frozen=True)
class PreprocessingContract:
    """Fixed Task 4 geometry and colour contract."""

    width: int
    height: int
    pad_color: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, Integral)
            or not isinstance(self.height, Integral)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("width and height must be positive integers")
        if len(self.pad_color) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value <= 255
            for value in self.pad_color
        ):
            raise ValueError("pad_color must contain three integers in [0, 255]")

    @property
    def image_size(self) -> tuple[int, int]:
        """Return the NumPy/Pillow adapter order of ``(height, width)``."""
        return int(self.height), int(self.width)

    @property
    def pixel_count(self) -> int:
        return int(self.width) * int(self.height)

    @property
    def key(self) -> str:
        return f"rgb-letterbox-lanczos-{int(self.width)}x{int(self.height)}"

    def to_dict(self) -> dict[str, object]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "pad_color": list(self.pad_color),
            "colour_mode": "RGB",
            "resize": "aspect_preserving_letterbox",
            "resample": "LANCZOS",
        }


@dataclass(frozen=True)
class PreprocessedImage:
    """One transformed RGB image and its real-content location."""

    pixels: np.ndarray
    content_mask: np.ndarray
    content_bounds: tuple[int, int, int, int]


def _oriented_rgb(image: Image.Image, pad_color: tuple[int, int, int]) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    if oriented.width <= 0 or oriented.height <= 0:
        raise ValueError("source image must have positive width and height")
    if "A" in oriented.getbands() or "transparency" in oriented.info:
        foreground = oriented.convert("RGBA")
        background = Image.new("RGBA", foreground.size, (*pad_color, 255))
        return Image.alpha_composite(background, foreground).convert("RGB")
    return oriented.convert("RGB")


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("transformed image has no content pixels")
    return (
        int(rows.min()),
        int(columns.min()),
        int(rows.max()) + 1,
        int(columns.max()) + 1,
    )


def preprocess_image(
    image: Image.Image,
    contract: PreprocessingContract,
) -> PreprocessedImage:
    """Apply EXIF, RGB, LANCZOS letterboxing, and return lossless uint8 pixels."""
    rgb = _oriented_rgb(image, contract.pad_color)
    pixels, content_mask = transform_image_with_mask(
        rgb,
        image_size=contract.image_size,
        pad_color=contract.pad_color,
        normalize_range=False,
    )
    uint8_pixels = np.rint(pixels).clip(0, 255).astype(np.uint8)
    return PreprocessedImage(
        pixels=uint8_pixels,
        content_mask=content_mask,
        content_bounds=_mask_bounds(content_mask),
    )


def load_preprocessed_image(
    path: str | Path,
    contract: PreprocessingContract,
) -> PreprocessedImage:
    """Load one file and apply :func:`preprocess_image`."""
    with Image.open(path) as image:
        return preprocess_image(image, contract)


def normalize_for_model(
    image: PreprocessedImage,
    *,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
) -> np.ndarray:
    """Scale RGB with saved training statistics and make padding neutral zero."""
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if (
        mean_array.shape != (3,)
        or std_array.shape != (3,)
        or not np.isfinite(mean_array).all()
        or not np.isfinite(std_array).all()
        or np.any(std_array <= 0)
    ):
        raise ValueError("mean/std must contain three finite values with positive std")
    normalized = (
        image.pixels.astype(np.float32) / np.float32(255.0) - mean_array
    ) / std_array
    normalized[~image.content_mask] = 0.0
    return normalized


def _require_frame_columns(frame: pd.DataFrame, required: set[str]) -> None:
    if missing := required.difference(frame.columns):
        raise ValueError(f"preprocessing frame is missing columns: {sorted(missing)}")


def _training_id_digest(ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for product_id in ids:
        digest.update(f"{int(product_id)}\n".encode())
    return digest.hexdigest()


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def fit_fold_rgb_statistics(
    frame: pd.DataFrame,
    *,
    path_column: str,
    contract: PreprocessingContract,
    validation_fold: int,
    root: str | Path = ROOT,
) -> dict[str, object]:
    """Fit RGB population statistics on current-round training content only."""
    if (
        isinstance(validation_fold, bool)
        or not isinstance(validation_fold, Integral)
        or validation_fold not in range(5)
    ):
        raise ValueError("validation_fold must be an integer in range(5)")
    _require_frame_columns(frame, {"id", "partition", "cv_fold", path_column})
    if frame.empty or not frame["partition"].eq("development").all():
        raise ValueError("RGB statistics require development rows only")

    working = frame.copy()
    numeric_ids = pd.to_numeric(working["id"], errors="coerce")
    numeric_folds = pd.to_numeric(working["cv_fold"], errors="coerce")
    valid_ids = numeric_ids.notna() & numeric_ids.mod(1).eq(0)
    valid_folds = numeric_folds.notna() & numeric_folds.mod(1).eq(0)
    if not valid_ids.all() or not valid_folds.all():
        raise ValueError("IDs and CV folds must be integer-compatible")
    working["id"] = numeric_ids.astype(int)
    working["cv_fold"] = numeric_folds.astype(int)
    if working["id"].duplicated().any():
        raise ValueError("RGB statistics require unique product IDs")
    if not working["cv_fold"].isin(range(5)).all():
        raise ValueError("RGB statistics require CV folds in range(5)")

    training = (
        working.loc[working["cv_fold"].ne(int(validation_fold))]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if training.empty:
        raise ValueError("RGB statistics have no training rows")

    stats = StreamingStats()
    root_path = Path(root)
    for row in training.itertuples(index=False):
        transformed = load_preprocessed_image(
            _resolve_path(getattr(row, path_column), root_path),
            contract,
        )
        stats.update(
            transformed.pixels.astype(np.float32) / 255.0,
            content_mask=transformed.content_mask,
        )

    return {
        "validation_fold": int(validation_fold),
        "training_rows": len(training),
        "training_id_sha256": _training_id_digest(training["id"]),
        "content_pixels": stats.total_pixels,
        "mean": stats.mean,
        "std": stats.std,
        "contract": contract.to_dict(),
    }
