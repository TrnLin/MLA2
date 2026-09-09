"""Deterministic query-image conditions declared before the holdout unlock."""

from __future__ import annotations

import io

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from fashion.task4.preprocessing_experiment import build_odd_aspect_canvas

QUERY_CONDITIONS: tuple[str, ...] = (
    "clean",
    "jpeg_quality_85",
    "gaussian_blur_radius_1",
    "brightness_0_85",
    "brightness_1_15",
    "wide_canvas",
    "tall_canvas",
)


def _rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _jpeg_round_trip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    _rgb(image).save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        return reopened.convert("RGB")


def apply_query_condition(image: Image.Image, condition: str) -> Image.Image:
    """Apply one frozen condition to a query image before the 240x320 contract."""
    if condition not in QUERY_CONDITIONS:
        raise ValueError(f"unknown query condition: {condition}")
    if condition == "clean":
        return _rgb(image)
    if condition == "jpeg_quality_85":
        return _jpeg_round_trip(image, 85)
    if condition == "gaussian_blur_radius_1":
        return _rgb(image).filter(ImageFilter.GaussianBlur(radius=1))
    if condition == "brightness_0_85":
        return ImageEnhance.Brightness(_rgb(image)).enhance(0.85)
    if condition == "brightness_1_15":
        return ImageEnhance.Brightness(_rgb(image)).enhance(1.15)
    orientation = "wide" if condition == "wide_canvas" else "tall"
    return build_odd_aspect_canvas(image, orientation)


__all__ = ["QUERY_CONDITIONS", "apply_query_condition"]
