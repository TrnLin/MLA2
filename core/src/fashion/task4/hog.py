"""Deterministic, mask-aware plain-NumPy HOG for Task 4."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fashion.task4.cache import DevelopmentImageCache
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import CachedFeatureIndex, FeatureIndex

HOG_DESCRIPTOR_DIM = 1_728
_CONTRACT = PreprocessingContract(width=240, height=320)
_EPSILON = np.float32(1e-6)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HOGConfig:
    """Complete identity of the bounded HOG comparison descriptor."""

    input_height: int = 320
    input_width: int = 240
    gaussian_kernel: tuple[int, ...] = (1, 4, 6, 4, 1)
    gaussian_kernel_normalizer: int = 16
    unsigned_orientation_bins: int = 8
    cell_size_pixels: int = 32
    block_size_cells: int = 2
    block_stride_cells: int = 1
    l2_hys_clip: float = 0.2

    def __post_init__(self) -> None:
        if (self.input_height, self.input_width) != (320, 240):
            raise ValueError("HOG input must use the frozen 320x240 array shape")
        if self.gaussian_kernel != (1, 4, 6, 4, 1):
            raise ValueError("HOG requires the frozen 5x5 Gaussian kernel")
        if self.gaussian_kernel_normalizer != 16:
            raise ValueError("HOG Gaussian normalizer must be 16")
        if self.unsigned_orientation_bins != 8:
            raise ValueError("HOG requires exactly 8 unsigned orientation bins")
        if self.cell_size_pixels != 32:
            raise ValueError("HOG requires 32x32-pixel cells")
        if self.block_size_cells != 2 or self.block_stride_cells != 1:
            raise ValueError("HOG requires 2x2-cell blocks with 50% overlap")
        if not np.isfinite(self.l2_hys_clip) or not 0 < self.l2_hys_clip <= 1:
            raise ValueError("HOG L2-Hys clip must be finite in (0, 1]")
        if self.descriptor_dimension != HOG_DESCRIPTOR_DIM:
            raise ValueError("HOG configuration does not produce 1,728 values")

    @property
    def descriptor_dimension(self) -> int:
        cell_rows = self.input_height // self.cell_size_pixels
        cell_columns = self.input_width // self.cell_size_pixels
        block_rows = (
            cell_rows - self.block_size_cells
        ) // self.block_stride_cells + 1
        block_columns = (
            cell_columns - self.block_size_cells
        ) // self.block_stride_cells + 1
        return (
            block_rows
            * block_columns
            * self.block_size_cells
            * self.block_size_cells
            * self.unsigned_orientation_bins
        )

    @property
    def method(self) -> str:
        clip = f"{self.l2_hys_clip:g}".replace(".", "")
        return (
            "hog-luma-g5-u8-"
            f"c{self.cell_size_pixels}-b{self.block_size_cells}-"
            f"s{self.block_stride_cells}-l2hys{clip}-v1"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "input_height": self.input_height,
            "input_width": self.input_width,
            "grayscale": "rec709-luminance",
            "gaussian_kernel": list(self.gaussian_kernel),
            "gaussian_kernel_normalizer": self.gaussian_kernel_normalizer,
            "unsigned_orientation_bins": self.unsigned_orientation_bins,
            "cell_size_pixels": self.cell_size_pixels,
            "block_size_cells": self.block_size_cells,
            "block_stride_cells": self.block_stride_cells,
            "l2_hys_clip": self.l2_hys_clip,
            "mask_policy": "normalized-blur-and-valid-central-differences",
            "zero_gradient_policy": "canonical-unit-sentinel",
            "descriptor_dimension": self.descriptor_dimension,
        }

    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))


HOG_CONFIG = HOGConfig()
HOG_METHOD = HOG_CONFIG.method
HOG_CONFIG_FINGERPRINT = HOG_CONFIG.fingerprint()

__all__ = (
    "HOG_CONFIG",
    "HOG_CONFIG_FINGERPRINT",
    "HOG_DESCRIPTOR_DIM",
    "HOG_METHOD",
    "HOGConfig",
    "ensure_hog_feature_index",
    "extract_hog",
)


def _convolve_axis(
    values: np.ndarray,
    kernel: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    output = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        output += np.float32(weight) * padded[tuple(slices)]
    return output


def _masked_gaussian(
    luminance: np.ndarray,
    mask: np.ndarray,
    config: HOGConfig,
) -> np.ndarray:
    kernel = np.asarray(config.gaussian_kernel, dtype=np.float32)
    kernel /= np.float32(config.gaussian_kernel_normalizer)
    weighted = luminance * mask
    numerator = _convolve_axis(
        _convolve_axis(weighted, kernel, axis=1),
        kernel,
        axis=0,
    )
    denominator = _convolve_axis(
        _convolve_axis(mask.astype(np.float32), kernel, axis=1),
        kernel,
        axis=0,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _cell_histograms(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    config: HOGConfig,
) -> np.ndarray:
    cell_size = config.cell_size_pixels
    cell_rows = config.input_height // cell_size
    cell_columns = config.input_width // cell_size
    used_height = cell_rows * cell_size
    used_width = cell_columns * cell_size
    magnitude = magnitude[:used_height, :used_width]
    scaled = (
        orientation[:used_height, :used_width]
        * np.float32(config.unsigned_orientation_bins / np.pi)
    )
    lower = np.floor(scaled).astype(np.int64) % config.unsigned_orientation_bins
    fraction = scaled - np.floor(scaled)
    upper = (lower + 1) % config.unsigned_orientation_bins

    row_cells = np.arange(used_height, dtype=np.int64)[:, None] // cell_size
    column_cells = np.arange(used_width, dtype=np.int64)[None, :] // cell_size
    cells = np.broadcast_to(
        row_cells * cell_columns + column_cells,
        magnitude.shape,
    ).ravel()
    histograms = np.zeros(
        (cell_rows * cell_columns, config.unsigned_orientation_bins),
        dtype=np.float32,
    )
    np.add.at(
        histograms,
        (cells, lower.ravel()),
        (magnitude * (np.float32(1.0) - fraction)).ravel(),
    )
    np.add.at(
        histograms,
        (cells, upper.ravel()),
        (magnitude * fraction).ravel(),
    )
    return histograms.reshape(
        cell_rows,
        cell_columns,
        config.unsigned_orientation_bins,
    )


def _block_descriptor(histograms: np.ndarray, config: HOGConfig) -> np.ndarray:
    parts: list[np.ndarray] = []
    block = config.block_size_cells
    stride = config.block_stride_cells
    for row in range(0, histograms.shape[0] - block + 1, stride):
        for column in range(0, histograms.shape[1] - block + 1, stride):
            values = histograms[row : row + block, column : column + block].ravel()
            normalized = values / np.sqrt(np.dot(values, values) + _EPSILON**2)
            clipped = np.minimum(normalized, np.float32(config.l2_hys_clip))
            clipped /= np.sqrt(np.dot(clipped, clipped) + _EPSILON**2)
            parts.append(clipped)
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_hog(
    pixels: np.ndarray,
    content_mask: np.ndarray,
    config: HOGConfig = HOG_CONFIG,
) -> np.ndarray:
    """Extract one finite, unit-normalized 1,728-value HOG descriptor."""

    pixels = np.asarray(pixels)
    content_mask = np.asarray(content_mask)
    if pixels.shape != (config.input_height, config.input_width, 3):
        raise ValueError("HOG expects frozen 320x240 HWC RGB pixels")
    if pixels.dtype != np.uint8:
        raise ValueError("HOG pixels must have uint8 dtype")
    if content_mask.shape != pixels.shape[:2]:
        raise ValueError("HOG requires a matching content mask")
    if content_mask.dtype != bool:
        raise ValueError("HOG content mask must be boolean")
    if not content_mask.any():
        raise ValueError("HOG requires at least one content pixel")

    rgb = pixels.astype(np.float32) / np.float32(255.0)
    luminance = (
        np.float32(0.2126) * rgb[..., 0]
        + np.float32(0.7152) * rgb[..., 1]
        + np.float32(0.0722) * rgb[..., 2]
    )
    blurred = _masked_gaussian(luminance, content_mask, config)
    gradient_x = np.zeros_like(blurred)
    gradient_y = np.zeros_like(blurred)
    gradient_x[:, 1:-1] = (blurred[:, 2:] - blurred[:, :-2]) / np.float32(2.0)
    gradient_y[1:-1] = (blurred[2:] - blurred[:-2]) / np.float32(2.0)
    valid_x = np.zeros_like(content_mask)
    valid_y = np.zeros_like(content_mask)
    valid_x[:, 1:-1] = (
        content_mask[:, :-2]
        & content_mask[:, 1:-1]
        & content_mask[:, 2:]
    )
    valid_y[1:-1] = (
        content_mask[:-2]
        & content_mask[1:-1]
        & content_mask[2:]
    )
    gradient_x[~valid_x] = 0
    gradient_y[~valid_y] = 0
    magnitude = np.hypot(gradient_x, gradient_y).astype(np.float32)
    orientation = np.mod(
        np.arctan2(gradient_y, gradient_x),
        np.float32(np.pi),
    ).astype(np.float32)
    descriptor = _block_descriptor(
        _cell_histograms(magnitude, orientation, config),
        config,
    )
    if descriptor.shape != (HOG_DESCRIPTOR_DIM,) or not np.isfinite(descriptor).all():
        raise ValueError("HOG descriptor is malformed or non-finite")
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(norm):
        raise ValueError("HOG descriptor norm is non-finite")
    if norm == 0:
        descriptor[0] = np.float32(1.0)
        return descriptor
    descriptor = descriptor / np.float32(norm)
    return descriptor.astype(np.float32, copy=False)


def _mask_from_bounds(
    bounds: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    top, left, bottom, right = (int(value) for value in bounds)
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("HOG cache contains invalid content bounds")
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _expected_manifest(
    cache: DevelopmentImageCache,
    config: HOGConfig,
) -> dict[str, object]:
    if cache.manifest.get("scope") != "development":
        raise ValueError("HOG feature cache requires development images only")
    if cache.manifest.get("contract") != _CONTRACT.to_dict():
        raise ValueError("HOG feature cache requires the frozen 240x320 preprocessing")
    source = cache.manifest.get("source")
    if source not in {"teacher", "v1"}:
        raise ValueError("HOG feature cache source must be teacher or v1")
    return {
        "schema_version": "1.0.0",
        "scope": "development",
        "fold": 1,
        "method": config.method,
        "descriptor_config": config.to_dict(),
        "descriptor_fingerprint": config.fingerprint(),
        "checkpoint_fingerprint": None,
        "config_fingerprint": None,
        "source": source,
        "rows": len(cache.ids),
        "dimension": HOG_DESCRIPTOR_DIM,
        "contract": _CONTRACT.to_dict(),
        "image_cache_manifest_sha256": _sha256_bytes(
            _canonical_json(cache.manifest)
        ),
        "source_fingerprint": cache.manifest.get("source_fingerprint"),
        "id_sha256": cache.manifest.get("id_sha256"),
    }


def _load_hog_cache(
    directory: Path,
    expected: dict[str, object],
) -> CachedFeatureIndex:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("HOG feature cache identity does not match")
    ids_path = directory / "ids.npy"
    features_path = directory / "features.npy"
    if (
        _sha256_file(ids_path) != manifest.get("ids_file_sha256")
        or _sha256_file(features_path) != manifest.get("features_file_sha256")
    ):
        raise ValueError("HOG feature cache files fail their hashes")
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if (
        ids.dtype != np.int64
        or ids.shape != (int(expected["rows"]),)
        or features.dtype != np.float32
        or features.shape != (int(expected["rows"]), HOG_DESCRIPTOR_DIM)
        or not np.array_equal(ids, np.sort(ids))
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("HOG feature cache arrays are malformed")
    for start in range(0, len(features), 1024):
        batch = np.asarray(features[start : start + 1024])
        if not np.isfinite(batch).all() or not np.allclose(
            np.linalg.norm(batch, axis=1),
            1.0,
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("HOG feature cache descriptors are invalid")
    index = FeatureIndex(
        source=str(expected["source"]),
        contract=_CONTRACT,
        ids=ids,
        features=features,
        transform_seconds=float(manifest["transform_seconds"]),
        source_bytes=int(manifest["source_bytes"]),
        method=str(expected["method"]),
        fold=1,
        checkpoint_fingerprint=None,
        config_fingerprint=None,
        descriptor_fingerprint=str(expected["descriptor_fingerprint"]),
    )
    return CachedFeatureIndex(cache_dir=directory, index=index, manifest=manifest)


@contextmanager
def _cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_hog_cache(
    directory: Path,
    cache: DevelopmentImageCache,
    expected: dict[str, object],
    *,
    config: HOGConfig,
    workers: int,
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent)
    )
    try:
        np.save(temporary / "ids.npy", np.asarray(cache.ids), allow_pickle=False)
        feature_memmap = np.lib.format.open_memmap(
            temporary / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cache.ids), HOG_DESCRIPTOR_DIM),
        )

        def encode(index: int) -> None:
            mask = _mask_from_bounds(
                np.asarray(cache.content_bounds[index]),
                height=config.input_height,
                width=config.input_width,
            )
            feature_memmap[index] = extract_hog(
                np.asarray(cache.images[index]),
                mask,
                config,
            )

        started = time.perf_counter()
        if workers == 1:
            for index in range(len(cache.ids)):
                encode(index)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(encode, range(len(cache.ids))))
        feature_memmap.flush()
        manifest = {
            **expected,
            "transform_seconds": time.perf_counter() - started,
            "source_bytes": int(cache.images.nbytes),
            "ids_file_sha256": _sha256_file(temporary / "ids.npy"),
            "features_file_sha256": _sha256_file(temporary / "features.npy"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def ensure_hog_feature_index(
    cache: DevelopmentImageCache,
    *,
    cache_root: str | Path,
    workers: int,
    config: HOGConfig = HOG_CONFIG,
) -> CachedFeatureIndex:
    """Reuse or build one exact cache-backed HOG feature index."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("HOG workers must be a positive integer")
    expected = _expected_manifest(cache, config)
    identity = _sha256_bytes(_canonical_json(expected))[:20]
    directory = Path(cache_root) / str(expected["source"]) / identity
    lock_path = directory.parent / f".{identity}.lock"
    with _cache_lock(lock_path):
        if directory.is_dir():
            try:
                return _load_hog_cache(directory, expected)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        _build_hog_cache(
            directory,
            cache,
            expected,
            config=config,
            workers=workers,
        )
        return _load_hog_cache(directory, expected)
