"""HOG feature extraction and cache contracts for Task 1."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from fashion.config import ROOT, TASK1_HOG_CACHE_DIR
from fashion.data.hashing import compute_sha256
from fashion.data.images import transform_image_with_mask
from fashion.task1.image_contract import TASK1_IMAGE_SIZE, TASK1_PAD_COLOR
from fashion.train.artifacts import atomic_write_bytes, canonical_json_bytes, canonical_sha256

HOG_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Task1HogSpec:
    hog_id: str
    pixels_per_cell: tuple[int, int]
    expected_features: int
    orientations: int = 9
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True
    image_size: tuple[int, int] = TASK1_IMAGE_SIZE
    pad_color: tuple[int, int, int] = TASK1_PAD_COLOR

    def __post_init__(self) -> None:
        if not self.hog_id.strip():
            raise ValueError("hog_id must not be blank")
        height, width = self.image_size
        cell_y, cell_x = self.pixels_per_cell
        block_y, block_x = self.cells_per_block
        if min(height, width, cell_y, cell_x, block_y, block_x, self.orientations) <= 0:
            raise ValueError("HOG geometry values must be positive")
        blocks_y, blocks_x = height // cell_y - block_y + 1, width // cell_x - block_x + 1
        calculated = blocks_y * blocks_x * block_y * block_x * self.orientations
        if blocks_y <= 0 or blocks_x <= 0 or calculated != self.expected_features:
            raise ValueError("HOG geometry does not match expected feature count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TASK1_HOG_COARSE = Task1HogSpec("task1_gray_hog_ppc16_v1", (16, 16), 288)
TASK1_HOG_FINE = Task1HogSpec("task1_gray_hog_ppc10_v1", (10, 10), 1260)
TASK1_HOG_SPECS = (TASK1_HOG_COARSE, TASK1_HOG_FINE)


@dataclass(frozen=True)
class Task1HogFeatureSet:
    ids: np.ndarray
    features: np.ndarray
    spec: Task1HogSpec
    cache_path: Path


def extract_task1_hog(path: str | Path, spec: Task1HogSpec) -> np.ndarray:
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            rgb, _ = transform_image_with_mask(
                source, image_size=spec.image_size, pad_color=spec.pad_color, normalize_range=True
            )
    except Exception as error:
        raise ValueError(f"cannot extract HOG from {image_path}") from error
    features = hog(
        rgb2gray(np.asarray(rgb, dtype=np.float32)),
        orientations=spec.orientations,
        pixels_per_cell=spec.pixels_per_cell,
        cells_per_block=spec.cells_per_block,
        block_norm=spec.block_norm,
        transform_sqrt=spec.transform_sqrt,
        feature_vector=True,
        channel_axis=None,
    ).astype(np.float32, copy=False)
    if features.shape != (spec.expected_features,) or not np.isfinite(features).all():
        raise ValueError(f"invalid HOG feature vector for {image_path}")
    return features


def _normalized_development_hog_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "path", "sha256", "partition"}
    if required.difference(rows.columns):
        raise ValueError("HOG rows are missing required inventory columns")
    if rows.empty or not rows["partition"].eq("development").all():
        raise ValueError("HOG cache requires unique development rows")
    if rows["id"].isna().any():
        raise ValueError("HOG cache requires unique development rows")
    normalized = rows.copy()
    try:
        normalized["id"] = normalized["id"].to_numpy(dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("HOG cache requires integer development row IDs") from error
    if normalized["id"].duplicated().any():
        raise ValueError("HOG cache requires unique development rows")
    return normalized.sort_values("id", kind="stable")


def _hog_inventory(rows: pd.DataFrame) -> list[dict[str, object]]:
    ordered = _normalized_development_hog_rows(rows)
    return ordered.loc[:, ["id", "path", "sha256"]].to_dict(orient="records")


def _hog_cache_implementation_identity() -> dict[str, object]:
    """Seal cached features to the code and packages that produce their values."""
    source_paths = {
        "classical_features.py": Path(__file__),
        "image_contract.py": ROOT / "src/fashion/task1/image_contract.py",
        "images.py": ROOT / "src/fashion/data/images.py",
    }
    return {
        "source_sha256": {
            name: compute_sha256(path) for name, path in sorted(source_paths.items())
        },
        "library_versions": {
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "scikit-image": version("scikit-image"),
        },
    }


def load_or_build_task1_hog_features(
    rows: pd.DataFrame,
    spec: Task1HogSpec,
    *,
    root: str | Path = ROOT,
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    split_sha256: str,
    extractor: Callable[[Path, Task1HogSpec], np.ndarray] = extract_task1_hog,
) -> Task1HogFeatureSet:
    ordered = _normalized_development_hog_rows(rows)
    inventory = _hog_inventory(ordered)
    identity = {
        "schema_version": HOG_CACHE_SCHEMA_VERSION,
        "split_sha256": split_sha256,
        "inventory_sha256": canonical_sha256(inventory),
        "hog": spec.to_dict(),
        "implementation": _hog_cache_implementation_identity(),
    }
    cache_path = Path(cache_root) / f"{spec.hog_id}-{canonical_sha256(identity)[:16]}.npz"
    expected_ids = np.asarray([int(row["id"]) for row in inventory], dtype=np.int64)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as stored:
                ids = stored["ids"]
                features = stored["features"]
                metadata = json.loads(bytes(stored["metadata"].tolist()).decode("utf-8"))
            valid = (
                ids.dtype == np.int64
                and features.dtype == np.float32
                and canonical_json_bytes(metadata) == canonical_json_bytes(identity)
                and np.array_equal(ids, expected_ids)
                and features.shape == (len(ids), spec.expected_features)
                and np.isfinite(features).all()
            )
            if valid:
                return Task1HogFeatureSet(ids, features, spec, cache_path)
        except Exception:
            pass

    ids = ordered["id"].to_numpy(dtype=np.int64)
    project_root = Path(root)
    vectors = [
        np.asarray(extractor(project_root / str(row.path), spec), dtype=np.float32)
        for row in ordered.itertuples()
    ]
    features = np.stack(vectors).astype(np.float32, copy=False)
    if features.shape != (len(ids), spec.expected_features) or not np.isfinite(features).all():
        raise ValueError("invalid Task 1 HOG cache features")
    metadata_bytes = canonical_json_bytes(identity)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        ids=ids,
        features=features,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    atomic_write_bytes(cache_path, buffer.getvalue())
    return Task1HogFeatureSet(ids, features, spec, cache_path)
