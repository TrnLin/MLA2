"""Classic HOG feature and model candidates for Task 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from fashion.data.images import transform_image_with_mask


@dataclass(frozen=True)
class Task1HogSpec:
    hog_id: str
    pixels_per_cell: tuple[int, int]
    expected_features: int
    orientations: int = 9
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True
    image_size: tuple[int, int] = (80, 60)
    pad_color: tuple[int, int, int] = (255, 255, 255)

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
