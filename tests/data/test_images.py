from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.data.dataset import FashionDataset
from fashion.data.hashing import compute_sha256
from fashion.data.images import StreamingStats, transform_image


def test_sha256_is_deterministic(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"fashion")
    assert compute_sha256(path) == compute_sha256(path)
    assert len(compute_sha256(path)) == 64


def test_transform_contract_is_float_rgb():
    transformed = transform_image(Image.new("L", (60, 80), 100))
    assert transformed.shape == (128, 128, 3)
    assert transformed.dtype == np.float32
    assert transformed.min() >= 0
    assert transformed.max() <= 1


def test_transform_letterboxes_without_stretching():
    transformed = transform_image(Image.new("RGB", (40, 80), (0, 0, 0)), image_size=100)
    assert np.allclose(transformed[:, :25], 1.0)
    assert np.allclose(transformed[:, 25:75], 0.0)
    assert np.allclose(transformed[:, 75:], 1.0)


def test_transform_requires_mean_and_std_together():
    with pytest.raises(ValueError, match="together"):
        transform_image(Image.new("RGB", (2, 2)), mean=(0.5, 0.5, 0.5))


def test_transform_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="positive"):
        transform_image(
            Image.new("RGB", (2, 2)),
            mean=(0.5, 0.5, 0.5),
            std=(1.0, 0.0, 1.0),
        )


def test_streaming_stats_match_numpy():
    first = np.full((2, 2, 3), 0.25, dtype=np.float32)
    second = np.full((2, 2, 3), 0.75, dtype=np.float32)
    stats = StreamingStats()
    stats.update(first)
    stats.update(second)
    assert stats.mean == pytest.approx([0.5, 0.5, 0.5])
    assert stats.std == pytest.approx([0.25, 0.25, 0.25])


def test_dataset_adapter_loads_relative_path(tiny_project):
    frame = pd.DataFrame(
        [{"id": 1, "path": "data/raw/teacher/train/images_train/1.jpg", "partition": "train"}]
    )
    sample = FashionDataset(frame, root=tiny_project.root, targets=())[0]
    assert sample["id"] == 1
    assert sample["partition"] == "train"
    assert sample["image"].shape == (128, 128, 3)
