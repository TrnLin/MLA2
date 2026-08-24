from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.data.dataset import FashionDataset
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.images import (
    StreamingStats,
    load_and_transform_image,
    transform_image,
    transform_image_with_mask,
)


def test_sha256_is_deterministic(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"fashion")
    assert compute_sha256(path) == compute_sha256(path)
    assert len(compute_sha256(path)) == 64


def test_gzip_csv_bytes_are_path_independent_with_zero_mtime(tmp_path):
    frame = pd.DataFrame({"id": [2, 1], "value": ["same", "bytes"]})
    first = write_deterministic_csv(frame, tmp_path / "first.csv.gz", index=False)
    second = write_deterministic_csv(frame, tmp_path / "nested/second.csv.gz", index=False)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\0\0\0\0"
    assert gzip.decompress(first.read_bytes()).decode("utf-8") == ("id,value\n2,same\n1,bytes\n")
    pd.testing.assert_frame_equal(pd.read_csv(first), frame)


def test_transform_contract_is_float_rgb():
    transformed = transform_image(Image.new("L", (60, 80), 100))
    assert transformed.shape == (128, 96, 3)
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


def test_standardized_letterbox_padding_is_neutral():
    transformed, content = transform_image_with_mask(
        Image.new("RGB", (40, 80), (64, 96, 128)),
        image_size=(128, 96),
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )
    assert not content.all()
    assert np.all(transformed[~content] == 0)
    assert not np.all(transformed[content] == 0)


def test_streaming_stats_match_numpy():
    first = np.full((2, 2, 3), 0.25, dtype=np.float32)
    second = np.full((2, 2, 3), 0.75, dtype=np.float32)
    stats = StreamingStats()
    stats.update(first)
    stats.update(second)
    assert stats.mean == pytest.approx([0.5, 0.5, 0.5])
    assert stats.std == pytest.approx([0.25, 0.25, 0.25])


def test_streaming_stats_can_exclude_padding():
    image = np.ones((2, 3, 3), dtype=np.float32)
    image[:, 0] = 0
    mask = np.array([[False, True, True], [False, True, True]])
    stats = StreamingStats()
    stats.update(image, content_mask=mask)
    assert stats.total_pixels == 4
    assert stats.mean == pytest.approx([1.0, 1.0, 1.0])


def test_dataset_adapter_loads_relative_path(tiny_project):
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "path": "data/raw/teacher/train/images_train/1.jpg",
                "partition": "development",
                "cv_fold": 0,
            }
        ]
    )
    sample = FashionDataset(
        frame,
        transform=lambda path: load_and_transform_image(path, image_size=(128, 96)),
        root=tiny_project.root,
        targets=(),
    )[0]
    assert sample["id"] == 1
    assert sample["partition"] == "development"
    assert sample["cv_fold"] == 0
    assert sample["image"].shape == (128, 96, 3)
