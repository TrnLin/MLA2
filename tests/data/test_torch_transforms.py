from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fashion.data.torch import FoldImageStats, build_image_transform, fit_fold_stats
from fashion.train.reproducibility import seed_everything


def _write_image(path: Path, colour: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def _training_frame(root: Path) -> pd.DataFrame:
    first = Path("images/1.png")
    second = Path("images/2.png")
    _write_image(root / first, (0, 64, 128), (60, 80))
    _write_image(root / second, (255, 192, 64), (30, 80))
    return pd.DataFrame(
        {
            "id": [1, 2],
            "path": [first.as_posix(), second.as_posix()],
            "partition": ["development", "development"],
            "cv_fold": [0, 0],
        }
    )


def test_fit_fold_stats_uses_training_content_not_padding(tmp_path: Path) -> None:
    frame = _training_frame(tmp_path)

    stats = fit_fold_stats(
        frame,
        validation_fold=1,
        image_size=(80, 60),
        root=tmp_path,
    )

    assert stats.image_count == 2
    assert stats.content_pixel_count == (80 * 60) + (80 * 30)
    assert stats.mean == pytest.approx((0.5, 128 / 255, 96 / 255), abs=2e-3)
    assert all(value > 0 for value in stats.std)
    assert len(stats.training_id_sha256) == 64


def test_fit_fold_stats_rejects_validation_or_protected_rows(tmp_path: Path) -> None:
    frame = _training_frame(tmp_path)
    frame.loc[1, "cv_fold"] = 1
    with pytest.raises(ValueError, match="validation-fold"):
        fit_fold_stats(frame, validation_fold=1, image_size=(80, 60), root=tmp_path)

    frame.loc[1, "cv_fold"] = 0
    frame.loc[1, "partition"] = "holdout"
    with pytest.raises(ValueError, match="development rows only"):
        fit_fold_stats(frame, validation_fold=1, image_size=(80, 60), root=tmp_path)


def test_evaluation_transform_returns_neutral_padded_chw_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "tall.png"
    _write_image(image_path, (64, 96, 128), (20, 80))
    stats = FoldImageStats(
        validation_fold=0,
        image_size=(80, 60),
        image_count=2,
        content_pixel_count=9600,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        training_id_sha256="a" * 64,
    )

    tensor = build_image_transform(stats, training=False)(image_path)

    assert tensor.shape == (3, 80, 60)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor[:, :, :20] == 0)
    assert torch.all(tensor[:, :, 40:] == 0)
    assert not torch.all(tensor[:, :, 20:40] == 0)


@pytest.mark.parametrize("policy", ["a0", "a1"])
def test_training_augmentation_is_seeded_and_identified(tmp_path: Path, policy: str) -> None:
    image_path = tmp_path / "pattern.png"
    pattern = np.zeros((80, 60, 3), dtype=np.uint8)
    pattern[:, :30] = (255, 32, 64)
    Image.fromarray(pattern).save(image_path)
    stats = FoldImageStats(
        validation_fold=0,
        image_size=(80, 60),
        image_count=2,
        content_pixel_count=9600,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        training_id_sha256="a" * 64,
    )
    transform = build_image_transform(stats, training=True, augmentation=policy)

    seed_everything(2753)
    first = transform(image_path)
    seed_everything(2753)
    second = transform(image_path)

    assert torch.equal(first, second)
    assert transform.spec.transform_id.startswith(f"{policy}-")


def test_validation_transform_rejects_random_augmentation() -> None:
    stats = FoldImageStats(
        validation_fold=0,
        image_size=(80, 60),
        image_count=1,
        content_pixel_count=4800,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        training_id_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="cannot use random"):
        build_image_transform(stats, training=False, augmentation="a0")
