from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fashion.task1.dataset import Task1TorchDataset, get_task1_fold_rows


def _rows(*, partition: str = "development", article_type: str = "Shirts") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 11,
                "path": "images/11.png",
                "partition": partition,
                "articleType": article_type,
            },
            {
                "id": 12,
                "path": "images/12.png",
                "partition": partition,
                "articleType": "Shoes",
            },
        ]
    )


def _save_images(root: Path) -> None:
    (root / "images").mkdir()
    Image.new("RGB", (60, 80), (10, 20, 30)).save(root / "images/11.png")
    Image.new("RGB", (60, 80), (30, 20, 10)).save(root / "images/12.png")


def _transform(_: Path) -> np.ndarray:
    return np.zeros((3, 80, 60), dtype=np.float32)


def test_dataset_converts_image_and_labels_to_tensors(tmp_path: Path) -> None:
    _save_images(tmp_path)
    dataset = Task1TorchDataset(
        _rows(),
        transform=_transform,
        label_to_index={"Shirts": 0, "Shoes": 1},
        root=tmp_path,
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (3, 80, 60)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.long
    assert sample["label"].item() == 0
    assert sample["id"].dtype == torch.long
    assert sample["id"].item() == 11


@pytest.mark.parametrize("partition", ["holdout", "quarantine"])
def test_dataset_rejects_protected_rows(tmp_path: Path, partition: str) -> None:
    with pytest.raises(ValueError, match="development"):
        Task1TorchDataset(
            _rows(partition=partition),
            transform=_transform,
            label_to_index={"Shirts": 0, "Shoes": 1},
            root=tmp_path,
        )


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = _rows()
    rows.loc[1, "id"] = rows.loc[0, "id"]

    with pytest.raises(ValueError, match="present and unique"):
        Task1TorchDataset(
            rows,
            transform=_transform,
            label_to_index={"Shirts": 0, "Shoes": 1},
            root=tmp_path,
        )


def test_dataset_rejects_blank_paths(tmp_path: Path) -> None:
    rows = _rows()
    rows.loc[1, "path"] = "  "

    with pytest.raises(ValueError, match="must not be blank"):
        Task1TorchDataset(
            rows,
            transform=_transform,
            label_to_index={"Shirts": 0, "Shoes": 1},
            root=tmp_path,
        )


def test_dataset_rejects_unknown_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown articleType labels"):
        Task1TorchDataset(
            _rows(article_type="Hats"),
            transform=_transform,
            label_to_index={"Shirts": 0, "Shoes": 1},
            root=tmp_path,
        )


def test_dataset_rejects_wrong_transform_shape(tmp_path: Path) -> None:
    def transform(_: Path) -> np.ndarray:
        return np.zeros((80, 60, 3), dtype=np.float32)

    dataset = Task1TorchDataset(
        _rows(),
        transform=transform,
        label_to_index={"Shirts": 0, "Shoes": 1},
        root=tmp_path,
    )

    with pytest.raises(ValueError, match=r"float32 shape \(3, 80, 60\)"):
        dataset[0]


def test_dataset_rejects_wrong_transform_dtype(tmp_path: Path) -> None:
    def transform(_: Path) -> np.ndarray:
        return np.zeros((3, 80, 60), dtype=np.float64)

    dataset = Task1TorchDataset(
        _rows(),
        transform=transform,
        label_to_index={"Shirts": 0, "Shoes": 1},
        root=tmp_path,
    )

    with pytest.raises(ValueError, match=r"float32 shape \(3, 80, 60\)"):
        dataset[0]


def _split_rows() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        rows.append(
            {
                "id": fold + 1,
                "sha256": f"sha-{fold}",
                "duplicate_group": f"dup-{fold}",
                "product_name_key": f"name-{fold}",
                "product_family_group": f"family-{fold}",
                "partition": "development",
                "cv_fold": fold,
                "is_cross_role_exact_duplicate": False,
                "is_cross_role_near_duplicate": False,
                "has_conflicting_target_labels": False,
                "conflicting_targets": "",
                "quarantine_reason": "",
                "articleType": "Shirts" if fold != 1 else "",
                "season": "Summer",
                "gender": "Unisex",
                "usage": "Casual",
                "has_articleType_label": fold != 1,
                "has_season_label": True,
                "has_gender_label": True,
                "has_usage_label": True,
            }
        )
    return pd.DataFrame(rows)


def test_get_task1_fold_rows_uses_precomputed_disjoint_folds_and_valid_labels() -> None:
    training, validation = get_task1_fold_rows(_split_rows(), validation_fold=2)

    assert set(training["id"]).isdisjoint(set(validation["id"]))
    assert set(validation["id"]) == {3}
    assert 2 not in set(training["id"])
    assert set(training["id"]) == {1, 4, 5}
    assert training["articleType"].notna().all()
    assert training["articleType"].eq("Shirts").all()
