from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fashion.data.pipeline import prepare_data


@dataclass(frozen=True)
class TinyProject:
    root: Path
    train_csv: Path
    prediction_csv: Path
    train_images: Path
    prediction_images: Path
    processed: Path
    audit: Path
    prediction_manifest: Path
    label_maps: Path
    splits: Path
    split_summary: Path
    development_summary: Path
    normalization: Path
    taxonomy: Path
    paired_normalization: Path


def _save_image(path: Path, color: tuple[int, int, int], mode: str = "RGB") -> None:
    image = Image.new("RGB", (60, 80), color=color)
    if mode == "L":
        image = image.convert("L")
    image.save(path, "JPEG", quality=95)


@pytest.fixture()
def tiny_project(tmp_path: Path) -> TinyProject:
    root = tmp_path
    train_dir = root / "data/raw/teacher/train"
    prediction_dir = root / "data/raw/teacher/test"
    train_images = train_dir / "images_train"
    prediction_images = prediction_dir / "images_test"
    train_images.mkdir(parents=True)
    prediction_images.mkdir(parents=True)

    rows = []
    article_types = {
        1: "A",
        2: "A",
        3: "A",
        4: "A",
        5: "A",
        6: "A",
        7: "A",
        8: "A",
        9: "C",
        10: "A",
        11: "D",
        12: "A",
    }
    for item_id, article_type in article_types.items():
        rows.append(
            {
                "id": item_id,
                "gender": "Unisex",
                "masterCategory": "Apparel",
                "subCategory": "Topwear",
                "articleType": article_type,
                "baseColour": "Blue",
                "season": "Winter" if item_id == 2 else "Summer",
                "year": 2026,
                "usage": None if item_id == 8 else "NA" if item_id == 9 else "Casual",
                "productDisplayName": f"Item {item_id}",
                "Unnamed: 10": "with comma" if item_id == 3 else None,
                "Unnamed: 11": None,
            }
        )
    train_csv = train_dir / "styles_train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    colors = {
        1: (10, 20, 30),
        2: (10, 20, 30),
        3: (30, 50, 70),
        4: (50, 80, 110),
        5: (70, 110, 150),
        6: (90, 140, 190),
        7: (110, 170, 220),
        8: (130, 190, 80),
        9: (150, 70, 100),
        10: (170, 90, 120),
        12: (190, 120, 60),
    }
    for item_id, color in colors.items():
        _save_image(train_images / f"{item_id}.jpg", color, mode="L" if item_id == 9 else "RGB")
    (train_images / ".DS_Store").write_text("metadata", encoding="utf-8")

    prediction_csv = prediction_dir / "styles_prediction.csv"
    pd.DataFrame(
        {
            "id": [101, 102],
            "gender": [None, None],
            "articleType": [None, None],
            "season": [None, None],
            "usage": [None, None],
        }
    ).to_csv(prediction_csv, index=False)
    _save_image(prediction_images / "101.jpg", colors[10])
    _save_image(prediction_images / "102.jpg", (210, 180, 140))

    processed = root / "data/processed"
    audit = processed / "audit"
    return TinyProject(
        root=root,
        train_csv=train_csv,
        prediction_csv=prediction_csv,
        train_images=train_images,
        prediction_images=prediction_images,
        processed=processed,
        audit=audit,
        prediction_manifest=processed / "prediction_manifest.csv",
        label_maps=processed / "label_maps.json",
        splits=processed / "splits.csv",
        split_summary=processed / "split_summary.json",
        development_summary=processed / "development_class_summary.csv",
        normalization=processed / "normalization_original_only.json",
        taxonomy=processed / "taxonomy.json",
        paired_normalization=processed / "paired_normalization.json",
    )


@pytest.fixture()
def prepared_project(tiny_project: TinyProject) -> TinyProject:
    project = tiny_project
    # Most small data-contract tests do not need the separate high-resolution fixture.
    # The official notebook test builds that fixture and exercises the doubled policy.
    prepare_data(root=project.root, workers=2, include_high_resolution_variants=False)
    return project
