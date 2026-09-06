"""Direct Usage training and unchanged teacher boundaries in the combined dataset."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.config import TARGET_COLUMNS
from fashion.data import FashionDataset, get_cv_split, get_samples, load_splits
from fashion.data.expanded_usage import (
    EXTERNAL_ID_START,
    assign_added_folds,
    combine_teacher_and_rare_usage,
    validate_combined_usage,
)
from fashion.data.external_usage import file_sha256, prepare_external_image
from fashion.data.images import load_and_transform_image


@pytest.fixture
def inputs(tmp_path):
    teachers = []
    for index in range(6):
        path = tmp_path / f"teacher_{index}.png"
        Image.new("RGB", (60, 80), (index * 20, 100, 200)).save(path)
        protected = index == 5
        row = {
            "id": str(index + 1),
            "path": path.name,
            "sha256": file_sha256(path),
            "duplicate_group": f"teacher_{index}",
            "product_name_key": f"teacher {index}",
            "product_family_group": f"teacher_{index}",
            "partition": "holdout" if protected else "development",
            "cv_fold": "" if protected else str(index),
            "is_cross_role_exact_duplicate": False,
            "is_cross_role_near_duplicate": False,
            "has_conflicting_target_labels": False,
            "conflicting_targets": "",
            "quarantine_reason": "",
        }
        for target in TARGET_COLUMNS:
            row[target] = "" if protected else ("NA" if index == 0 else "Casual")
            row[f"has_{target}_label"] = not protected
        teachers.append(row)
    sources = []
    for index in range(8):
        path = tmp_path / f"source_{index}.png"
        Image.new("RGB", (70, 40), (150, index * 20, 30)).save(path)
        prepared = tmp_path / f"prepared_{index}.png"
        sources.append(
            {
                **prepare_external_image(path, prepared),
                "external_id": f"shop:{index}",
                "source": "shop",
                "source_id": str(index),
                "source_title": f"Source product {index // 2}",
                "source_label": "Home",
                "label_strength": "source_exact_occasion",
                "label_evidence": "Occasion: Home",
                "source_url": f"https://example.org/{index}",
                "image_url": f"https://example.org/{index}.png",
                "rights_basis": "fixture only",
                "external_group_id": f"family_{index // 2}",
                "external_partition": "train",
                "path": prepared.name,
                "original_sha256": file_sha256(path),
                "teacher_overlap": False,
                "teacher_usage_compatible": False,
                "visual_decision": "keep",
                "training_target": "source_usage",
            }
        )
    classes = [
        "Casual",
        "Ethnic",
        "Formal",
        "Home",
        "NA",
        "Party",
        "Smart Casual",
        "Sports",
        "Travel",
    ]
    mapping = {
        "usage": {
            "label_column": "usage",
            "classes": classes,
            "label_to_index": {value: index for index, value in enumerate(classes)},
        }
    }
    return pd.DataFrame(teachers), pd.DataFrame(sources), mapping


def test_combined_rows_use_normal_loader_and_same_nine_class_target(inputs, tmp_path):
    teacher, source, maps = inputs
    combined = combine_teacher_and_rare_usage(teacher, source, maps, root=tmp_path)
    path = tmp_path / "splits.csv"
    combined.to_csv(path, index=False)
    loaded = load_splits(path)
    assert len(loaded) == 14
    assert loaded.loc[loaded.id.eq(1), "usage"].item() == "NA"
    assert loaded.loc[loaded.id.eq(6), "usage"].item() == ""
    added = loaded.loc[loaded.source_dataset.ne("teacher")]
    assert added.usage.eq("Home").all() and added.has_usage_label.all()
    assert not added.has_gender_label.any()
    assert not added.has_season_label.any()
    assert not added.has_articleType_label.any()
    dataset = FashionDataset(
        get_samples(loaded, partition="development", target="usage"),
        transform=lambda p: load_and_transform_image(p, image_size=(80, 60)),
        root=tmp_path,
        targets=("usage",),
    )
    targets = [maps["usage"]["label_to_index"][dataset[i]["usage"]] for i in range(len(dataset))]
    assert targets.count(3) == 8
    assert dataset[0]["image"].shape == (80, 60, 3)
    assert np.isfinite(dataset[0]["image"]).all()
    assert "source_usage" not in loaded.columns


def test_families_never_cross_combined_train_validation(inputs, tmp_path):
    teacher, source, maps = inputs
    combined = combine_teacher_and_rare_usage(teacher, source, maps, root=tmp_path)
    for fold in range(5):
        train, validation = get_cv_split(combined, fold)
        assert not set(train.product_family_group).intersection(validation.product_family_group)
        assert "6" not in set(train.id) | set(validation.id)
    pd.testing.assert_frame_equal(
        combined.iloc[: len(teacher)][teacher.columns], teacher, check_dtype=False
    )
    shuffled = combine_teacher_and_rare_usage(
        teacher, source.sample(frac=1, random_state=42), maps, root=tmp_path
    )
    pd.testing.assert_frame_equal(combined, shuffled)


@pytest.mark.parametrize(
    "change", ["teacher_fold", "source_label", "gender_label", "missing_source"]
)
def test_changed_boundaries_or_targets_fail_validation(inputs, tmp_path, change):
    teacher, source, maps = inputs
    combined = combine_teacher_and_rare_usage(teacher, source, maps, root=tmp_path)
    if change == "teacher_fold":
        combined.loc[combined.id.eq("1"), "cv_fold"] = "1"
    elif change == "source_label":
        combined.loc[combined.source_dataset.ne("teacher"), "usage"] = "Party"
    elif change == "gender_label":
        combined.loc[combined.source_dataset.ne("teacher"), "gender"] = "Women"
    else:
        combined = combined.iloc[:-1]
    with pytest.raises((ValueError, AssertionError)):
        validate_combined_usage(combined, teacher, source)


@pytest.mark.parametrize("change", ["teacher_id", "teacher_image", "teacher_name", "unknown_label"])
def test_invalid_merge_inputs_are_rejected(inputs, tmp_path, change):
    teacher, source, maps = inputs
    if change == "teacher_id":
        teacher.loc[0, "id"] = str(EXTERNAL_ID_START)
    elif change == "teacher_image":
        teacher.loc[0, "sha256"] = source.loc[0, "sha256"]
    elif change == "teacher_name":
        teacher.loc[0, "product_name_key"] = "source product 0"
    else:
        maps["usage"]["classes"].remove("Home")
    with pytest.raises(ValueError):
        combine_teacher_and_rare_usage(teacher, source, maps, root=tmp_path)


def test_large_and_small_source_families_remain_whole(inputs):
    _, source, _ = inputs
    first = assign_added_folds(source)
    shuffled = source.sample(frac=1, random_state=7)
    second = assign_added_folds(shuffled)
    assert first.to_dict() == second.to_dict()
    assert source.assign(fold=first).groupby("external_group_id").fold.nunique().eq(1).all()
