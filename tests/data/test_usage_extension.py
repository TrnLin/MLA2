"""New Usage images preserve old folds and keep related products together."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.config import TARGET_COLUMNS
from fashion.data import get_cv_split, get_samples, load_splits
from fashion.data.external_usage import file_sha256, prepare_external_image
from fashion.data.usage_extension import (
    USAGE_CLASSES,
    extend_usage_dataset,
    validate_usage_extension,
)


@pytest.fixture
def extension_inputs(tmp_path):
    previous = []
    for index in range(7):
        protected, external = index == 5, index == 6
        row = {
            "id": str(1_000_000_000 if external else index + 1),
            "path": f"old_{index}.png",
            "sha256": f"old_hash_{index}",
            "product_name_key": f"old name {index}",
            "product_family_group": f"old_family_{index}",
            "duplicate_group": f"old_duplicate_{index}",
            "partition": "holdout" if protected else "development",
            "cv_fold": "" if protected else str(index % 5),
            "is_cross_role_exact_duplicate": False,
            "is_cross_role_near_duplicate": False,
            "is_cross_role_duplicate": False,
            "has_conflicting_target_labels": False,
            "conflicting_targets": "",
            "quarantine_reason": "",
            "pre_quarantine_reason": "",
            "source_dataset": "old_shop" if external else "teacher",
            "source_id": "previous" if external else str(index + 1),
            "external_id": "old_shop:previous" if external else "",
            "source_group_id": "previous_group" if external else "",
            "source_label": "Home" if external else "",
            "label_mapping_basis": "old_policy",
            "productDisplayName": f"Old name {index}",
            "product_name_repaired": False,
            "width": 60,
            "height": 80,
            "aspect_ratio": 0.75,
            "mode": "RGB",
            "format": "PNG",
            "file_size_bytes": 0,
            "family_group_basis": "fixture",
            "source_url": "",
            "image_url": "",
            "label_evidence": "",
            "label_strength": "",
            "rights_basis": "fixture",
            "original_sha256": f"old_original_{index}",
        }
        for target in TARGET_COLUMNS:
            valid = not protected and (not external or target == "usage")
            row[target] = (
                ("Home" if external else "NA" if index == 0 else "Casual") if valid else ""
            )
            row[f"has_{target}_label"] = valid
        previous.append(row)
    new = []
    for index in range(12):
        original = tmp_path / f"source_{index}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (70, 45, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(original)
        path = tmp_path / f"new_{index}.png"
        new.append(
            {
                **prepare_external_image(original, path),
                "external_id": f"new_shop:{index:02d}",
                "source_dataset": "new_shop",
                "source_record_id": str(index),
                "product_name": f"Decorative cushion design {index // 2}",
                "description": "Home cushion cover",
                "usage": "Home",
                "proposed_usage": "Home",
                "evidence_text": "Decorative cushion cover for sofa",
                "evidence_basis": "product_text_inference",
                "product_url": f"https://example.org/item/{index}",
                "image_url": f"https://example.org/item/{index}.png",
                "path": path.name,
                "source_file_sha256": file_sha256(original),
                "source_original_path": original.name,
                "external_group_id": f"new_family_{index // 2}",
                "partition": "unassigned",
                "label_status": "proposed_from_source_text_or_collection",
                "rejection_reason": "",
                "rights_basis": "fixture",
                "rights_status": "fixture",
                "attribution": "fixture",
                "existing_image_overlap": False,
                "previously_admitted_identity": False,
            }
        )
    previous, new = pd.DataFrame(previous), pd.DataFrame(new)
    links = pd.concat(
        [
            pd.DataFrame([{"external_id": "old_shop:previous", "external_group_id": "old_link"}]),
            new[["external_id", "external_group_id"]],
        ],
        ignore_index=True,
    )
    maps = {
        "usage": {
            "label_column": "usage",
            "classes": list(USAGE_CLASSES),
            "label_to_index": {label: index for index, label in enumerate(USAGE_CLASSES)},
        }
    }
    return previous, new, maps, links


def test_preserves_old_rows_and_normal_loader_boundaries(extension_inputs, tmp_path):
    previous, new, maps, links = extension_inputs
    combined = extend_usage_dataset(previous, new, maps, links, root=tmp_path)
    pd.testing.assert_frame_equal(
        combined.iloc[: len(previous)][previous.columns], previous, check_dtype=False
    )
    assert len(combined) == 19
    path = tmp_path / "splits.csv"
    combined.to_csv(path, index=False)
    loaded = load_splits(path)
    assert loaded.loc[loaded.id.eq(1), "usage"].item() == "NA"
    assert loaded.loc[loaded.id.eq(6), "usage"].item() == ""
    for fold in range(5):
        train, validation = (get_samples(f, target="usage") for f in get_cv_split(loaded, fold))
        assert not set(train.id) & set(validation.id)
        assert not set(train.product_family_group) & set(validation.product_family_group)
        assert not (set(train.extension_family_group) - {""}) & set(
            validation.extension_family_group
        )
    added = loaded.loc[loaded.source_dataset.eq("new_shop")]
    assert added.id.min() == 1_000_000_001
    assert added.has_usage_label.all() and not added.has_gender_label.any()
    assert added.label_strength.eq("product_text_inference").all()
    shuffled = extend_usage_dataset(
        previous,
        new.sample(frac=1, random_state=4),
        maps,
        links.sample(frac=1, random_state=8),
        root=tmp_path,
    )
    pd.testing.assert_frame_equal(combined, shuffled)


def test_new_relatives_follow_old_frozen_fold(extension_inputs, tmp_path):
    previous, new, maps, links = extension_inputs
    links.loc[links.external_group_id.eq("new_family_0"), "external_group_id"] = "old_link"
    combined = extend_usage_dataset(previous, new, maps, links, root=tmp_path)
    family = combined.loc[combined.extension_family_group.eq("old_link")]
    assert len(family) == 3 and family.cv_fold.astype(int).eq(1).all()


@pytest.mark.parametrize(
    "change", ["na", "evidence", "overlap", "label", "geometry", "sha", "pixels", "family"]
)
def test_rejects_changed_or_unsafe_source_inputs(extension_inputs, tmp_path, change):
    previous, new, maps, links = extension_inputs
    if change == "na":
        new.loc[0, ["usage", "proposed_usage"]] = "NA"
    elif change == "evidence":
        new.loc[0, "evidence_text"] = ""
    elif change == "overlap":
        new.loc[0, "existing_image_overlap"] = True
    elif change == "label":
        new.loc[0, "usage"] = "Party"
    elif change == "geometry":
        new.loc[0, "content_width"] = 90
    elif change == "sha":
        new.loc[0, "sha256"] = "changed"
    elif change == "pixels":
        Image.new("RGB", (60, 80), "black").save(tmp_path / new.loc[0, "path"])
        new.loc[0, "sha256"] = file_sha256(tmp_path / new.loc[0, "path"])
    else:
        links.loc[links.external_id.eq("new_shop:00"), "external_group_id"] = "broken"
    with pytest.raises(ValueError):
        extend_usage_dataset(previous, new, maps, links, root=tmp_path)


def test_frozen_fold_edits_and_conflicting_anchors_fail(extension_inputs, tmp_path):
    previous, new, maps, links = extension_inputs
    combined = extend_usage_dataset(previous, new, maps, links, root=tmp_path)
    combined.loc[combined.id.eq("1000000000"), "cv_fold"] = "2"
    with pytest.raises((AssertionError, ValueError)):
        validate_usage_extension(combined, previous, new, links)
    second = previous.iloc[[-1]].copy()
    for column, value in {
        "id": "1000000001",
        "external_id": "old_shop:second",
        "product_family_group": "other_family",
        "source_group_id": "other_group",
        "product_name_key": "different name",
        "sha256": "other_hash",
        "duplicate_group": "other_duplicate",
        "cv_fold": "3",
    }.items():
        second[column] = value
    previous = pd.concat([previous, second], ignore_index=True)
    links = pd.concat(
        [
            links,
            pd.DataFrame([{"external_id": "old_shop:second", "external_group_id": "old_link"}]),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="previously frozen folds"):
        extend_usage_dataset(previous, new, maps, links, root=tmp_path)
