"""Check auxiliary intake boundaries independently of teacher data."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.data.external_usage import (
    ExternalUsageDataset,
    assign_external_partitions,
    decoded_sha256,
    file_sha256,
    fit_external_statistics,
    load_external_manifest,
    prepare_external_image,
    validate_external_manifest,
)


@pytest.fixture
def accepted(tmp_path):
    original = tmp_path / "original.png"
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = [32, 64, 96]
    pixels[:, 20:] = [160, 192, 224]
    Image.fromarray(pixels).save(original)
    prepared = prepare_external_image(original, tmp_path / "prepared.png")
    return pd.DataFrame(
        [
            {
                **prepared,
                "external_id": "shop:item:view1",
                "source": "shop",
                "source_id": "item",
                "source_label": "Home",
                "label_strength": "source_exact_occasion",
                "label_evidence": "Occasion: Home",
                "source_url": "https://example.org/item",
                "image_url": "https://example.org/image.png",
                "rights_basis": "fixture only",
                "external_group_id": "shop:item",
                "external_partition": "train",
                "path": "prepared.png",
                "original_sha256": file_sha256(original),
                "teacher_overlap": False,
                "teacher_usage_compatible": False,
                "visual_decision": "keep",
                "training_target": "source_usage",
            }
        ]
    )


def test_preparation_geometry_and_encoding_independent_hash(tmp_path, accepted):
    row = accepted.iloc[0]
    assert (row.content_left, row.content_top, row.content_width, row.content_height) == (
        0,
        25,
        60,
        30,
    )
    with Image.open(tmp_path / row.path) as image:
        assert image.mode == "RGB"
        assert image.size == (60, 80)
        assert image.format == "PNG"
        assert np.all(np.asarray(image)[:25] == 255)
        image.save(tmp_path / "other.bmp")
        expected = decoded_sha256(image)
    with Image.open(tmp_path / "other.bmp") as other:
        assert decoded_sha256(other) == expected
    assert file_sha256(tmp_path / "other.bmp") != row.sha256


def test_group_split_is_stable_and_keeps_views_together():
    rows = [
        {"external_group_id": f"g{i}", "source_label": "Home", "source": "shop"}
        for i in range(10)
        for _ in range(i % 3 + 1)
    ]
    rows.append({"external_group_id": "single", "source_label": "Party", "source": "shop"})
    frame = pd.DataFrame(rows)
    first = assign_external_partitions(frame)
    second = assign_external_partitions(frame.sample(frac=1, random_state=71))
    first_groups = first.groupby("external_group_id").external_partition
    second_groups = second.groupby("external_group_id").external_partition
    pd.testing.assert_series_equal(first_groups.first(), second_groups.first())
    assert first_groups.nunique().eq(1).all()
    assert first_groups.first().value_counts().to_dict() == {"train": 9, "validation": 2}
    assert first_groups.first()["single"] == "train"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("source_label", "NA", "NA definition"),
        ("source_label", "Sports", "rare source classes"),
        ("source_label", "", "blank external field"),
        ("label_strength", "weak_title", "weak text"),
        ("teacher_overlap", True, "teacher overlap"),
        ("teacher_usage_compatible", True, "source-label supervision"),
        ("training_target", "gender", "source-label supervision"),
        ("visual_decision", "reject", "visual keep"),
        ("path", "../outside.png", "escapes"),
        ("content_width", 61, "content rectangle"),
    ],
)
def test_invalid_admission_fails_closed(accepted, tmp_path, column, value, message):
    accepted[column] = value
    with pytest.raises(ValueError, match=message):
        validate_external_manifest(accepted, root=tmp_path)


@pytest.mark.parametrize(
    "shared_key", ["external_group_id", "sha256", "decoded_sha256", "original_decoded_sha256"]
)
def test_duplicate_identity_cannot_cross_partitions(accepted, tmp_path, shared_key):
    second = accepted.copy()
    second["external_id"] = "other"
    second["external_partition"] = "validation"
    for column in ("external_group_id", "sha256", "decoded_sha256", "original_decoded_sha256"):
        if column != shared_key:
            second[column] = "different"
    with pytest.raises(ValueError, match=f"{shared_key} crosses"):
        validate_external_manifest(pd.concat([accepted, second]), root=tmp_path)


def test_statistics_use_training_content_not_padding(accepted, tmp_path):
    stats = fit_external_statistics(accepted, root=tmp_path)
    with Image.open(tmp_path / "prepared.png") as image:
        content = np.asarray(image, dtype=np.float64)[25:55] / 255
    np.testing.assert_allclose(stats["mean"], content.mean(axis=(0, 1)), atol=1e-7)
    assert stats["content_pixels"] == 1800
    accepted["external_partition"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        fit_external_statistics(accepted, root=tmp_path)


def test_adapter_and_load_preserve_auxiliary_scope(accepted, tmp_path):
    manifest = tmp_path / "manifest.csv"
    accepted.to_csv(manifest, index=False)
    loaded = load_external_manifest(manifest, root=tmp_path)
    options = dict(
        partition="train", label_to_index={"Home": 2}, mean=[0.5] * 3, std=[0.2] * 3, root=tmp_path
    )
    sample = ExternalUsageDataset(loaded, **options)[0]
    assert sample["image"].shape == (3, 80, 60)
    assert sample["image"].dtype == np.float32
    assert np.isfinite(sample["image"]).all()
    assert np.all(sample["image"][:, :25] == 0)
    assert sample["label"] == 2
    assert sample["sample_weight"] == 1.0
    assert sample["external_id"] == "shop:item:view1"
    assert sample["training_target"] == "source_usage"
    assert not {"id", "cv_fold", "gender", "season", "articleType", "usage"}.intersection(sample)
    with pytest.raises(ValueError, match="only supports"):
        ExternalUsageDataset(loaded, **options, target="usage")
    accepted["source_label"] = "NA"
    accepted.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="NA definition"):
        load_external_manifest(manifest, root=tmp_path)


def test_file_tampering_is_detected(accepted, tmp_path):
    Image.new("RGB", (60, 80), "red").save(tmp_path / "prepared.png")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_external_manifest(accepted, root=tmp_path, check_files=True)


def test_repeated_views_share_one_unit_of_training_weight(accepted, tmp_path):
    second = accepted.copy()
    second["external_id"] = "shop:item:view2"
    dataset = ExternalUsageDataset(
        pd.concat([accepted, second]),
        partition="train",
        label_to_index={"Home": 0},
        mean=[0.5] * 3,
        std=[0.2] * 3,
        root=tmp_path,
    )
    assert len(dataset) == 2
    assert [dataset[i]["sample_weight"] for i in range(2)] == [0.5, 0.5]


def test_conflicting_family_labels_cannot_be_split():
    frame = pd.DataFrame(
        {
            "external_group_id": ["same", "same"],
            "source_label": ["Home", "Party"],
            "source": ["shop", "shop"],
        }
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        assign_external_partitions(frame)


@pytest.mark.parametrize("coordinate", [0.9, float("nan"), float("inf")])
def test_content_rectangle_rejects_noninteger_coordinates(accepted, tmp_path, coordinate):
    accepted["content_left"] = coordinate
    with pytest.raises(ValueError, match="blank external field|finite integers"):
        validate_external_manifest(accepted, root=tmp_path)


@pytest.mark.parametrize("mapping", [{"Home": -1}, {"Home": 0.5}, {"Home": 0, "Party": 0}])
def test_adapter_rejects_invalid_label_indices(accepted, tmp_path, mapping):
    with pytest.raises(ValueError, match="unique nonnegative integers"):
        ExternalUsageDataset(
            accepted,
            partition="train",
            label_to_index=mapping,
            mean=[0.5] * 3,
            std=[0.2] * 3,
            root=tmp_path,
        )
