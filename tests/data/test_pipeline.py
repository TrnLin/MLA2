from __future__ import annotations

import json

import pandas as pd
import pytest

from fashion.data.dataset import get_samples, load_label_maps, load_manifest, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.splits import find_exact_duplicate_label_conflicts, validate_splits
from fashion.data.statistics import train_ids_digest


def test_audit_preserves_raw_csv_digest(prepared_project):
    before = json.loads((prepared_project.audit / "csv_summary.json").read_text())
    assert compute_sha256(prepared_project.train_csv) == before["styles_train"]["sha256"]
    assert compute_sha256(prepared_project.prediction_csv) == before["styles_prediction"]["sha256"]


def test_prepare_data_scopes_every_output_to_supplied_root(prepared_project):
    expected = (
        prepared_project.audit / "csv_summary.json",
        prepared_project.train_manifest,
        prepared_project.prediction_manifest,
        prepared_project.label_maps,
        prepared_project.splits,
        prepared_project.split_summary,
        prepared_project.development_summary,
        prepared_project.normalization,
    )
    assert all(path.exists() for path in expected)


def test_audit_separates_non_image_entries(prepared_project):
    audit = pd.read_csv(prepared_project.audit / "image_audit.csv")
    non_images = audit[audit["error"].eq("non_image_extension")]
    assert non_images["filename"].tolist() == [".DS_Store"]
    assert audit.loc[audit["filename"].str.endswith(".jpg"), "decode_ok"].all()


def test_manifest_excludes_missing_image_and_repairs_spill(prepared_project):
    manifest = pd.read_csv(prepared_project.train_manifest)
    assert 11 not in set(manifest["id"])
    row = manifest.set_index("id").loc[3]
    assert row["productDisplayName"] == "Item 3, with comma"
    assert bool(row["product_name_repaired"])


def test_prediction_manifest_contains_no_target_columns(prepared_project):
    prediction = pd.read_csv(prepared_project.prediction_manifest)
    assert prediction["id"].tolist() == [101, 102]
    assert not {"gender", "articleType", "season", "usage"}.intersection(prediction.columns)


def test_label_maps_cover_image_backed_labels(prepared_project):
    maps = load_label_maps(prepared_project.label_maps)
    assert maps["articleType"]["classes"] == ["A", "B", "C"]
    assert "D" not in maps["articleType"]["classes"]


def test_literal_na_survives_csv_to_manifest_split_and_label_map(prepared_project):
    raw = pd.read_csv(prepared_project.train_csv, keep_default_na=False).set_index("id")
    manifest = load_manifest(prepared_project.train_manifest).set_index("id")
    splits = load_manifest(prepared_project.splits).set_index("id")
    maps = load_label_maps(prepared_project.label_maps)

    assert raw.loc[9, "usage"] == "NA"
    assert manifest.loc[9, "usage"] == "NA"
    assert bool(manifest.loc[9, "has_usage_label"])
    assert splits.loc[9, "usage"] == "NA"
    assert bool(splits.loc[9, "has_usage_label"])
    assert "NA" in maps["usage"]["classes"]

    summary = json.loads((prepared_project.audit / "csv_summary.json").read_text())
    assert summary["styles_train"]["missing_counts"]["usage"] == 1


def test_splits_are_complete_and_valid(prepared_project):
    splits = load_splits(prepared_project.splits)
    validate_splits(splits)
    assert set(splits["partition"]) == {"train", "val", "holdout", "quarantine"}
    assert splits["id"].nunique() == len(splits)


def test_downstream_split_loader_rejects_crossed_exact_duplicate_group(prepared_project, tmp_path):
    splits = load_splits(prepared_project.splits)
    splits.loc[splits["id"].eq(1), "partition"] = "train"
    invalid_path = tmp_path / "invalid-splits.csv"
    splits.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="exact SHA-256 duplicate group crosses partitions"):
        load_manifest(invalid_path)
    with pytest.raises(ValueError, match="exact SHA-256 duplicate group crosses partitions"):
        load_splits(invalid_path)


def test_cross_role_exact_duplicate_is_quarantined(prepared_project):
    splits = load_manifest(prepared_project.splits).set_index("id")
    assert splits.loc[10, "partition"] == "quarantine"
    assert bool(splits.loc[10, "is_cross_role_duplicate"])


def test_internal_exact_duplicates_share_partition(prepared_project):
    splits = load_manifest(prepared_project.splits).set_index("id")
    assert splits.loc[1, "sha256"] == splits.loc[2, "sha256"]
    assert splits.loc[1, "partition"] == splits.loc[2, "partition"]


def test_conflicting_exact_duplicate_labels_are_quarantined(prepared_project):
    splits = load_manifest(prepared_project.splits)
    conflict = splits[splits["id"].isin([1, 2])].set_index("id")

    assert conflict["partition"].eq("quarantine").all()
    assert conflict["has_conflicting_target_labels"].all()
    assert conflict["conflicting_targets"].eq("season").all()
    assert find_exact_duplicate_label_conflicts(splits) == {conflict.loc[1, "sha256"]: ("season",)}


def test_split_validation_rejects_conflicting_duplicate_outside_quarantine(prepared_project):
    splits = load_manifest(prepared_project.splits)
    splits.loc[splits["id"].isin([1, 2]), "partition"] = "train"
    with pytest.raises(ValueError, match="conflicting labels exists outside quarantine"):
        validate_splits(splits)


def test_missing_and_valid_duplicate_labels_are_masked_not_conflicting():
    frame = pd.DataFrame(
        {
            "sha256": ["same", "same"],
            "articleType": ["A", "A"],
            "season": ["Summer", "Summer"],
            "gender": ["Unisex", "Unisex"],
            "usage": ["", "Casual"],
            "has_articleType_label": [True, True],
            "has_season_label": [True, True],
            "has_gender_label": [True, True],
            "has_usage_label": [False, True],
        }
    )
    assert find_exact_duplicate_label_conflicts(frame) == {}


def test_split_summary_does_not_aggregate_protected_targets(prepared_project):
    summary = json.loads(prepared_project.split_summary.read_text())
    article = summary["development_target_distributions"]["articleType"]
    assert set(article) == {"train", "val", "protected_partitions"}
    assert "valid_count" not in article["protected_partitions"]
    conflicts = summary["conflicting_label_exact_duplicates"]
    assert conflicts["hash_groups"] == 1
    assert conflicts["quarantined_rows"] == 2
    assert conflicts["groups_by_target"]["season"] == 1


def test_normalization_provenance_is_train_only(prepared_project):
    splits = load_manifest(prepared_project.splits)
    stats = json.loads(prepared_project.normalization.read_text())
    train_ids = splits.loc[splits["partition"].eq("train"), "id"].tolist()
    assert stats["source_partition"] == "train"
    assert stats["num_images"] == len(train_ids)
    assert stats["train_ids_digest"] == train_ids_digest(train_ids)
    assert all(0 < value < 1 for value in stats["mean"])
    assert all(0 < value < 1 for value in stats["std"])


def test_get_samples_uses_existing_partition_and_masks(prepared_project):
    splits = load_manifest(prepared_project.splits)
    train_usage = get_samples(splits, partition="train", target="usage")
    assert train_usage["partition"].eq("train").all()
    assert train_usage["has_usage_label"].all()


def test_official_prediction_ids_never_enter_labelled_splits(prepared_project):
    splits = pd.read_csv(prepared_project.splits)
    prediction = pd.read_csv(prepared_project.prediction_manifest)
    assert set(splits["id"]).isdisjoint(prediction["id"])
