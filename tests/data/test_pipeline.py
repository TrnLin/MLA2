from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from fashion.data.dataset import (
    get_samples,
    load_label_maps,
    load_manifest,
    load_splits,
    load_splits_for_final_evaluation,
)
from fashion.data.hashing import compute_sha256, csv_header_and_id_fingerprint
from fashion.data.manifests import write_structural_train_manifest
from fashion.data.metadata import build_label_maps
from fashion.data.pipeline import (
    _tree_fingerprint,
    prepare_data,
    refresh_protected_safe_tabular_artifacts,
    validate_prepared_data_cache,
)
from fashion.data.splits import find_exact_duplicate_label_conflicts, validate_splits
from fashion.data.statistics import train_ids_digest
from fashion.data.taxonomy import apply_deployment_taxonomy


def test_audit_hashes_only_csv_headers_and_ids(prepared_project):
    summary = json.loads((prepared_project.audit / "csv_summary.json").read_text())
    assert summary["protected_target_values_hashed"] == 0
    for name, path in (
        ("styles_train", prepared_project.train_csv),
        ("styles_prediction", prepared_project.prediction_csv),
    ):
        expected = csv_header_and_id_fingerprint(path)
        assert summary[name]["header_and_id_sha256"] == expected["header_and_id_sha256"]
        assert "sha256" not in summary[name]


def test_prepare_data_scopes_every_output_to_supplied_root(prepared_project):
    expected = (
        prepared_project.audit / "csv_summary.json",
        prepared_project.prediction_manifest,
        prepared_project.label_maps,
        prepared_project.splits,
        prepared_project.split_summary,
        prepared_project.development_summary,
        prepared_project.normalization,
        prepared_project.taxonomy,
        prepared_project.processed / "preparation_cache.json",
    )
    assert all(path.exists() for path in expected)


def test_audit_separates_non_image_entries(prepared_project):
    audit = pd.read_csv(prepared_project.audit / "image_audit.csv.gz")
    non_images = audit[audit["error"].eq("non_image_extension")]
    assert non_images["filename"].tolist() == [".DS_Store"]
    assert audit.loc[audit["filename"].str.endswith(".jpg"), "decode_ok"].all()


def test_manifest_excludes_missing_image_and_repairs_spill(prepared_project):
    assert not (prepared_project.processed / "train_manifest.csv").exists()
    assert not (prepared_project.processed / "train_manifest.csv.gz").exists()
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    assert 11 not in set(splits["id"])
    row = splits.set_index("id").loc[3]
    assert row["productDisplayName"] == "Item 3, with comma"
    assert bool(row["product_name_repaired"])


def test_prediction_manifest_contains_no_target_columns(prepared_project):
    prediction = pd.read_csv(prepared_project.prediction_manifest)
    assert prediction["id"].tolist() == [101, 102]
    assert not {"gender", "articleType", "season", "usage"}.intersection(prediction.columns)


def test_label_maps_cover_every_trainable_official_output_label(prepared_project):
    maps = load_label_maps(prepared_project.label_maps)
    assert maps["articleType"]["classes"] == ["A", "C"]
    assert maps["articleType"]["source_partition"] == "train"
    assert "D" not in maps["articleType"]["classes"]


def test_literal_na_is_an_official_output_but_not_a_supported_primary_metric(prepared_project):
    raw = pd.read_csv(prepared_project.train_csv, keep_default_na=False).set_index("id")
    splits = load_manifest(prepared_project.splits).set_index("id")
    maps = load_label_maps(prepared_project.label_maps)

    assert raw.loc[9, "usage"] == "NA"
    assert splits.loc[9, "usage"] == "NA"
    assert bool(splits.loc[9, "has_usage_label"])
    assert splits.loc[9, "usage_deployed"] == "NA"
    assert bool(splits.loc[9, "has_usage_deployed_label"])
    assert splits.loc[9, "usage_deployment_status"] == "official_output"
    assert splits.loc[9, "usage_supported"] == ""
    assert not bool(splits.loc[9, "has_usage_supported_label"])
    assert splits.loc[9, "usage_evaluation_status"] == (
        "official_but_insufficient_for_primary_metrics"
    )
    assert "NA" in maps["usage"]["classes"]

    summary = json.loads((prepared_project.audit / "csv_summary.json").read_text())
    usage_audit = summary["development_target_audit"]["targets"]["usage"]
    expected_missing = int(
        pd.read_csv(prepared_project.splits, keep_default_na=False)
        .query("partition in ['train', 'val']")["has_usage_label"]
        .astype(str)
        .str.lower()
        .eq("false")
        .sum()
    )
    assert sum(row["missing_count"] for row in usage_audit.values()) == expected_missing


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
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    conflict = splits[splits["id"].isin([1, 2])].set_index("id")

    assert conflict["partition"].eq("quarantine").all()
    assert conflict["has_conflicting_target_labels"].all()
    assert conflict["conflicting_targets"].eq("season").all()
    assert conflict["sha256"].nunique() == 1
    assert find_exact_duplicate_label_conflicts(splits) == {}


def test_normal_loading_and_diagnostics_ignore_protected_target_values(prepared_project, tmp_path):
    source = pd.read_csv(prepared_project.splits, keep_default_na=False)
    changed = source.copy()
    protected = changed["partition"].isin({"holdout", "quarantine"})
    for target in ("articleType", "season", "gender", "usage"):
        for column in (target, f"{target}_deployed", f"{target}_supported"):
            changed.loc[protected, column] = f"SENTINEL_{target}"
        for column in (
            f"has_{target}_label",
            f"has_{target}_deployed_label",
            f"has_{target}_supported_label",
        ):
            changed.loc[protected, column] = True
    changed_path = tmp_path / "changed-protected-targets.csv"
    changed.to_csv(changed_path, index=False)

    original_loaded = load_splits(prepared_project.splits)
    changed_loaded = load_splits(changed_path)
    pd.testing.assert_frame_equal(original_loaded, changed_loaded)

    with pytest.raises(ValueError, match="final evaluation is unlocked"):
        load_splits_for_final_evaluation(changed_path)
    unsealed = load_splits_for_final_evaluation(
        changed_path,
        evaluation_unlocked=True,
        raw_teacher_csv=prepared_project.train_csv,
        taxonomy_json=prepared_project.taxonomy,
    )
    raw = pd.read_csv(prepared_project.train_csv, keep_default_na=False).set_index("id")
    expected = unsealed.loc[protected, "id"].map(raw["articleType"])
    assert (
        unsealed.loc[protected, "articleType"]
        .reset_index(drop=True)
        .equals(expected.reset_index(drop=True))
    )
    assert not unsealed.loc[protected, "articleType"].str.startswith("SENTINEL_").any()


def test_persisted_split_redacts_every_protected_target(prepared_project):
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    protected = splits["partition"].isin({"holdout", "quarantine"})
    for target in ("articleType", "season", "gender", "usage"):
        assert splits.loc[protected, target].eq("").all()
        assert not splits.loc[protected, f"has_{target}_label"].astype(bool).any()
        assert splits.loc[protected, f"{target}_deployed"].eq("").all()
        assert splits.loc[protected, f"{target}_supported"].eq("").all()


def test_public_manifest_loader_cannot_bypass_protected_target_redaction(prepared_project):
    assert "redact_protected_targets" not in inspect.signature(load_manifest).parameters
    with pytest.raises(TypeError, match="redact_protected_targets"):
        load_manifest(  # type: ignore[call-arg]
            prepared_project.splits,
            redact_protected_targets=False,
        )

    loaded = load_manifest(prepared_project.splits)
    protected = loaded["partition"].isin({"holdout", "quarantine"})
    for target in ("articleType", "season", "gender", "usage"):
        assert loaded.loc[protected, target].eq("").all()
        assert not loaded.loc[protected, f"has_{target}_label"].any()


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
    quarantine = summary["quarantine"]
    assert quarantine["conflicting_exact_sha_groups"] == 1
    assert quarantine["conflicting_exact_sha_rows"] == 2
    assert summary["product_family_policy"]["approved_for_model_comparison"] is True
    assert summary["product_family_policy"]["family_groups_crossing_model_partitions"] == 0
    assert "sealed_build_time_target_audit" not in summary
    assert summary["taxonomy_fit"] == {
        "partition": "train",
        "minimum_independent_family_groups": 3,
        "sensitivity_scope": "train_only",
        "holdout_target_coverage_audited": False,
    }
    sensitivity = summary["taxonomy_sensitivity"]
    assert sensitivity["fit_scope"] == "train_only"
    assert sensitivity["fit_partition"] == "train"
    assert sensitivity["support_unit"] == "distinct_product_family_group"
    assert summary["persisted_protected_targets"] == "redacted"


def test_supported_taxonomy_classes_have_three_train_families(prepared_project):
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    taxonomy = json.loads(prepared_project.taxonomy.read_text(encoding="utf-8"))
    train = splits[splits["partition"].eq("train")]
    assert taxonomy["fit_partition"] == "train"
    for target, policy in taxonomy["targets"].items():
        supported = set(policy["supported_classes"])
        support = (
            train[train[f"has_{target}_supported_label"]]
            .groupby(f"{target}_supported")["product_family_group"]
            .nunique()
        )
        assert set(support.index.astype(str)) == supported
        assert support.ge(3).all()


def test_fixed_holdout_targets_cannot_change_train_fitted_taxonomy(prepared_project):
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    changed = splits.copy()
    protected = changed["partition"].isin({"holdout", "quarantine"})
    for target in ("articleType", "season", "gender", "usage"):
        changed.loc[protected, target] = f"SENTINEL_{target}"
        changed.loc[protected, f"has_{target}_label"] = True

    _, original_policy = apply_deployment_taxonomy(splits)
    _, changed_policy = apply_deployment_taxonomy(changed)
    assert changed_policy == original_policy
    assert build_label_maps(changed, tuple(original_policy["targets"])) == build_label_maps(
        splits, tuple(original_policy["targets"])
    )


def test_regenerated_public_artifacts_ignore_protected_target_mutations(prepared_project, tmp_path):
    public_paths = (
        prepared_project.audit / "csv_summary.json",
        prepared_project.audit / "missing_values.csv",
        prepared_project.audit / "target_class_counts.csv",
        prepared_project.audit / "issues.csv",
        prepared_project.splits,
        prepared_project.split_summary,
        prepared_project.taxonomy,
        prepared_project.label_maps,
        prepared_project.development_summary,
        prepared_project.processed / "preparation_cache.json",
    )
    before = {
        path.relative_to(prepared_project.root): compute_sha256(path) for path in public_paths
    }

    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    protected = splits["partition"].isin({"holdout", "quarantine"})
    protected_ids = set(splits.loc[protected, "id"].astype(int))
    staging_path = tmp_path / "staging.csv.gz"
    write_structural_train_manifest(splits, staging_path)
    original_staging_digest = compute_sha256(staging_path)
    for target in ("articleType", "season", "gender", "usage"):
        splits.loc[protected, target] = f"SENTINEL_{target}"
        splits.loc[protected, f"has_{target}_label"] = True
    write_structural_train_manifest(splits, staging_path)
    assert original_staging_digest == compute_sha256(staging_path)
    splits.to_csv(prepared_project.splits, index=False)

    raw = pd.read_csv(prepared_project.train_csv, keep_default_na=False)
    raw_protected = raw["id"].astype(int).isin(protected_ids)
    for target in ("articleType", "season", "gender", "usage"):
        raw.loc[raw_protected, target] = f"SENTINEL_{target}"
    raw.to_csv(prepared_project.train_csv, index=False)

    refresh_protected_safe_tabular_artifacts(
        root=prepared_project.root,
        include_high_resolution_variants=False,
    )
    after = {path.relative_to(prepared_project.root): compute_sha256(path) for path in public_paths}
    assert after == before
    assert "SENTINEL_" not in "".join(path.read_text(encoding="utf-8") for path in public_paths)

    summary = json.loads(prepared_project.split_summary.read_text(encoding="utf-8"))
    sensitivity = summary["taxonomy_sensitivity"]
    assert sensitivity["fit_scope"] == "train_only"
    train = pd.read_csv(prepared_project.splits, keep_default_na=False).query(
        "partition == 'train'"
    )
    for target in ("articleType", "season", "gender", "usage"):
        support = (
            train.loc[train[f"has_{target}_label"], [target, "product_family_group"]]
            .drop_duplicates()
            .groupby(target)["product_family_group"]
            .nunique()
        )
        assert sensitivity["targets"][target]["3"]["kept_classes"] == int(support.ge(3).sum())


def test_split_validation_rejects_product_family_crossing(prepared_project):
    splits = load_splits(prepared_project.splits)
    active = splits[splits["partition"].ne("quarantine")]
    first, second = active.index[:2]
    splits.loc[second, "product_family_group"] = splits.loc[first, "product_family_group"]
    if splits.loc[first, "partition"] == splits.loc[second, "partition"]:
        replacement = active[active["partition"].ne(splits.loc[first, "partition"])].index[0]
        splits.loc[replacement, "product_family_group"] = splits.loc[first, "product_family_group"]

    with pytest.raises(ValueError, match="product family group crosses partitions"):
        validate_splits(splits)


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


def test_prepare_data_regenerates_deterministic_policy_artifacts(prepared_project):
    paths = (
        prepared_project.splits,
        prepared_project.split_summary,
        prepared_project.label_maps,
        prepared_project.normalization,
        prepared_project.taxonomy,
        prepared_project.audit / "near_duplicate_candidates.csv.gz",
        prepared_project.audit / "product_family_groups.csv.gz",
        prepared_project.processed / "preparation_cache.json",
    )
    before = {path.name: compute_sha256(path) for path in paths}
    prepare_data(
        root=prepared_project.root,
        workers=2,
        include_high_resolution_variants=False,
    )
    after = {path.name: compute_sha256(path) for path in paths}
    assert after == before


def test_prepared_data_cache_validates_without_rebuilding(prepared_project):
    result = validate_prepared_data_cache(
        root=prepared_project.root,
        include_high_resolution_variants=False,
    )
    assert result["status"] == "validated"
    assert result["protected_target_values_hashed"] == 0
    assert result["image_files_inventoried"] == 14
    assert result["raw_image_content_fully_hashed"] is False
    assert result["critical_prepared_artifacts_fully_hashed"] is True


def test_source_inventory_guard_does_not_claim_full_unsampled_content_assurance(tmp_path):
    tree = tmp_path / "images"
    tree.mkdir()
    for index in range(65):
        (tree / f"{index:03d}.bin").write_bytes(bytes([index]) * 8)
    before = _tree_fingerprint(tree, sample_size=4)
    (tree / "001.bin").write_bytes(b"changed!")
    after = _tree_fingerprint(tree, sample_size=4)
    assert before == after


def test_prepared_data_cache_rejects_a_changed_critical_artifact(prepared_project):
    label_maps = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    label_maps["articleType"]["classes"].append("STALE")
    prepared_project.label_maps.write_text(
        json.dumps(label_maps, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="critical prepared-data artifact is stale"):
        validate_prepared_data_cache(
            root=prepared_project.root,
            include_high_resolution_variants=False,
        )
