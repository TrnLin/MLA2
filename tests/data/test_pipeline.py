from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import TARGET_COLUMNS
from fashion.data.dataset import (
    FashionDataset,
    get_cv_split,
    iter_cv_folds,
    load_label_maps,
    load_splits,
    load_splits_for_final_evaluation,
)
from fashion.data.pipeline import _BASE_ARTIFACTS, prepare_data, validate_prepared_data_cache
from fashion.data.splits import id_set_digest, validate_splits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {relative: _sha256(root / relative) for relative in _BASE_ARTIFACTS}


def test_prepare_data_writes_teacher_only_development_contract(prepared_project) -> None:
    splits = pd.read_csv(prepared_project.splits, keep_default_na=False)
    validate_splits(splits)

    assert set(splits["partition"]) == {"development", "holdout", "quarantine"}
    development = splits["partition"].eq("development")
    protected = splits["partition"].isin({"holdout", "quarantine"})
    assert set(splits.loc[development, "cv_fold"].astype(int)) == set(range(5))
    assert splits.loc[protected, "cv_fold"].eq("").all()
    assert not any("supported" in column or "deployed" in column for column in splits)
    for target in TARGET_COLUMNS:
        assert splits.loc[protected, target].eq("").all()
        assert not (
            splits.loc[protected, f"has_{target}_label"]
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )

    for path in (
        prepared_project.cv_fold_summary,
        prepared_project.development_summary,
        prepared_project.development_image_profile,
        prepared_project.label_maps,
        prepared_project.split_summary,
        prepared_project.taxonomy,
    ):
        assert path.is_file()
    assert not (prepared_project.processed / "paired_normalization.json").exists()
    assert not (prepared_project.processed / "training_image_variants.csv.gz").exists()

    profile = json.loads(prepared_project.development_image_profile.read_text(encoding="utf-8"))
    assert profile["source_scope"] == "development"
    assert profile["allowed_for_model_fit"] is False
    cache_result = validate_prepared_data_cache(prepared_project.root)
    assert cache_result["shared_source_policy"] == "teacher_only"
    assert cache_result["protected_target_values_hashed"] == 0


def test_cv_helpers_reuse_only_precomputed_folds(prepared_project) -> None:
    splits = load_splits(prepared_project.splits)
    development_ids = set(splits.loc[splits["partition"].eq("development"), "id"])
    rounds = list(iter_cv_folds(splits))
    assert [fold for fold, _, _ in rounds] == list(range(5))
    for fold, training, validation in rounds:
        assert set(training["id"]).isdisjoint(validation["id"])
        assert set(training["id"]) | set(validation["id"]) == development_ids
        assert validation["cv_fold"].eq(fold).all()
        assert training["cv_fold"].ne(fold).all()
        assert training["partition"].eq("development").all()
        assert validation["partition"].eq("development").all()
    with pytest.raises(ValueError, match="validation_fold"):
        get_cv_split(splits, 5)


def test_dataset_requires_an_explicit_task_transform(prepared_project) -> None:
    splits = load_splits(prepared_project.splits)
    development = splits[splits["partition"].eq("development")].head(1)
    dataset = FashionDataset(
        development,
        transform=lambda path: path.name,
        root=prepared_project.root,
    )
    sample = dataset[0]
    assert sample["image"].endswith(".jpg")
    assert sample["partition"] == "development"
    assert int(sample["cv_fold"]) in range(5)
    with pytest.raises(TypeError):
        FashionDataset(development, root=prepared_project.root)  # type: ignore[call-arg]


def test_label_maps_keep_every_nonblank_development_label(prepared_project) -> None:
    splits = load_splits(prepared_project.splits)
    development = splits[splits["partition"].eq("development")]
    mappings = load_label_maps(prepared_project.label_maps)
    for target in TARGET_COLUMNS:
        expected = sorted(
            development.loc[development[f"has_{target}_label"], target].astype(str).unique()
        )
        assert mappings[target]["source_scope"] == "development"
        assert mappings[target]["classes"] == expected
    assert "NA" in mappings["usage"]["classes"]


def test_final_loader_requires_explicit_unlock(prepared_project) -> None:
    with pytest.raises(ValueError, match="stay sealed"):
        load_splits_for_final_evaluation(prepared_project.splits)
    normal = load_splits(prepared_project.splits)
    protected = normal["partition"].isin({"holdout", "quarantine"})
    assert normal.loc[protected, list(TARGET_COLUMNS)].eq("").all().all()

    unlocked = load_splits_for_final_evaluation(
        prepared_project.splits,
        evaluation_unlocked=True,
        raw_teacher_csv=prepared_project.train_csv,
    )
    assert unlocked.loc[protected, "articleType"].astype(str).str.strip().ne("").all()


def test_missing_canonical_split_is_not_silently_recreated(tiny_project) -> None:
    with pytest.raises(FileNotFoundError, match="refuses to recreate protected membership"):
        prepare_data(root=tiny_project.root, workers=1)


def test_repeated_full_build_is_byte_stable_and_needs_no_external_folder(
    prepared_project,
) -> None:
    assert not (prepared_project.root / "data/raw/external").exists()
    before = _artifact_hashes(prepared_project.root)
    prepare_data(root=prepared_project.root, workers=2)
    after = _artifact_hashes(prepared_project.root)
    assert after == before


def test_protected_target_sentinel_cannot_change_public_artifacts(prepared_project) -> None:
    splits = load_splits(prepared_project.splits)
    protected_ids = set(
        splits.loc[splits["partition"].isin({"holdout", "quarantine"}), "id"].astype(int)
    )
    before = _artifact_hashes(prepared_project.root)
    raw = pd.read_csv(prepared_project.train_csv, keep_default_na=False)
    protected = raw["id"].astype(int).isin(protected_ids)
    for target in TARGET_COLUMNS:
        raw.loc[protected, target] = f"PROTECTED_SENTINEL_{target}"
    raw.to_csv(prepared_project.train_csv, index=False, lineterminator="\n")

    prepare_data(root=prepared_project.root, workers=2)
    after = _artifact_hashes(prepared_project.root)
    assert after == before


def test_id_set_digest_has_a_declared_newline_contract() -> None:
    expected = hashlib.sha256(b"1\n2\n3\n").hexdigest()
    assert id_set_digest([3, 1, 2]) == expected
