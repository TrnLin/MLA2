"""Name-based labels must never change split membership or the source dataset."""

import json

import pandas as pd
import pytest

from fashion.data import get_cv_split, load_splits
from fashion.data.gender_name_truth import (
    VARIANT_RELATIVE_PATH,
    apply_name_truth_labels,
    build_gender_name_truth_variant,
    load_gender_name_truth_variant,
    make_name_truth_labels,
    product_name_gender_cues,
)
from fashion.data.hashing import compute_sha256


@pytest.fixture
def project(tmp_path):
    names = [
        "Doodle Kids Boy Blue Tee",
        "Doodle Kids Girl Blue Tee",
        "Puma Unisex Backpack",
        "Nike Men's Shoe",
        "Women's Top",
        "Mr. Men Boys Tee",
        "Plain Backpack",
        "MEN Boys Tee",
        "Girl’s dress",
        "Women Boyfriend Jeans",
        "Girls protected name",
        "Men protected name",
    ]
    genders = [
        "Men",
        "Women",
        "Men",
        "Men",
        "Women",
        "Boys",
        "Unisex",
        "Men",
        "Women",
        "Women",
        "",
        "",
    ]
    rows = []
    for index, (name, gender) in enumerate(zip(names, genders, strict=True)):
        item = index + 1
        protected = index >= 10
        row = dict(
            id=item,
            gender=gender,
            productDisplayName=name,
            partition="development"
            if not protected
            else "holdout"
            if index == 10
            else "quarantine",
            cv_fold=index // 2 if not protected else "",
            sha256=f"{item:064x}",
            path=f"data/raw/images/{item}.jpg",
            product_family_group=f"family-{item}",
            product_name_key=f"name-{item}",
            duplicate_group="",
            is_cross_role_exact_duplicate=False,
            is_cross_role_near_duplicate=False,
            has_conflicting_target_labels=False,
            conflicting_targets="",
            quarantine_reason="" if index < 11 else "fixture",
        )
        for target in ("articleType", "season", "usage"):
            row[target] = (
                ""
                if protected
                else {"articleType": "Tshirts", "season": "Summer", "usage": "NA"}[target]
            )
        for target in ("articleType", "season", "gender", "usage"):
            row[f"has_{target}_label"] = not protected
        rows.append(row)
    path = tmp_path / "data/processed/splits.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return tmp_path


@pytest.mark.parametrize(
    "name,label,count",
    [
        ("Doodle Girl's Blue Tee", "Girls", 1),
        ("Doodle GIRL’S Blue Tee", "Girls", 1),
        ("Women Women's Tee", "Women", 1),
        ("Mens Shoe", "Men", 1),
        ("Boys' Boys Tee", "Boys", 1),
        ("Unisex Bag", "Unisex", 1),
        ("Mr. Men Boys Tee", "", 2),
        ("Enamor Women Boy Shorts", "", 2),
        ("Boyfriend Jeans", "", 0),
        ("Kids Shirt", "", 0),
        ("NA", "", 0),
        (None, "", 0),
    ],
)
def test_clear_name_cues_and_ambiguous_names(name, label, count):
    result = product_name_gender_cues(pd.Series([name], index=[42])).loc[42]
    assert result.name_gender == label and result.cue_count == count


def test_only_gender_changes_and_both_fold_roles_receive_the_same_labels(project):
    splits = load_splits(project / "data/processed/splits.csv")
    before = splits.copy(deep=True)
    labels = make_name_truth_labels(splits)
    assert not {"cv_fold", "partition", "path"} & set(labels.columns)
    assert set(labels.loc[labels.changed, "id"]) == {1, 2, 3, 9}
    variant = apply_name_truth_labels(splits, labels)
    pd.testing.assert_frame_equal(splits, before)
    pd.testing.assert_frame_equal(variant.drop(columns="gender"), splits.drop(columns="gender"))
    assert variant.set_index("id").loc[[1, 2, 3, 9], "gender"].tolist() == [
        "Boys",
        "Girls",
        "Unisex",
        "Girls",
    ]
    assert labels.loc[labels.id.eq(6), "label_source"].item() == "original_multiple_cues"
    assert labels.loc[labels.id.eq(7), "label_source"].item() == "original_no_cue"
    pd.testing.assert_frame_equal(
        variant[variant.partition.ne("development")], splits[splits.partition.ne("development")]
    )
    for fold in range(5):
        original_train, original_val = get_cv_split(splits, fold)
        train, val = get_cv_split(variant, fold)
        assert train.id.tolist() == original_train.id.tolist()
        assert val.id.tolist() == original_val.id.tolist()
        assert not set(train.product_family_group) & set(val.product_family_group)
        combined = pd.concat([train, val]).set_index("id").sort_index()
        assert combined.gender.equals(
            variant[variant.partition.eq("development")].set_index("id").sort_index().gender
        )


def test_build_load_and_rebuild_are_exact_without_raw_data(project):
    source = project / "data/processed/splits.csv"
    before = source.read_bytes()
    summary = build_gender_name_truth_variant(project)
    assert summary["changed_labels"] == 4 and summary["protected_rows_unchanged"] == 2
    assert summary["no_cue_rows"] == 1 and summary["multiple_cue_rows"] == 2
    assert summary["independent_blind_test"] is False
    assert sum(row["changed_validation_labels"] for row in summary["folds"]) == 4
    directory = project / VARIANT_RELATIVE_PATH
    contents = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert build_gender_name_truth_variant(project) == summary
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == contents
    loaded = load_gender_name_truth_variant(project)
    assert loaded.attrs["gender_label_variant"]["labels_sha256"] == compute_sha256(
        directory / "labels.csv"
    )
    assert loaded.partition.value_counts().to_dict() == {
        "development": 10,
        "holdout": 1,
        "quarantine": 1,
    }
    assert source.read_bytes() == before
    assert not (directory / "splits.csv").exists() and not (project / "data/raw").exists()


@pytest.mark.parametrize("fault", ["wrong_gender", "missing_id", "duplicate_id", "protected_id"])
def test_apply_rejects_wrong_labels_or_membership(project, fault):
    splits = load_splits(project / "data/processed/splits.csv")
    labels = make_name_truth_labels(splits)
    if fault == "wrong_gender":
        labels.loc[0, "gender"] = "Women"
    elif fault == "missing_id":
        labels = labels.iloc[1:]
    elif fault == "duplicate_id":
        labels.loc[1, "id"] = labels.loc[0, "id"]
    else:
        labels.loc[0, "id"] = 11
    with pytest.raises(ValueError, match="match|unique"):
        apply_name_truth_labels(splits, labels)


def test_stale_source_and_tampered_artifacts_cannot_load_or_overwrite(project):
    build_gender_name_truth_variant(project)
    directory = project / VARIANT_RELATIVE_PATH
    path = directory / "labels.csv"
    labels = pd.read_csv(path, keep_default_na=False)
    labels.loc[0, "gender"] = "Women"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_gender_name_truth_variant(project)
    # Even updating a manifest hash cannot bypass the name-based contract.
    manifest = directory / "summary.json"
    summary = json.loads(manifest.read_text())
    summary["files"]["labels.csv"] = compute_sha256(path)
    manifest.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="frozen rule"):
        load_gender_name_truth_variant(project)
    with pytest.raises(FileExistsError, match="Different variant"):
        build_gender_name_truth_variant(project)
    source = project / "data/processed/splits.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    frame.loc[0, "productDisplayName"] = "Girls changed source"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Canonical split changed"):
        load_gender_name_truth_variant(project)


def test_conflicting_name_truth_on_identical_images_is_reported_without_moving_rows(project):
    source = project / "data/processed/splits.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    for column in ("gender", "sha256", "product_family_group"):
        frame.loc[1, column] = frame.loc[0, column]
    frame.to_csv(source, index=False)
    summary = build_gender_name_truth_variant(project)
    assert summary["same_image_conflicting_variant_label_groups"] == 1
    assert summary["same_image_conflicting_variant_label_rows"] == 2
    variant = load_gender_name_truth_variant(project)
    assert variant.loc[0, "gender"] == "Boys" and variant.loc[1, "gender"] == "Girls"
    assert variant.loc[[0, 1], "cv_fold"].eq(0).all()
    assert variant.loc[[0, 1], "partition"].eq("development").all()
