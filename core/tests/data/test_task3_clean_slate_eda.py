from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fashion.data.hashing import compute_sha256
from fashion.data.task3_clean_slate_eda import (
    DIAGNOSTIC_FEATURES,
    _hog_descriptor,
    _probe_feature,
    analyse_observability_reviews,
    build_clean_slate_audit_contract,
    build_observability_review_pack,
    build_teacher_image_diagnostics,
    family_and_component_audit,
    fold_artifact_audit,
    foreground_proposal,
    foreground_views,
    validate_observability_reviewer,
    write_clean_slate_eda_tables,
)


def _development_rows(root: Path) -> pd.DataFrame:
    rows = []
    genders = ("Boys", "Girls", "Men", "Unisex", "Women")
    usages = (
        "NA",
        "Smart Casual",
        "Travel",
        "Party",
        "Home",
        "Casual",
        "Sports",
        "Ethnic",
        "Formal",
        "Casual",
    )
    image_dir = root / "data/raw/teacher/train/images_train"
    image_dir.mkdir(parents=True)
    for index in range(10):
        item_id = index + 1
        path = image_dir / f"{item_id}.jpg"
        image = np.full((80, 60, 3), 255, dtype=np.uint8)
        image[10 + index : 45 + index, 12:48] = (
            10 + index * 10,
            30 + index * 8,
            60 + index * 6,
        )
        Image.fromarray(image).save(path)
        rows.append(
            {
                "id": item_id,
                "partition": "development",
                "cv_fold": index % 5,
                "path": path.relative_to(root).as_posix(),
                "product_family_group": f"family_{item_id}",
                "duplicate_group": f"duplicate_{item_id}",
                "gender": genders[index % len(genders)],
                "usage": usages[index],
                "articleType": "Top" if index % 2 else "Shoe",
                "has_gender_label": True,
                "has_usage_label": True,
            }
        )
    return pd.DataFrame(rows)


def test_foreground_rule_is_border_connected_and_has_safe_blank_fallback() -> None:
    image = np.full((80, 60, 3), 255, dtype=np.uint8)
    image[20:60, 15:45] = (20, 40, 60)
    proposal = foreground_proposal(image)

    assert proposal.fallback_reason == ""
    assert 0.20 < proposal.raw_fraction < 0.30
    assert proposal.mask[30, 30]
    assert not proposal.mask[0, 0]
    assert set(foreground_views(image)) == {
        "full",
        "foreground_masked",
        "foreground_letterbox",
        "border",
        "silhouette",
        "grayscale",
    }

    blank = foreground_proposal(np.full((80, 60, 3), 255, dtype=np.uint8))
    assert blank.fallback_reason == "empty_or_tiny_foreground"
    assert blank.mask.all()


def test_probe_representations_standardize_mixed_source_shapes(tmp_path: Path) -> None:
    paths = []
    for name, size in (("usual.jpg", (60, 80)), ("wide.jpg", (100, 50))):
        path = tmp_path / name
        Image.new("RGB", size, (20, 40, 60)).save(path)
        paths.append(path)
    diagnostics = {feature: 0.0 for feature in DIAGNOSTIC_FEATURES}

    probe_shapes = {
        _probe_feature(path, "full_rgb_hog", diagnostics).shape for path in paths
    }
    neighbourhood_shapes = {_hog_descriptor(path).shape for path in paths}

    assert len(probe_shapes) == 1
    assert len(neighbourhood_shapes) == 1


def test_observability_pack_keeps_labels_out_of_reviewer_files(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    pack = build_observability_review_pack(
        splits,
        output_dir=tmp_path / "review",
        root=tmp_path,
        sample_per_class=1,
    )

    assert pack["summary"].iloc[0]["review_status"] == "deferred_non_blocking"
    assert int(pack["summary"].iloc[0]["ultra_rare_usage_rows"]) == 5
    for path in pack["reviewer_paths"]:
        reviewer = pd.read_csv(path, keep_default_na=False)
        assert "gender" not in reviewer
        assert "usage" not in reviewer
        assert reviewer["gender_knowability"].eq("").all()
        assert reviewer["usage_knowability"].eq("").all()
    assert not pack["answer_key_path"].exists()


def test_observability_answers_are_locked_before_key_creation(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    output = tmp_path / "review"
    pack = build_observability_review_pack(
        splits,
        output_dir=output,
        root=tmp_path,
        sample_per_class=1,
    )
    for path in pack["reviewer_paths"]:
        reviewer = pd.read_csv(path, keep_default_na=False)
        reviewer["gender_knowability"] = "clear"
        reviewer["gender_judgement"] = [
            "Men" if index % 2 else "Women" for index in range(len(reviewer))
        ]
        reviewer["gender_confidence"] = "high"
        reviewer["usage_knowability"] = "clear"
        reviewer["usage_judgement"] = [
            "Casual" if index % 2 else "Sports" for index in range(len(reviewer))
        ]
        reviewer["usage_confidence"] = "medium"
        reviewer.to_csv(path, index=False)

    result = analyse_observability_reviews(
        splits, output_dir=output, root=tmp_path, sample_per_class=1
    )

    assert pack["answer_key_path"].is_file()
    assert set(result["summary"]["target"]) == {"gender", "usage"}
    assert result["summary"]["reviewer_agreement"].eq(1.0).all()
    assert set(result["by_fold"]["cv_fold"]) == {0, 1, 2, 3, 4}

    changed = pd.read_csv(pack["reviewer_paths"][0], keep_default_na=False)
    changed.loc[0, "reviewer_notes"] = "changed after lock"
    changed.to_csv(pack["reviewer_paths"][0], index=False)
    with np.testing.assert_raises_regex(ValueError, "changed after they were locked"):
        analyse_observability_reviews(
            splits, output_dir=output, root=tmp_path, sample_per_class=1
        )


def test_observability_rejects_subset_reordering_and_partial_cannot_tell(
    tmp_path: Path,
) -> None:
    splits = _development_rows(tmp_path)
    pack = build_observability_review_pack(
        splits,
        output_dir=tmp_path / "review",
        root=tmp_path,
        sample_per_class=1,
    )
    expected = pd.read_csv(
        tmp_path / "review/observability_review_scope.csv", keep_default_na=False
    )
    reviewer = pd.read_csv(pack["reviewer_paths"][0], keep_default_na=False)
    reviewer["gender_knowability"] = "clear"
    reviewer["gender_judgement"] = "Men"
    reviewer["gender_confidence"] = "high"
    reviewer["usage_knowability"] = "clear"
    reviewer["usage_judgement"] = "Casual"
    reviewer["usage_confidence"] = "high"

    with np.testing.assert_raises_regex(ValueError, "frozen scope"):
        validate_observability_reviewer(reviewer.iloc[:-1], expected=expected)
    with np.testing.assert_raises_regex(ValueError, "frozen scope"):
        validate_observability_reviewer(
            reviewer.iloc[::-1].reset_index(drop=True), expected=expected
        )
    reviewer.loc[0, "gender_judgement"] = "cannot_tell"
    with np.testing.assert_raises_regex(ValueError, "cannot_tell fields"):
        validate_observability_reviewer(reviewer, expected=expected)

    shortened_scope = expected.iloc[:-1].copy()
    shortened_scope.to_csv(
        tmp_path / "review/observability_review_scope.csv", index=False
    )
    shortened_reviewer = reviewer.iloc[:-1].copy()
    shortened_reviewer.loc[:, "gender_judgement"] = "Men"
    for path in pack["reviewer_paths"]:
        shortened_reviewer.to_csv(path, index=False)
    with np.testing.assert_raises_regex(ValueError, "canonical Task 3 sample"):
        analyse_observability_reviews(
            splits, output_dir=tmp_path / "review", root=tmp_path
        )


def test_teacher_only_contract_rejects_paths_outside_teacher_train(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    outside = tmp_path / "other.jpg"
    Image.new("RGB", (60, 80), (255, 255, 255)).save(outside)

    absolute = splits.copy()
    absolute.loc[0, "path"] = str(outside)
    with np.testing.assert_raises_regex(ValueError, "absolute image paths"):
        build_clean_slate_audit_contract(absolute, root=tmp_path)

    relative = splits.copy()
    relative.loc[0, "path"] = outside.relative_to(tmp_path).as_posix()
    with np.testing.assert_raises_regex(ValueError, "non-teacher image path"):
        build_clean_slate_audit_contract(relative, root=tmp_path)


def test_audit_contract_checks_physical_images_and_task3_labels(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    splits["sha256"] = [compute_sha256(tmp_path / path) for path in splits["path"]]
    original = build_clean_slate_audit_contract(splits, root=tmp_path)

    changed_labels = splits.copy()
    changed_labels.loc[0, "usage"] = "Formal"
    changed = build_clean_slate_audit_contract(changed_labels, root=tmp_path)
    assert changed["audit_contract_hash"] != original["audit_contract_hash"]

    Image.new("RGB", (60, 80), (0, 0, 0)).save(tmp_path / splits.loc[0, "path"])
    with np.testing.assert_raises_regex(ValueError, "hash disagrees"):
        build_clean_slate_audit_contract(splits, root=tmp_path)


def test_fold_and_family_audits_preserve_the_fixed_groups(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    diagnostics = splits[["id", "cv_fold", "product_family_group"]].copy()
    for offset, feature in enumerate(DIAGNOSTIC_FEATURES):
        diagnostics[feature] = np.arange(len(diagnostics), dtype=float) + offset

    shifts = fold_artifact_audit(diagnostics)
    assert set(shifts["fold"]) == {0, 1, 2, 3, 4}
    assert set(shifts["feature"]) == set(DIAGNOSTIC_FEATURES)
    assert shifts["jensen_shannon_divergence"].between(0, 1).all()

    oof = splits[["id", "product_family_group"]].copy()
    oof["true_label"] = splits["gender"]
    oof["predicted_label"] = splits["gender"]
    oof["confidence"] = 0.9
    oof_path = tmp_path / "oof.csv"
    oof.to_csv(oof_path, index=False)

    family = family_and_component_audit(splits, anchor_oof_paths={"gender": oof_path})
    assert family["boundary"]["fold_crossings"].eq(0).all()
    assert int(family["family_profile"]["products"].sum()) == len(splits)
    assert set(family["component_weights"]["target"]) == {"gender", "usage"}
    assert family["anchor_error_slices"]["accuracy"].eq(1.0).all()

    usage_oof = splits[["id", "product_family_group"]].copy()
    usage_oof["true_label"] = splits["usage"].replace("NA", "")
    usage_oof["predicted_label"] = splits["usage"].replace("NA", "")
    usage_oof["confidence"] = 0.9
    usage_path = tmp_path / "usage_oof.csv"
    usage_oof.to_csv(usage_path, index=False)
    usage_family = family_and_component_audit(
        splits, anchor_oof_paths={"usage": usage_path}
    )
    assert usage_family["anchor_error_slices"]["accuracy"].eq(1.0).all()


def test_teacher_only_orchestrator_writes_hashed_completion_manifest(
    tmp_path: Path,
) -> None:
    splits = _development_rows(tmp_path)
    output = tmp_path / "evidence"

    result = write_clean_slate_eda_tables(
        splits=splits,
        output_dir=output,
        root=tmp_path,
        run_probes=False,
        run_neighbourhoods=False,
    )

    completion = result["completion"]
    assert completion["automated_audits_complete"] is True
    assert completion["training_screen_ready"] is True
    assert completion["training_blockers"] == []
    assert completion["human_observability_review_status"] == "deferred_non_blocking"
    assert completion["deferred_items"] == [
        "two_independent_human_observability_reviews"
    ]
    assert completion["image_scope"] == "teacher_development_images_only"
    assert completion["audit_contract_hash"]
    assert completion["artifact_manifest_sha256"]
    assert (output / "artifact_manifest.csv").is_file()
    assert set(result["artifact_manifest"]["path"]) >= {
        "audit_contract.json",
        "teacher_image_diagnostics.csv.gz",
    }

    diagnostics_path = output / "teacher_image_diagnostics.csv.gz"
    tampered = pd.read_csv(diagnostics_path, keep_default_na=False)
    tampered.loc[0, "brightness"] = -999.0
    tampered.to_csv(diagnostics_path, index=False, compression="gzip")
    rebuilt = build_teacher_image_diagnostics(
        splits,
        output_path=diagnostics_path,
        root=tmp_path,
        workers=1,
        audit_contract_hash=completion["audit_contract_hash"],
    )
    assert rebuilt["brightness"].min() >= 0


def test_label_change_invalidates_completed_human_review(tmp_path: Path) -> None:
    splits = _development_rows(tmp_path)
    output = tmp_path / "evidence"
    first = write_clean_slate_eda_tables(
        splits=splits,
        output_dir=output,
        root=tmp_path,
        run_probes=False,
        run_neighbourhoods=False,
    )
    review = first["review"]
    for path in review["reviewer_paths"]:
        reviewer = pd.read_csv(path, keep_default_na=False)
        reviewer["gender_knowability"] = "clear"
        reviewer["gender_judgement"] = "Men"
        reviewer["gender_confidence"] = "high"
        reviewer["usage_knowability"] = "clear"
        reviewer["usage_judgement"] = "Casual"
        reviewer["usage_confidence"] = "high"
        reviewer.to_csv(path, index=False)
    analyse_observability_reviews(
        splits,
        output_dir=output / "observability_review",
        root=tmp_path,
        sample_per_class=25,
    )

    changed = splits.copy()
    changed.loc[0, "usage"] = "Formal"
    rerun = write_clean_slate_eda_tables(
        splits=changed,
        output_dir=output,
        root=tmp_path,
        run_probes=False,
        run_neighbourhoods=False,
    )

    assert rerun["completion"]["human_observability_review_complete"] is False
    assert rerun["completion"]["training_screen_ready"] is True
    assert rerun["completion"]["training_blockers"] == []
    assert not (output / "observability_review/observability_review_lock.json").exists()
    assert not (
        output / "observability_review/answer_key/observability_answer_key.csv"
    ).exists()
