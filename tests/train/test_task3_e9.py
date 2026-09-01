from __future__ import annotations

import pandas as pd
import pytest

from fashion.config import SPLITS_CSV
from fashion.data import load_splits
from fashion.train.task3_e9 import (
    GENDER_E9_EXPECTED_CONFLICT_IDS_HASH,
    GENDER_E9_EXPECTED_DISTRIBUTION_HASHES,
    build_usage_article_type_contract,
    gender_conflict_audit,
    gender_semantic_conflicts,
    prepare_gender_e9_training,
    verify_gender_e9_deterministic_audit,
)
from fashion.train.task3_experiments import (
    gender_gem_p3_early_stopping_spec,
    gender_gem_p3_spec,
    gender_semantic_filter_spec,
    usage_exception_balance_spec,
)


def test_gender_semantic_rule_is_conservative_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "id": range(8),
            "gender": ["Men", "Women", "Men", "Women", "Men", "Women", "Men", "Women"],
            "productDisplayName": [
                "Levis Kids Boy's Polo",
                "Disney Kids Girl Pink Top",
                "Boys Fangear Tee",
                "Girls Party Dress",
                "Doodle Boy Little Hero Tee",
                "Doodle Girl Little Princess Dress",
                "Men Boys Sports Tee",
                "Women Girls Casual Top",
            ],
        }
    )

    annotated = gender_semantic_conflicts(frame)

    assert annotated["e9_semantic_conflict"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_gender_filter_matches_the_locked_305_row_development_audit() -> None:
    splits = load_splits(SPLITS_CSV)
    audit = gender_conflict_audit(splits)

    assert audit["summary"]["conflict_rows"] == 305
    assert audit["summary"]["source_class_counts"] == {"Men": 168, "Women": 137}
    assert audit["summary"]["training_removals"] == {
        "0": 242,
        "1": 224,
        "2": 259,
        "3": 263,
        "4": 232,
    }
    assert audit["summary"]["conflict_ids_hash"] == GENDER_E9_EXPECTED_CONFLICT_IDS_HASH
    assert (
        audit["summary"]["distribution_hashes"]
        == GENDER_E9_EXPECTED_DISTRIBUTION_HASHES
    )
    verification = verify_gender_e9_deterministic_audit(audit)
    assert verification["verified"]
    assert not verification["human_rating_gate_required"]
    for fold, expected in enumerate((242, 224, 259, 263, 232)):
        training = splits.loc[
            splits["partition"].eq("development")
            & splits["has_gender_label"]
            & splits["cv_fold"].ne(fold)
        ]
        selected, excluded, metadata = prepare_gender_e9_training(training)
        assert len(excluded) == expected
        assert len(selected) == len(training) - expected
        assert metadata["excluded_rows"] == expected


def test_usage_contract_uses_fixed_ties_and_mean_one_group_factors() -> None:
    training = pd.DataFrame(
        {
            "id": range(8),
            "usage": [
                "Casual",
                "Ethnic",
                "Casual",
                "Sports",
                "Sports",
                "Sports",
                "Formal",
                "Casual",
            ],
            "articleType": ["Top", "Top", "Top", "Shoe", "Shoe", "Shoe", "Bag", "Bag"],
        }
    )
    classes = ("Casual", "Ethnic", "Formal", "Sports")

    annotated, mapping, metadata = build_usage_article_type_contract(training, classes=classes)

    top = mapping.set_index("articleType").loc["Top"]
    bag = mapping.set_index("articleType").loc["Bag"]
    assert top["usual_usage"] == "Casual"
    assert top["exception_count"] == 1
    assert bag["is_tie"]
    assert bag["usual_usage"] == "Casual"
    assert bag["tied_usages"] == "Casual|Formal"
    assert annotated["e9_usage_group"].value_counts().to_dict() == {
        "usual": 6,
        "exception": 2,
    }
    assert annotated["e9_group_factor"].mean() == pytest.approx(1.0)
    assert metadata["group_factors"] == pytest.approx(
        {"usual": 8 / 12, "exception": 2.0}
    )


def test_e9_specs_keep_the_exact_e6_and_e2_parents() -> None:
    gender_parents = tuple(f"gender-e6-{fold}" for fold in range(5))
    usage_parents = tuple(f"usage-e2-{fold}" for fold in range(5))

    gender = gender_semantic_filter_spec(gender_parents)
    usage = usage_exception_balance_spec(usage_parents)

    assert gender.parent_artifact_dir == "experiments/t3_gender_e6_gem_p3"
    assert gender.model_family == "task3_small_cnn_gem_p3"
    assert gender.loss_name == "cross_entropy"
    assert gender.training_selection_strategy == "gender_semantic_conflicts_v1"
    assert usage.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert usage.model_family == "task3_small_cnn"
    assert usage.loss_name == "effective_number_group_balanced_cross_entropy"
    assert usage.training_selection_strategy == "usage_article_type_exception_balance_v1"


def test_new_serialisation_keeps_historical_e6_and_e8_contract_shapes() -> None:
    parents = tuple(f"parent-{fold}" for fold in range(5))

    e6 = gender_gem_p3_spec(parents).to_dict()
    e8 = gender_gem_p3_early_stopping_spec(parents).to_dict()
    e9 = gender_semantic_filter_spec(parents).to_dict()

    assert "checkpoint_policy" not in e6
    assert "training_selection_strategy" not in e6
    assert e8["checkpoint_policy"] == "best_validation_macro_f1"
    assert "training_selection_strategy" not in e8
    assert e9["checkpoint_policy"] == "final_epoch"
    assert e9["training_selection_strategy"] == "gender_semantic_conflicts_v1"
