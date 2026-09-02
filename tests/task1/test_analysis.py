import pandas as pd
import pytest

from fashion.task1.analysis import (
    build_task1_confusion_pairs,
    build_task1_decision_evidence,
    build_task1_problem_profile,
    build_task1_weak_class_table,
)


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "partition": [
                "development",
                "development",
                "development",
                "development",
                "holdout",
                "quarantine",
            ],
            "articleType": ["Shirt", "Shirt", "Shoe", "Bag", "Changed", "Changed"],
            "width": [60, 60, 70, 60, 1, 999],
            "height": [80, 80, 80, 80, 1, 999],
            "mode": ["RGB", "L", "RGB", "RGB", "L", "RGB"],
        }
    )


def _class_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["articleType", "articleType", "articleType", "gender"],
            "class_name": ["Shirt", "Shoe", "Bag", "Men"],
            "rare_warning": ["", "rare", "rare", "rare"],
            "untrainable_fold_count": [0, 1, 0, 99],
        }
    )


def _per_class() -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "class_index": [0, 1, 2],
            "class_name": ["Bag", "Shirt", "Shoe"],
            "support": [2, 4, 1],
            "precision": [0.2, 0.9, 0.5],
            "recall": [0.1, 0.8, 0.4],
            "f1": [0.13, 0.85, 0.44],
        }
    )
    return {"hog-svm": frame, "scratch-cnn": frame.assign(f1=[0.2, 0.8, 0.3])}


def _oof_predictions() -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "id": [6, 3, 2, 8, 7],
            "true_label": ["Shoe", "Bag", "Bag", "Shirt", "Shirt"],
            "predicted_label": ["Bag", "Shoe", "Shoe", "Bag", "Bag"],
        }
    )
    return {"hog-svm": frame, "scratch-cnn": frame}


def test_problem_profile_summarises_task1_eda() -> None:
    profile = build_task1_problem_profile(_splits(), _class_summary())
    assert profile.development_products == 4
    assert profile.class_count == 3
    assert profile.minimum_class_products == 1
    assert profile.maximum_class_products == 2
    assert profile.grayscale_images == 1
    assert profile.unusual_geometry_images == 1
    assert profile.classes_with_fold_warnings == 2
    assert profile.classes_untrainable_in_any_fold == 1


def test_decision_evidence_connects_data_problem_and_test() -> None:
    table = build_task1_decision_evidence(build_task1_problem_profile(_splits(), _class_summary()))
    assert list(table.columns) == ["evidence", "problem", "choice_to_test"]
    assert len(table) == 7
    assert table["choice_to_test"].str.contains("macro-F1", regex=False).any()
    assert table["choice_to_test"].str.contains("HOG", regex=False).any()
    assert table["choice_to_test"].str.contains("scratch CNN", regex=False).any()


@pytest.mark.parametrize("column", ["width", "height", "mode"])
def test_problem_profile_rejects_missing_geometry_columns(column: str) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_task1_problem_profile(_splits().drop(columns=column), _class_summary())


def test_problem_profile_rejects_duplicate_development_ids() -> None:
    splits = _splits()
    splits.loc[1, "id"] = 1
    with pytest.raises(ValueError, match="unique development"):
        build_task1_problem_profile(splits, _class_summary())


def test_problem_profile_rejects_missing_article_type_summary() -> None:
    with pytest.raises(ValueError, match="articleType class summary"):
        build_task1_problem_profile(_splits(), _class_summary().query("target != 'articleType'"))


@pytest.mark.parametrize("column", ["target", "untrainable_fold_count", "rare_warning"])
def test_problem_profile_rejects_missing_class_summary_columns(column: str) -> None:
    with pytest.raises(ValueError, match="class summary is missing columns"):
        build_task1_problem_profile(_splits(), _class_summary().drop(columns=column))


def test_problem_profile_rejects_negative_untrainable_count() -> None:
    summary = _class_summary()
    summary.loc[0, "untrainable_fold_count"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        build_task1_problem_profile(_splits(), summary)


def test_problem_profile_ignores_protected_row_labels_and_geometry() -> None:
    splits = _splits()
    protected = splits["partition"].ne("development")
    splits.loc[protected, "articleType"] = "Anything"
    splits.loc[protected, "width"] = 12345
    splits.loc[protected, "height"] = 54321
    splits.loc[protected, "mode"] = "L"
    assert build_task1_problem_profile(splits, _class_summary()) == build_task1_problem_profile(
        _splits(), _class_summary()
    )


def test_failure_tables_keep_candidate_identity_and_example_ids() -> None:
    weak = build_task1_weak_class_table(_per_class(), limit=2)
    pairs = build_task1_confusion_pairs(_oof_predictions(), limit=2)
    assert list(weak.columns) == [
        "candidate_id", "class_index", "class_name", "support", "precision", "recall", "f1"
    ]
    assert weak.groupby("candidate_id").size().eq(2).all()
    assert list(pairs.columns) == [
        "candidate_id", "true_label", "predicted_label", "error_count", "example_ids"
    ]
    assert pairs["example_ids"].str.split(",").map(len).le(3).all()


@pytest.mark.parametrize(
    ("builder", "evidence"),
    [
        (build_task1_weak_class_table, {"one": pd.DataFrame()}),
        (build_task1_confusion_pairs, {"one": pd.DataFrame()}),
    ],
)
def test_failure_tables_validate_required_columns(
    builder: object, evidence: dict[str, pd.DataFrame]
) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        builder(evidence)  # type: ignore[operator]


@pytest.mark.parametrize("builder", [build_task1_weak_class_table, build_task1_confusion_pairs])
def test_failure_tables_reject_non_positive_limit(builder: object) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        builder({}, limit=0)  # type: ignore[operator]
