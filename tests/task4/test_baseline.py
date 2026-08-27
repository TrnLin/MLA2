from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fashion.task4.baseline as baseline_module
from fashion.task4.baseline import (
    build_headline_summary,
    build_random_primary_rankings,
    evaluate_baseline,
    verify_preprocessing_reproduction,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import FeatureIndex
from fashion.task4.probe import PROBE_VERSION
from fashion.task4.protocol import RetrievalViews


def _retrieval_views() -> RetrievalViews:
    return RetrievalViews(
        queries=pd.DataFrame({"id": [9, 1]}),
        gallery=pd.DataFrame({"id": [40, 20, 30]}),
    )


def test_random_floor_is_seeded_and_independent_of_gallery_row_order() -> None:
    views = _retrieval_views()

    first = build_random_primary_rankings(views, seed=2753, max_k=2)
    reversed_views = RetrievalViews(
        queries=views.queries,
        gallery=views.gallery.iloc[::-1].reset_index(drop=True),
    )
    second = build_random_primary_rankings(reversed_views, seed=2753, max_k=2)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("query_id")["candidate_id"].apply(list).to_dict() == {
        1: [40, 30],
        9: [40, 30],
    }


def _split_row(product_id: int, fold: int) -> dict[str, object]:
    return {
        "id": product_id,
        "sha256": f"sha-{product_id}",
        "duplicate_group": f"duplicate-{product_id}",
        "product_name_key": f"family-{product_id}",
        "product_family_group": f"family-{product_id}",
        "partition": "development",
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "articleType": "Shirts",
        "baseColour": "Blue",
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
    }


def _baseline_splits() -> pd.DataFrame:
    rows = [_split_row(100 + offset, 1) for offset in range(11)]
    gallery_folds = (0, 2, 3, 4)
    rows.extend(
        _split_row(200 + offset, gallery_folds[offset % len(gallery_folds)])
        for offset in range(20)
    )
    return pd.DataFrame(rows)


def _feature_index(source: str, *, width: int = 240) -> FeatureIndex:
    ids = np.array([*range(100, 111), *range(200, 220)], dtype=np.int64)
    features = np.ones((len(ids), 1), dtype=np.float32)
    return FeatureIndex(
        source=source,
        contract=PreprocessingContract(width=width, height=320),
        ids=ids,
        features=features,
        transform_seconds=1.0,
        source_bytes=100,
    )


def test_baseline_runs_all_four_source_directions_and_labels_evidence() -> None:
    result = evaluate_baseline(
        _baseline_splits(),
        {"teacher": _feature_index("teacher"), "v1": _feature_index("v1")},
    )

    assert set(result.pair_evaluations) == {
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    }
    assert set(result.query_metrics["method"]) == {PROBE_VERSION}
    assert set(result.query_metrics["scope"]) == {"development"}
    assert set(result.query_metrics["fold"]) == {1}
    assert set(result.query_metrics["size"]) == {"240x320"}
    assert set(result.query_metrics["protocol"]) == {"primary", "family"}
    assert result.query_metrics.columns[:6].tolist() == [
        "method",
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "protocol",
    ]
    assert set(result.summary["method"]) == {
        PROBE_VERSION,
        "random-seed-2753",
        "headline",
    }
    assert result.summary.columns[:6].tolist() == [
        "method",
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "protocol",
    ]
    random_rows = result.summary.loc[
        result.summary["method"].eq("random-seed-2753")
    ]
    assert set(random_rows["protocol"]) == {"primary"}
    assert set(random_rows["k"]) == {5, 10, 20}
    assert result.random_rankings.groupby("query_id").size().eq(20).all()
    random_floor = result.summary.loc[
        result.summary["method"].eq("headline")
        & result.summary["metric"].eq("beats_random"),
        "passed",
    ]
    assert random_floor.item() is False


def test_baseline_passes_frozen_quality_chunk_size_to_all_four_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_chunk_sizes: list[int | None] = []
    real_evaluate_source_pair = baseline_module.evaluate_source_pair

    def record_chunk_size(*args, **kwargs):
        received_chunk_sizes.append(kwargs.get("chunk_size"))
        return real_evaluate_source_pair(*args, **kwargs)

    monkeypatch.setattr(
        baseline_module,
        "evaluate_source_pair",
        record_chunk_size,
    )

    evaluate_baseline(
        _baseline_splits(),
        {"teacher": _feature_index("teacher"), "v1": _feature_index("v1")},
    )

    assert received_chunk_sizes == [256, 256, 256, 256]


def test_baseline_rejects_non_frozen_fold_or_index_scope() -> None:
    valid_indexes = {
        "teacher": _feature_index("teacher"),
        "v1": _feature_index("v1"),
    }
    with pytest.raises(ValueError, match="fold 1"):
        evaluate_baseline(_baseline_splits(), valid_indexes, fold=2)

    with pytest.raises(ValueError, match="exactly"):
        evaluate_baseline(
            _baseline_splits(),
            {"teacher": valid_indexes["teacher"]},
        )

    with pytest.raises(ValueError, match="240x320"):
        evaluate_baseline(
            _baseline_splits(),
            {
                "teacher": _feature_index("teacher"),
                "v1": _feature_index("v1", width=96),
            },
        )


def test_headline_uses_equal_source_weight_and_records_claim_failures() -> None:
    summary = build_headline_summary(
        teacher_ndcg=0.50,
        v1_ndcg=0.40,
        teacher_to_v1_ndcg=0.30,
        v1_to_teacher_ndcg=0.30,
        random_ndcg=0.10,
    )

    assert summary["same_source_mean"] == pytest.approx(0.45)
    assert summary["cross_source_mean"] == pytest.approx(0.30)
    assert summary["beats_random"] is True
    assert summary["cross_source_within_95_percent"] is False


def _reproduction_rows() -> pd.DataFrame:
    values = {
        ("teacher", "teacher"): 0.49521342,
        ("v1", "v1"): 0.48454184,
        ("teacher", "v1"): 0.46774562,
        ("v1", "teacher"): 0.45568546,
    }
    return pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "size": "240x320",
                "query_source": query_source,
                "gallery_source": gallery_source,
                "protocol": "primary",
                "metric": "ndcg",
                "k": 10,
                "aggregation": "query_mean",
                "value": value,
            }
            for (query_source, gallery_source), value in values.items()
        ]
    )


def test_reproduction_check_accepts_all_four_selected_size_scores() -> None:
    summary = _reproduction_rows()
    summary.insert(0, "method", PROBE_VERSION)

    verify_preprocessing_reproduction(summary, _reproduction_rows(), atol=5e-8)


def test_reproduction_check_rejects_changed_selected_size_score() -> None:
    summary = _reproduction_rows()
    summary.insert(0, "method", PROBE_VERSION)
    summary.loc[
        summary["query_source"].eq("teacher")
        & summary["gallery_source"].eq("v1"),
        "value",
    ] += 1e-4

    with pytest.raises(ValueError, match="preprocessing probe"):
        verify_preprocessing_reproduction(summary, _reproduction_rows(), atol=5e-8)


def test_reproduction_failure_lists_every_mismatched_direction() -> None:
    summary = _reproduction_rows()
    summary.insert(0, "method", PROBE_VERSION)
    summary.loc[
        summary["query_source"].eq("teacher")
        & summary["gallery_source"].eq("v1"),
        "value",
    ] += 1e-4
    summary.loc[
        summary["query_source"].eq("v1")
        & summary["gallery_source"].eq("teacher"),
        "value",
    ] -= 2e-4

    with pytest.raises(ValueError) as error:
        verify_preprocessing_reproduction(summary, _reproduction_rows(), atol=5e-8)

    message = str(error.value)
    assert (
        "teacher->v1: observed=0.46784562, expected=0.46774562, "
        "absolute_delta=0.0001"
    ) in message
    assert (
        "v1->teacher: observed=0.45548546, expected=0.45568546, "
        "absolute_delta=0.0002"
    ) in message
    assert "teacher->teacher" not in message
    assert "v1->v1" not in message
