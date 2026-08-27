from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.task4.analysis import (
    CanvasStressEvaluation,
    build_query_support,
    evaluate_canvas_stress,
    mark_failure_slices,
    select_example_ids,
    summarize_failure_slices,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import FeatureIndex, PairEvaluation
from fashion.task4.protocol import RetrievalViews


def test_failure_slice_boundaries_are_frozen() -> None:
    marked = mark_failure_slices(
        pd.DataFrame(
            {
                "query_id": list(range(1, 13)),
                "mode": ["L", "RGB", "RGB", "RGB", "RGB", "RGB"] * 2,
                "aspect_ratio": [0.75, 0.75, 1.0, 0.75, 0.75, 0.75] * 2,
                "primary_positive_count": [0, 1, 9, 10, 11, 10] * 2,
                "primary_strict_count": [0, 1, 9, 10, 11, 10] * 2,
                "family_positive_count": [0, 1, 4, 5, 6, 10] * 2,
            }
        )
    )

    assert marked["grayscale"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ] * 2
    assert marked["rare_article_type"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ] * 2
    assert marked["rare_type_colour"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ] * 2
    assert marked["unusual_geometry"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ] * 2
    assert marked["family_unavailable"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ] * 2
    assert marked["weak_family"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ] * 2


def test_query_support_uses_gallery_counts_and_family_exclusions() -> None:
    primary_queries = pd.DataFrame(
        {
            "id": [2, 1],
            "articleType": ["Dress", "Shirts"],
            "baseColour": ["Red", "Blue"],
            "mode": ["RGB", "L"],
            "aspect_ratio": [1.0, 0.75],
        }
    )
    primary_gallery = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "articleType": ["Shirts", "Shirts", "Dress"],
            "baseColour": ["Blue", "Green", "Red"],
        }
    )
    family_queries = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "product_family_group": ["family-a"] * 4,
            "duplicate_group": ["dup-1", "dup-2", "dup-1", "dup-4"],
            "sha256": ["sha-1", "sha-2", "sha-3", "sha-1"],
        }
    )

    support = build_query_support(
        RetrievalViews(primary_queries, primary_gallery),
        RetrievalViews(family_queries, family_queries),
    )

    assert support.to_dict("list") == {
        "query_id": [1, 2],
        "mode": ["L", "RGB"],
        "aspect_ratio": [0.75, 1.0],
        "primary_positive_count": [2, 1],
        "primary_strict_count": [1, 1],
        "family_positive_count": [1, 3],
    }


def test_slice_summary_keeps_unavailable_family_null_and_visible() -> None:
    membership = mark_failure_slices(
        pd.DataFrame(
            {
                "query_id": [1, 2],
                "mode": ["L", "RGB"],
                "aspect_ratio": [0.75, 1.0],
                "primary_positive_count": [2, 20],
                "primary_strict_count": [1, 20],
                "family_positive_count": [0, 3],
            }
        )
    )
    context = {
        "scope": "development",
        "fold": 1,
        "size": "240x320",
        "query_source": "v1",
        "gallery_source": "v1",
    }
    query_metrics = pd.DataFrame.from_records(
        [
            {
                **context,
                "protocol": "primary",
                "query_id": 1,
                "ndcg_at_10": 0.25,
                "recall_at_10": np.nan,
            },
            {
                **context,
                "protocol": "primary",
                "query_id": 2,
                "ndcg_at_10": 0.75,
                "recall_at_10": np.nan,
            },
            {
                **context,
                "protocol": "family",
                "query_id": 1,
                "ndcg_at_10": np.nan,
                "recall_at_10": np.nan,
            },
            {
                **context,
                "protocol": "family",
                "query_id": 2,
                "ndcg_at_10": np.nan,
                "recall_at_10": 0.5,
            },
        ]
    )

    summary = summarize_failure_slices(query_metrics, membership)

    assert summary.columns.tolist() == [
        "scope",
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "protocol",
        "slice",
        "metric",
        "k",
        "aggregation",
        "value",
        "total_queries",
        "scored_queries",
        "excluded_queries",
        "coverage",
    ]
    assert summary["slice"].tolist() == [
        "grayscale",
        "rare_article_type",
        "rare_type_colour",
        "unusual_geometry",
        "family_unavailable",
        "weak_family",
    ]
    unavailable = summary.loc[summary["slice"].eq("family_unavailable")].iloc[0]
    assert pd.isna(unavailable["value"])
    assert unavailable[
        ["total_queries", "scored_queries", "excluded_queries", "coverage"]
    ].tolist() == pytest.approx([1, 0, 1, 0.0])
    unusual = summary.loc[summary["slice"].eq("unusual_geometry")].iloc[0]
    assert unusual["protocol"] == "primary"
    assert unusual["metric"] == "ndcg"
    assert unusual["value"] == pytest.approx(0.75)


def _canvas_views() -> RetrievalViews:
    return RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [2, 1],
                "articleType": ["Shirts", "Shirts"],
                "baseColour": ["Blue", "Blue"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": list(range(10, 21)),
                "articleType": ["Shirts", *(["Dress"] * 10)],
                "baseColour": ["Blue", *(["Red"] * 10)],
            }
        ),
    )


def _canvas_index(source: str, ids: list[int], features: list[list[float]]) -> FeatureIndex:
    return FeatureIndex(
        source=source,
        contract=PreprocessingContract(width=240, height=320),
        ids=np.asarray(ids, dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
        transform_seconds=1.0,
        source_bytes=100,
    )


def _clean_canvas_pair() -> PairEvaluation:
    rankings = pd.DataFrame.from_records(
        [
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "distance": float(rank - 1),
                "rank": rank,
            }
            for query_id in (1, 2)
            for rank, candidate_id in enumerate(range(10, 20), start=1)
        ]
    )
    per_query = pd.DataFrame(
        {
            "query_id": [1, 2],
            "articleType": ["Shirts", "Shirts"],
            "ndcg_at_10": [1.0, 1.0],
        }
    )
    summary = pd.DataFrame(
        {
            "fold": [1],
            "size": ["240x320"],
            "query_source": ["v1"],
            "gallery_source": ["v1"],
            "protocol": ["primary"],
        }
    )
    return PairEvaluation(
        summary=summary,
        primary_rankings=rankings,
        family_rankings=pd.DataFrame(),
        primary_per_query=per_query,
        family_per_query=pd.DataFrame(),
    )


def test_canvas_stress_keeps_per_query_scores_changes_and_top10_overlap() -> None:
    gallery_features = [[1.0, 0.0], *([[0.0, 1.0]] * 10)]
    clean = _clean_canvas_pair()

    result = evaluate_canvas_stress(
        clean,
        {
            "wide": _canvas_index("v1", [2, 1], [[1.0, 0.0], [1.0, 0.0]]),
            "tall": _canvas_index("v1", [2, 1], [[0.0, 1.0], [0.0, 1.0]]),
        },
        _canvas_index("v1", list(range(10, 21)), gallery_features),
        _canvas_views(),
    )

    assert isinstance(result, CanvasStressEvaluation)
    assert result.per_query["query_variant"].tolist() == [
        "clean",
        "wide",
        "tall",
        "clean",
        "wide",
        "tall",
    ]
    assert set(result.rankings) == {"clean", "wide", "tall"}
    clean_rows = result.per_query.loc[result.per_query["query_variant"].eq("clean")]
    assert clean_rows["ndcg_change_from_clean"].tolist() == pytest.approx([0.0, 0.0])
    assert clean_rows["top10_overlap"].tolist() == pytest.approx([1.0, 1.0])
    tall_rows = result.per_query.loc[result.per_query["query_variant"].eq("tall")]
    assert (tall_rows["canvas_ndcg_at_10"] < tall_rows["clean_ndcg_at_10"]).all()
    assert tall_rows["top10_overlap"].tolist() == pytest.approx([0.9, 0.9])
    assert result.summary["query_variant"].tolist() == ["clean", "wide", "tall"]


def test_example_ids_use_v1_scores_slice_rules_and_numeric_ties() -> None:
    membership = pd.DataFrame(
        {
            "query_id": [9, 2, 4],
            "grayscale": [False, True, True],
            "rare_article_type": [False, True, True],
            "rare_type_colour": [False, False, False],
            "unusual_geometry": [False, False, False],
            "family_unavailable": [False, True, False],
            "weak_family": [False, False, True],
        }
    )
    context = {
        "scope": "development",
        "fold": 1,
        "size": "240x320",
        "query_source": "v1",
        "gallery_source": "v1",
    }
    query_metrics = pd.DataFrame.from_records(
        [
            {
                **context,
                "protocol": protocol,
                "query_id": query_id,
                "ndcg_at_10": ndcg if protocol == "primary" else np.nan,
                "recall_at_10": recall if protocol == "family" else np.nan,
            }
            for query_id, ndcg, recall in (
                (9, 0.9, 1.0),
                (2, 0.2, np.nan),
                (4, 0.2, 0.1),
            )
            for protocol in ("primary", "family")
        ]
    )
    canvas = pd.DataFrame(
        {
            "query_id": [9, 4, 2],
            "query_variant": ["wide", "tall", "tall"],
            "ndcg_change_from_clean": [-0.1, -0.5, -0.5],
        }
    )

    selected = select_example_ids(query_metrics, membership, canvas)

    assert selected == {
        "normal_success": 9,
        "grayscale": 2,
        "rare_article_type": 2,
        "family_unavailable": 2,
        "weak_family": 4,
        "canvas_failure": 2,
    }
