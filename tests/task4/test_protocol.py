"""Tests for the fold-safe Task 4 retrieval protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.task4 import protocol
from fashion.task4.protocol import (
    RetrievalViews,
    _assert_primary_isolation,
    build_development_views,
    compute_relevance_coverage,
    family_candidate_mask,
    family_relevance,
    primary_relevance,
)


def _row(
    product_id: int,
    fold: int,
    *,
    article_type: str,
    colour: str,
    family: str,
    duplicate: str,
    sha256: str,
) -> dict[str, object]:
    return {
        "id": product_id,
        "sha256": sha256,
        "duplicate_group": duplicate,
        "product_name_key": family,
        "product_family_group": family,
        "partition": "development",
        "cv_fold": fold,
        "is_cross_role_exact_duplicate": False,
        "is_cross_role_near_duplicate": False,
        "has_conflicting_target_labels": False,
        "conflicting_targets": "",
        "quarantine_reason": "",
        "articleType": article_type,
        "baseColour": colour,
        "season": "Summer",
        "gender": "Unisex",
        "usage": "Casual",
        "has_articleType_label": True,
        "has_season_label": True,
        "has_gender_label": True,
        "has_usage_label": True,
    }


def _split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                100 + fold,
                fold,
                article_type="Tshirts",
                colour="Blue",
                family=f"family-{fold}",
                duplicate=f"duplicate-{fold}",
                sha256=f"sha-{fold}",
            )
            for fold in range(5)
        ]
    )


def _coverage_split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                10,
                1,
                article_type="Tshirts",
                colour="Blue",
                family="eligible-family",
                duplicate="duplicate-10",
                sha256="sha-10",
            ),
            _row(
                11,
                1,
                article_type="Dress",
                colour="Green",
                family="eligible-family",
                duplicate="duplicate-11",
                sha256="sha-11",
            ),
            _row(
                12,
                1,
                article_type="Jeans",
                colour="Black",
                family="exact-family",
                duplicate="exact-duplicate",
                sha256="exact-sha",
            ),
            _row(
                13,
                1,
                article_type="Jeans",
                colour="Black",
                family="exact-family",
                duplicate="exact-duplicate",
                sha256="exact-sha",
            ),
            _row(
                20,
                0,
                article_type="Tshirts",
                colour="Red",
                family="family-0",
                duplicate="duplicate-20",
                sha256="sha-20",
            ),
            _row(
                22,
                2,
                article_type="Jeans",
                colour="Black",
                family="family-2",
                duplicate="duplicate-22",
                sha256="sha-22",
            ),
            _row(
                23,
                3,
                article_type="Skirts",
                colour="Pink",
                family="family-3",
                duplicate="duplicate-23",
                sha256="sha-23",
            ),
            _row(
                24,
                4,
                article_type="Shorts",
                colour="Yellow",
                family="family-4",
                duplicate="duplicate-24",
                sha256="sha-24",
            ),
        ]
    )


def _small_primary_views() -> RetrievalViews:
    queries = pd.DataFrame(
        {
            "id": [1],
            "articleType": ["Tshirts"],
            "baseColour": ["Blue"],
            "sha256": ["sha-1"],
            "duplicate_group": ["duplicate-1"],
            "product_family_group": ["family-1"],
        }
    )
    gallery = pd.DataFrame(
        {
            "id": [2, 3, 4],
            "articleType": ["Tshirts", "Tshirts", "Jeans"],
            "baseColour": ["Blue", "Red", "Blue"],
            "sha256": ["sha-2", "sha-3", "sha-4"],
            "duplicate_group": ["duplicate-2", "duplicate-3", "duplicate-4"],
            "product_family_group": ["family-2", "family-3", "family-4"],
        }
    )
    return RetrievalViews(queries=queries, gallery=gallery)


def test_development_views_freeze_fold_one() -> None:
    primary, family = build_development_views(_split_frame())

    assert set(primary.queries["cv_fold"]) == {1}
    assert set(primary.gallery["cv_fold"]) == {0, 2, 3, 4}
    assert set(family.queries["cv_fold"]) == {1}
    assert set(family.gallery["cv_fold"]) == {1}


@pytest.mark.parametrize(
    "offending_key",
    ["id", "sha256", "duplicate_group", "product_family_group"],
)
def test_primary_isolation_names_offending_key(offending_key: str) -> None:
    query = pd.DataFrame(
        [
            {
                "id": 10,
                "sha256": "query-sha",
                "duplicate_group": "query-duplicate",
                "product_family_group": "query-family",
            }
        ]
    )
    gallery = pd.DataFrame(
        [
            {
                "id": 20,
                "sha256": "gallery-sha",
                "duplicate_group": "gallery-duplicate",
                "product_family_group": "gallery-family",
            }
        ]
    )
    gallery.loc[0, offending_key] = query.loc[0, offending_key]

    with pytest.raises(ValueError, match=offending_key):
        _assert_primary_isolation(query, gallery)


def test_primary_relevance_uses_linear_two_one_zero_grades() -> None:
    query = pd.Series({"articleType": "Tshirts", "baseColour": "Blue"})
    candidates = pd.DataFrame(
        {
            "articleType": ["Tshirts", "Tshirts", "Jeans"],
            "baseColour": ["Blue", "Red", "Blue"],
        }
    )

    assert primary_relevance(query, candidates).tolist() == [2, 1, 0]


def test_family_mask_removes_self_sha_and_duplicate_group() -> None:
    query = pd.Series(
        {"id": 10, "sha256": "same", "duplicate_group": "dup", "product_family_group": "f"}
    )
    candidates = pd.DataFrame(
        {
            "id": [10, 11, 12, 13, 14],
            "sha256": ["same", "same", "other-12", "other-13", "other-14"],
            "duplicate_group": ["dup", "other-11", "dup", "other-13", "other-14"],
            "product_family_group": ["f", "f", "f", "f", "other-family"],
        }
    )

    assert family_candidate_mask(query, candidates).tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert family_relevance(query, candidates.loc[[3, 4]]).tolist() == [1, 0]


def test_coverage_reports_undefined_and_strict_positive_counts() -> None:
    primary, family = build_development_views(_coverage_split_frame())

    coverage = compute_relevance_coverage(primary, family, k_values=(1, 2))

    expected = pd.DataFrame(
        [
            {
                "protocol": "primary",
                "k": 1,
                "total_queries": 4,
                "scored_queries": 3,
                "excluded_queries": 1,
                "zero_strict_queries": 2,
                "fewer_than_k_strict_queries": 2,
            },
            {
                "protocol": "primary",
                "k": 2,
                "total_queries": 4,
                "scored_queries": 3,
                "excluded_queries": 1,
                "zero_strict_queries": 2,
                "fewer_than_k_strict_queries": 4,
            },
            {
                "protocol": "family",
                "k": 1,
                "total_queries": 4,
                "scored_queries": 2,
                "excluded_queries": 2,
                "zero_strict_queries": 2,
                "fewer_than_k_strict_queries": 2,
            },
            {
                "protocol": "family",
                "k": 2,
                "total_queries": 4,
                "scored_queries": 2,
                "excluded_queries": 2,
                "zero_strict_queries": 2,
                "fewer_than_k_strict_queries": 4,
            },
        ]
    )
    pd.testing.assert_frame_equal(coverage, expected)


@pytest.mark.parametrize(
    "k_values",
    [(True,), (1.0,), (0,), (-1,), (1, 1)],
)
def test_coverage_rejects_invalid_k_values(
    k_values: tuple[object, ...],
) -> None:
    primary, family = build_development_views(_coverage_split_frame())

    with pytest.raises(ValueError, match="positive unique integers"):
        compute_relevance_coverage(primary, family, k_values=k_values)


def test_prepare_rankings_collapses_variants_and_breaks_ties_by_id() -> None:
    views = _small_primary_views()
    raw = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "candidate_id": [4, 3, 3, 2],
            "distance": [0.2, 0.1, 0.3, 0.1],
        }
    )

    ranked = protocol.prepare_rankings(raw, views, protocol="primary", max_k=3)

    assert ranked["candidate_id"].tolist() == [2, 3, 4]
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked["distance"].tolist() == [0.1, 0.1, 0.2]


def test_prepare_rankings_sorts_numeric_string_distances() -> None:
    views = _small_primary_views()
    raw = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "candidate_id": [2, 3, 4],
            "distance": ["10", "2", "3.5"],
        }
    )

    ranked = protocol.prepare_rankings(raw, views, protocol="primary", max_k=3)

    assert ranked["candidate_id"].tolist() == [3, 4, 2]
    assert ranked["distance"].tolist() == [2.0, 3.5, 10.0]
    assert pd.api.types.is_float_dtype(ranked["distance"])


@pytest.mark.parametrize(
    "invalid_distance",
    ["not-a-number", np.nan, np.inf, -np.inf],
)
def test_prepare_rankings_rejects_non_finite_numeric_distances(
    invalid_distance: object,
) -> None:
    raw = pd.DataFrame(
        {
            "query_id": [1],
            "candidate_id": [2],
            "distance": [invalid_distance],
        }
    )

    with pytest.raises(ValueError, match="finite numeric distances"):
        protocol.prepare_rankings(raw, _small_primary_views(), "primary", max_k=1)


@pytest.mark.parametrize("max_k", [True, 1.0, 0, -1])
def test_prepare_rankings_rejects_invalid_max_k(max_k: object) -> None:
    raw = pd.DataFrame(
        {"query_id": [1], "candidate_id": [2], "distance": [0.1]}
    )

    with pytest.raises(ValueError, match="positive unique integers"):
        protocol.prepare_rankings(
            raw, _small_primary_views(), "primary", max_k=max_k
        )


def test_prepare_rankings_breaks_string_id_ties_numerically() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame({"id": ["1"]}),
        gallery=pd.DataFrame({"id": ["2", "10"]}),
    )
    raw = pd.DataFrame(
        {
            "query_id": ["1", "1"],
            "candidate_id": ["10", "2"],
            "distance": [0.1, 0.1],
        }
    )

    ranked = protocol.prepare_rankings(raw, views, protocol="primary", max_k=2)

    assert ranked["candidate_id"].tolist() == ["2", "10"]


@pytest.mark.parametrize("invalid_id", ["two", "2.5"])
def test_prepare_rankings_rejects_non_integer_compatible_ids(
    invalid_id: str,
) -> None:
    views = RetrievalViews(
        queries=pd.DataFrame({"id": ["1"]}),
        gallery=pd.DataFrame({"id": [invalid_id]}),
    )
    raw = pd.DataFrame(
        {
            "query_id": ["1"],
            "candidate_id": [invalid_id],
            "distance": [0.1],
        }
    )

    with pytest.raises(ValueError, match="integer-compatible"):
        protocol.prepare_rankings(raw, views, protocol="primary", max_k=1)


def test_prepare_rankings_filters_family_leakage_before_top_k() -> None:
    queries = pd.DataFrame(
        {
            "id": [1],
            "sha256": ["query-sha"],
            "duplicate_group": ["query-duplicate"],
            "product_family_group": ["family-a"],
        }
    )
    gallery = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "sha256": ["query-sha", "query-sha", "sha-3", "sha-4", "sha-5"],
            "duplicate_group": [
                "query-duplicate",
                "duplicate-2",
                "query-duplicate",
                "duplicate-4",
                "duplicate-5",
            ],
            "product_family_group": ["family-a"] * 5,
        }
    )
    raw = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1, 1],
            "candidate_id": [1, 2, 3, 4, 5],
            "distance": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    ranked = protocol.prepare_rankings(
        raw,
        RetrievalViews(queries=queries, gallery=gallery),
        protocol="family",
        max_k=2,
    )

    assert ranked["candidate_id"].tolist() == [4, 5]
    assert ranked["rank"].tolist() == [1, 2]


def test_prepare_rankings_filters_numeric_equivalent_self_id() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": ["01"],
                "sha256": ["query-sha"],
                "duplicate_group": ["query-duplicate"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [1, 2],
                "sha256": ["different-sha", "sha-2"],
                "duplicate_group": ["different-duplicate", "duplicate-2"],
                "product_family_group": ["family-a", "family-a"],
            }
        ),
    )
    raw = pd.DataFrame(
        {
            "query_id": ["01", "01"],
            "candidate_id": [1, 2],
            "distance": [0.1, 0.2],
        }
    )

    ranked = protocol.prepare_rankings(raw, views, protocol="family", max_k=1)

    assert ranked["candidate_id"].tolist() == [2]


def test_prepare_rankings_rejects_family_query_with_no_eligible_results() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1],
                "sha256": ["query-sha"],
                "duplicate_group": ["query-duplicate"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [2],
                "sha256": ["query-sha"],
                "duplicate_group": ["duplicate-2"],
                "product_family_group": ["family-a"],
            }
        ),
    )
    raw = pd.DataFrame(
        {"query_id": [1], "candidate_id": [2], "distance": [0.1]}
    )

    with pytest.raises(ValueError, match="fewer than max_k"):
        protocol.prepare_rankings(raw, views, protocol="family", max_k=1)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            pd.DataFrame({"query_id": [1], "candidate_id": [2]}),
            "missing columns",
        ),
        (
            pd.DataFrame(
                {"query_id": [99], "candidate_id": [2], "distance": [0.1]}
            ),
            "unknown query",
        ),
        (
            pd.DataFrame(
                {"query_id": [1], "candidate_id": [99], "distance": [0.1]}
            ),
            "unknown candidate",
        ),
    ],
)
def test_prepare_rankings_rejects_invalid_ranking_rows(
    raw: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        protocol.prepare_rankings(raw, _small_primary_views(), "primary", max_k=1)


@pytest.mark.parametrize("side", ["queries", "gallery"])
def test_prepare_rankings_rejects_duplicate_product_ids(side: str) -> None:
    views = _small_primary_views()
    duplicated = getattr(views, side)
    duplicated = pd.concat([duplicated, duplicated.iloc[[0]]], ignore_index=True)
    invalid_views = RetrievalViews(
        queries=duplicated if side == "queries" else views.queries,
        gallery=duplicated if side == "gallery" else views.gallery,
    )
    raw = pd.DataFrame(
        {"query_id": [1], "candidate_id": [2], "distance": [0.1]}
    )

    with pytest.raises(ValueError, match="duplicate product IDs"):
        protocol.prepare_rankings(raw, invalid_views, "primary", max_k=1)


def test_prepare_rankings_requires_max_k_results_for_every_query() -> None:
    raw = pd.DataFrame(
        {"query_id": [1], "candidate_id": [2], "distance": [0.1]}
    )

    with pytest.raises(ValueError, match="fewer than max_k"):
        protocol.prepare_rankings(raw, _small_primary_views(), "primary", max_k=2)


@pytest.mark.parametrize(
    "ranks",
    [
        [1, 1, 3],
        [0, 1, 2],
        [1, 2, 4],
        [1, 2, 3.5],
    ],
)
def test_metric_evaluators_reject_invalid_ranks(ranks: list[float]) -> None:
    rankings = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "candidate_id": [2, 3, 4],
            "distance": [0.1, 0.2, 0.3],
            "rank": ranks,
        }
    )

    with pytest.raises(
        ValueError, match="unique consecutive one-based integer ranks"
    ):
        protocol.evaluate_primary_rankings(rankings, _small_primary_views(), (3,))


def test_metric_evaluators_reject_duplicate_candidate_products() -> None:
    rankings = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "candidate_id": [2, 2, 4],
            "distance": [0.1, 0.2, 0.3],
            "rank": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="duplicate product IDs"):
        protocol.evaluate_primary_rankings(rankings, _small_primary_views(), (3,))


@pytest.mark.parametrize(
    ("query_ids", "message"),
    [
        ([1], "missing query IDs.*5"),
        ([1, 5, 99], "unknown query IDs.*99"),
    ],
)
def test_metric_evaluators_require_exactly_every_expected_query(
    query_ids: list[int],
    message: str,
) -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1, 5],
                "articleType": ["Tshirts", "Tshirts"],
                "baseColour": ["Blue", "Blue"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [2],
                "articleType": ["Tshirts"],
                "baseColour": ["Blue"],
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": query_ids,
            "candidate_id": [2] * len(query_ids),
            "distance": [0.1] * len(query_ids),
            "rank": [1] * len(query_ids),
        }
    )

    with pytest.raises(ValueError, match=message):
        protocol.evaluate_primary_rankings(rankings, views, (1,))


@pytest.mark.parametrize("k_values", [(0,), (-1,), (1, 1), (1.5,)])
def test_primary_metric_rejects_invalid_k_values(
    k_values: tuple[object, ...],
) -> None:
    rankings = pd.DataFrame(
        {
            "query_id": [1],
            "candidate_id": [2],
            "distance": [0.1],
            "rank": [1],
        }
    )

    with pytest.raises(ValueError, match="positive unique integers"):
        protocol.evaluate_primary_rankings(
            rankings, _small_primary_views(), k_values=k_values
        )


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_family_metric_rejects_invalid_k_values(k: object) -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1],
                "sha256": ["sha-1"],
                "duplicate_group": ["duplicate-1"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [2],
                "sha256": ["sha-2"],
                "duplicate_group": ["duplicate-2"],
                "product_family_group": ["family-a"],
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": [1],
            "candidate_id": [2],
            "distance": [0.1],
            "rank": [1],
        }
    )

    with pytest.raises(ValueError, match="positive unique integers"):
        protocol.evaluate_family_rankings(rankings, views, k=k)


def test_primary_metric_uses_linear_gain_and_reports_undefined_coverage() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "articleType": ["Tshirts", "Tshirts", "Hats", "Jeans"],
                "baseColour": ["Blue", "Blue", "Black", "Blue"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [10, 11, 12],
                "articleType": ["Tshirts", "Tshirts", "Jeans"],
                "baseColour": ["Blue", "Red", "Blue"],
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "candidate_id": [
                10,
                11,
                12,
                12,
                11,
                10,
                10,
                11,
                12,
                12,
                10,
                11,
            ],
            "distance": [0.1, 0.2, 0.3] * 4,
            "rank": [1, 2, 3] * 4,
        }
    )

    per_query, summary = protocol.evaluate_primary_rankings(
        rankings, views, k_values=(3,)
    )

    expected = (1 / np.log2(3) + 2 / np.log2(4)) / (
        2 / np.log2(2) + 1 / np.log2(3)
    )
    assert per_query.loc[per_query["query_id"].eq(1), "ndcg_at_3"].item() == 1.0
    assert per_query.loc[
        per_query["query_id"].eq(2), "ndcg_at_3"
    ].item() == pytest.approx(expected)
    assert per_query.loc[
        per_query["query_id"].eq(1), "precision_any_at_3"
    ].item() == pytest.approx(2 / 3)
    assert per_query.loc[
        per_query["query_id"].eq(1), "precision_strict_at_3"
    ].item() == pytest.approx(1 / 3)
    assert per_query.loc[per_query["query_id"].eq(3), "ndcg_at_3"].isna().item()
    assert list(summary.columns) == [
        "metric",
        "k",
        "aggregation",
        "value",
        "query_count",
        "class_count",
    ]
    ndcg_query_mean = summary.query(
        "metric == 'ndcg' and aggregation == 'query_mean'"
    ).iloc[0]
    ndcg_macro = summary.query(
        "metric == 'ndcg' and aggregation == 'article_type_macro'"
    ).iloc[0]
    assert ndcg_query_mean["value"] == pytest.approx((2 + expected) / 3)
    assert ndcg_query_mean["query_count"] == 3
    assert ndcg_macro["value"] == pytest.approx((3 + expected) / 4)
    assert ndcg_macro["value"] != pytest.approx(ndcg_query_mean["value"])
    assert ndcg_macro["class_count"] == 2


def test_primary_ndcg_snaps_only_tiny_upper_bound_roundoff() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1],
                "articleType": ["Nail Polish"],
                "baseColour": ["Orange"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": list(range(10, 23)),
                "articleType": ["Nail Polish"] * 13,
                "baseColour": ["Blue"] * 13,
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": [1] * 13,
            "candidate_id": list(range(10, 23)),
            "distance": [float(index) for index in range(13)],
            "rank": list(range(1, 14)),
        }
    )

    per_query, summary = protocol.evaluate_primary_rankings(
        rankings,
        views,
        k_values=(20,),
    )

    assert per_query["ndcg_at_20"].item() == 1.0
    assert summary.loc[summary["metric"].eq("ndcg"), "value"].max() == 1.0


@pytest.mark.parametrize(
    ("candidate_id", "reason"),
    [(1, "self"), (2, "same SHA"), (3, "same duplicate group")],
)
def test_family_metric_rejects_ineligible_ranked_candidates(
    candidate_id: int, reason: str
) -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1],
                "sha256": ["query-sha"],
                "duplicate_group": ["query-duplicate"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "sha256": ["query-sha", "query-sha", "sha-3", "sha-4"],
                "duplicate_group": [
                    "query-duplicate",
                    "duplicate-2",
                    "query-duplicate",
                    "duplicate-4",
                ],
                "product_family_group": ["family-a"] * 4,
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": [1],
            "candidate_id": [candidate_id],
            "distance": [0.1],
            "rank": [1],
        }
    )

    with pytest.raises(ValueError, match="ineligible Protocol B candidate"):
        protocol.evaluate_family_rankings(rankings, views, k=1)


def test_family_metric_rejects_numeric_equivalent_self_id() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": ["01"],
                "sha256": ["query-sha"],
                "duplicate_group": ["query-duplicate"],
                "product_family_group": ["family-a"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [1, 2],
                "sha256": ["different-sha", "sha-2"],
                "duplicate_group": ["different-duplicate", "duplicate-2"],
                "product_family_group": ["family-a", "family-a"],
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": ["01"],
            "candidate_id": [1],
            "distance": [0.1],
            "rank": [1],
        }
    )

    with pytest.raises(ValueError, match="ineligible Protocol B candidate"):
        protocol.evaluate_family_rankings(rankings, views, k=1)


def test_family_metric_scores_eligible_relatives_and_counts_undefined() -> None:
    views = RetrievalViews(
        queries=pd.DataFrame(
            {
                "id": [1, 5, 9],
                "sha256": ["sha-1", "sha-5", "shared-sha"],
                "duplicate_group": ["duplicate-1", "duplicate-5", "duplicate-9"],
                "product_family_group": ["family-a", "family-b", "family-c"],
            }
        ),
        gallery=pd.DataFrame(
            {
                "id": [2, 3, 4, 6, 7, 8, 10],
                "sha256": [
                    "sha-2",
                    "sha-3",
                    "sha-4",
                    "sha-6",
                    "sha-7",
                    "sha-8",
                    "shared-sha",
                ],
                "duplicate_group": [
                    "duplicate-2",
                    "duplicate-3",
                    "duplicate-4",
                    "duplicate-6",
                    "duplicate-7",
                    "duplicate-8",
                    "duplicate-10",
                ],
                "product_family_group": [
                    "family-a",
                    "family-a",
                    "other",
                    "family-b",
                    "family-b",
                    "other",
                    "family-c",
                ],
            }
        ),
    )
    rankings = pd.DataFrame(
        {
            "query_id": [1, 1, 5, 5, 9, 9],
            "candidate_id": [2, 4, 4, 8, 4, 8],
            "distance": [0.1, 0.1, 0.1, 0.2, 0.1, 0.2],
            "rank": [1, 2, 1, 2, 1, 2],
        }
    )

    per_query, summary = protocol.evaluate_family_rankings(rankings, views, k=2)

    assert per_query["recall_at_2"].tolist()[:2] == [0.5, 0.0]
    assert per_query["hit_rate_at_2"].tolist()[:2] == [1.0, 0.0]
    assert per_query["precision_at_2"].tolist()[:2] == [0.5, 0.0]
    assert per_query.loc[per_query["query_id"].eq(9), "recall_at_2"].isna().item()
    assert set(summary["aggregation"]) == {"query_mean"}
    recall = summary.loc[summary["metric"].eq("recall")].iloc[0]
    coverage = summary.loc[summary["metric"].eq("coverage")].iloc[0]
    tie_rate = summary.loc[summary["metric"].eq("tie_rate")].iloc[0]
    assert recall["value"] == pytest.approx(0.25)
    assert recall["query_count"] == 2
    assert coverage["value"] == pytest.approx(2 / 3)
    assert coverage["query_count"] == 3
    assert tie_rate["value"] == pytest.approx(1 / 3)
