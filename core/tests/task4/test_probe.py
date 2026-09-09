from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fashion.task4 as task4
import fashion.task4.probe as probe
from fashion.task4.probe import (
    COLOUR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    extract_spatial_probe,
    rank_probe_embeddings,
    rank_single_embedding,
)
from fashion.task4.protocol import RetrievalViews


def test_probe_excludes_padding_from_colour_and_edges() -> None:
    first = np.full((8, 8, 3), 255, dtype=np.uint8)
    second = np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    first[mask] = (255, 0, 0)
    second[mask] = (255, 0, 0)

    first_feature = extract_spatial_probe(first, mask)
    second_feature = extract_spatial_probe(second, mask)

    assert np.array_equal(first_feature, second_feature)


def test_probe_does_not_create_edges_at_the_padding_boundary() -> None:
    pixels = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    pixels[mask] = (255, 0, 0)

    feature = extract_spatial_probe(pixels, mask)
    edge = feature[COLOUR_FEATURE_DIM : COLOUR_FEATURE_DIM + EDGE_FEATURE_DIM]

    assert np.count_nonzero(edge) == 0


def test_probe_keeps_spatial_colour_layout() -> None:
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    first[:4] = (255, 0, 0)
    first[4:] = (0, 0, 255)
    second = first[::-1].copy()
    mask = np.ones((8, 8), dtype=bool)

    assert not np.array_equal(
        extract_spatial_probe(first, mask),
        extract_spatial_probe(second, mask),
    )


def test_probe_blocks_have_equal_unit_weight() -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4] = (255, 0, 0)
    pixels[:, 4:] = (0, 0, 255)

    feature = extract_spatial_probe(pixels, np.ones((8, 8), dtype=bool))
    colour = feature[:COLOUR_FEATURE_DIM]
    edge = feature[COLOUR_FEATURE_DIM : COLOUR_FEATURE_DIM + EDGE_FEATURE_DIM]

    assert np.linalg.norm(colour) == pytest.approx(1 / np.sqrt(2))
    assert np.linalg.norm(edge) == pytest.approx(1 / np.sqrt(2))
    assert np.linalg.norm(feature) == pytest.approx(1.0)


def test_probe_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content"):
        extract_spatial_probe(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=bool),
        )


def _primary_views(query_ids: list[int], gallery_ids: list[int]) -> RetrievalViews:
    return RetrievalViews(
        queries=pd.DataFrame({"id": query_ids}),
        gallery=pd.DataFrame({"id": gallery_ids}),
    )


def test_ranker_uses_cosine_distance_and_numeric_id_ties() -> None:
    ranked = rank_probe_embeddings(
        query_ids=np.array([1]),
        query_features=np.array([[1.0, 0.0]], dtype=np.float32),
        gallery_ids=np.array([10, 2, 3]),
        gallery_features=np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        views=_primary_views([1], [10, 2, 3]),
        protocol="primary",
        max_k=2,
    )

    assert ranked["candidate_id"].tolist() == [2, 10]
    assert ranked["distance"].tolist() == [0.0, 0.0]


def test_generic_ranker_accepts_learned_embeddings_through_the_frozen_path() -> None:
    query_features = np.zeros((1, 128), dtype=np.float32)
    query_features[0, 0] = 1.0
    gallery_features = np.zeros((3, 128), dtype=np.float32)
    gallery_features[0, 0] = 1.0
    gallery_features[1, 1] = 1.0
    gallery_features[2, :2] = 2**-0.5

    ranked = probe.rank_embeddings(
        query_ids=np.array([1]),
        query_features=query_features,
        gallery_ids=np.array([10, 2, 3]),
        gallery_features=gallery_features,
        views=_primary_views([1], [10, 2, 3]),
        protocol="primary",
        max_k=3,
    )

    assert ranked["candidate_id"].tolist() == [10, 3, 2]
    assert ranked["distance"].tolist() == pytest.approx([0.0, 1 - 2**-0.5, 1.0])
    assert task4.rank_embeddings is probe.rank_embeddings


def test_single_ranker_uses_cosine_distance_and_numeric_id_ties() -> None:
    ranked = rank_single_embedding(
        query_feature=np.array([1.0, 0.0], dtype=np.float32),
        gallery_ids=np.array([30, 10, 20], dtype=np.int64),
        gallery_features=np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        max_k=3,
    )

    assert ranked.columns.tolist() == ["candidate_id", "distance", "rank"]
    assert ranked["candidate_id"].tolist() == [10, 30, 20]
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked["distance"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert task4.rank_single_embedding is probe.rank_single_embedding


@pytest.mark.parametrize("max_k", [0, 4])
def test_single_ranker_rejects_k_outside_gallery_range(max_k: int) -> None:
    with pytest.raises(ValueError, match="max_k"):
        rank_single_embedding(
            query_feature=np.array([1.0, 0.0], dtype=np.float32),
            gallery_ids=np.array([10, 20, 30], dtype=np.int64),
            gallery_features=np.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                dtype=np.float32,
            ),
            max_k=max_k,
        )


def test_single_ranker_rejects_wrong_query_shape() -> None:
    with pytest.raises(ValueError, match="query feature"):
        rank_single_embedding(
            query_feature=np.array([[1.0, 0.0]], dtype=np.float32),
            gallery_ids=np.array([10], dtype=np.int64),
            gallery_features=np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_single_ranker_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        rank_single_embedding(
            query_feature=np.array([1.0, 0.0], dtype=np.float32),
            gallery_ids=np.array([10], dtype=np.int64),
            gallery_features=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("query_feature", "gallery_features"),
    [
        (
            np.array([np.nan, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[np.inf, 0.0]], dtype=np.float32),
        ),
        (
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[0.0, 0.0]], dtype=np.float32),
        ),
        (
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[2.0, 0.0]], dtype=np.float32),
        ),
    ],
)
def test_single_ranker_rejects_invalid_feature_values(
    query_feature: np.ndarray,
    gallery_features: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        rank_single_embedding(
            query_feature=query_feature,
            gallery_ids=np.array([10], dtype=np.int64),
            gallery_features=gallery_features,
            max_k=1,
        )


def test_ranker_filters_family_candidates_before_top_k() -> None:
    queries = pd.DataFrame(
        {
            "id": [1],
            "sha256": ["query-sha"],
            "duplicate_group": ["query-duplicate"],
            "product_family_group": ["family"],
        }
    )
    gallery = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "sha256": ["other-self-sha", "query-sha", "sha-3", "sha-4"],
            "duplicate_group": [
                "other-self-duplicate",
                "duplicate-2",
                "query-duplicate",
                "duplicate-4",
            ],
            "product_family_group": ["family"] * 4,
        }
    )

    ranked = rank_probe_embeddings(
        query_ids=np.array([1]),
        query_features=np.array([[1.0, 0.0]], dtype=np.float32),
        gallery_ids=np.array([1, 2, 3, 4]),
        gallery_features=np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        views=RetrievalViews(queries=queries, gallery=gallery),
        protocol="family",
        max_k=1,
    )

    assert ranked["candidate_id"].tolist() == [4]


def test_ranker_rejects_misaligned_ids_and_features() -> None:
    with pytest.raises(ValueError, match="row"):
        rank_probe_embeddings(
            query_ids=np.array([1, 2]),
            query_features=np.array([[1.0, 0.0]], dtype=np.float32),
            gallery_ids=np.array([3]),
            gallery_features=np.array([[1.0, 0.0]], dtype=np.float32),
            views=_primary_views([1, 2], [3]),
            protocol="primary",
            max_k=1,
        )


def test_ranker_returns_max_k_rows_for_every_query() -> None:
    ranked = rank_probe_embeddings(
        query_ids=np.array([1, 2]),
        query_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        gallery_ids=np.array([3, 4, 5]),
        gallery_features=np.array(
            [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]],
            dtype=np.float32,
        ),
        views=_primary_views([1, 2], [3, 4, 5]),
        protocol="primary",
        max_k=2,
    )

    assert ranked.groupby("query_id").size().to_dict() == {1: 2, 2: 2}
