"""Failure analysis and robustness evidence for the Task 4 baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fashion.task4.preprocessing_experiment import FeatureIndex, PairEvaluation
from fashion.task4.probe import rank_probe_embeddings
from fashion.task4.protocol import (
    RetrievalViews,
    evaluate_primary_rankings,
)

__all__ = (
    "CanvasStressEvaluation",
    "build_query_support",
    "evaluate_canvas_stress",
    "mark_failure_slices",
    "select_example_ids",
    "summarize_failure_slices",
)


@dataclass(frozen=True)
class CanvasStressEvaluation:
    """Aggregate and query-level evidence for deterministic canvas stress."""

    summary: pd.DataFrame
    per_query: pd.DataFrame
    rankings: dict[str, pd.DataFrame]


def build_query_support(
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
) -> pd.DataFrame:
    """Calculate frozen Protocol A and B support for every query."""
    queries = primary_views.queries.sort_values("id").reset_index(drop=True)
    type_counts = primary_views.gallery.groupby("articleType", dropna=False).size()
    strict_counts = primary_views.gallery.groupby(
        ["articleType", "baseColour"],
        dropna=False,
    ).size()
    strict_keys = pd.Series(
        list(queries[["articleType", "baseColour"]].itertuples(index=False, name=None))
    )

    family_queries = family_views.queries
    family_size = family_queries.groupby("product_family_group")["id"].transform("size")
    duplicate_size = family_queries.groupby(
        ["product_family_group", "duplicate_group"],
        dropna=False,
    )["id"].transform("size")
    sha_size = family_queries.groupby(
        ["product_family_group", "sha256"],
        dropna=False,
    )["id"].transform("size")
    intersection_size = family_queries.groupby(
        ["product_family_group", "duplicate_group", "sha256"],
        dropna=False,
    )["id"].transform("size")
    family_counts = pd.Series(
        (family_size - duplicate_size - sha_size + intersection_size).to_numpy(),
        index=family_queries["id"].astype(int),
    )

    return pd.DataFrame(
        {
            "query_id": queries["id"].astype(int),
            "mode": queries["mode"],
            "aspect_ratio": queries["aspect_ratio"].astype(float),
            "primary_positive_count": queries["articleType"]
            .map(type_counts)
            .fillna(0)
            .astype(int),
            "primary_strict_count": strict_keys.map(strict_counts).fillna(0).astype(int),
            "family_positive_count": queries["id"]
            .astype(int)
            .map(family_counts)
            .astype(int),
        }
    )


def mark_failure_slices(support: pd.DataFrame) -> pd.DataFrame:
    """Mark the frozen query-level failure-slice boundaries."""
    marked = support.copy()
    marked["grayscale"] = marked["mode"].eq("L")
    marked["rare_article_type"] = marked["primary_positive_count"].between(1, 9)
    marked["rare_type_colour"] = marked["primary_strict_count"].lt(10)
    marked["unusual_geometry"] = marked["aspect_ratio"].ne(0.75)
    marked["family_unavailable"] = marked["family_positive_count"].eq(0)
    marked["weak_family"] = marked["family_positive_count"].between(1, 4)
    return marked


def summarize_failure_slices(
    query_metrics: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize frozen failure slices with their score coverage."""
    joined = query_metrics.merge(membership, on="query_id", validate="many_to_one")
    slice_metrics = (
        ("grayscale", "primary", "ndcg_at_10", "ndcg"),
        ("rare_article_type", "primary", "ndcg_at_10", "ndcg"),
        ("rare_type_colour", "primary", "ndcg_at_10", "ndcg"),
        ("unusual_geometry", "primary", "ndcg_at_10", "ndcg"),
        ("family_unavailable", "family", "recall_at_10", "recall"),
        ("weak_family", "family", "recall_at_10", "recall"),
    )
    context = ["scope", "fold", "size", "query_source", "gallery_source"]
    rows: list[dict[str, object]] = []
    for context_values, direction_rows in joined.groupby(
        context,
        sort=True,
        dropna=False,
    ):
        common = dict(zip(context, context_values, strict=True))
        for slice_name, protocol, value_column, metric in slice_metrics:
            selected = direction_rows.loc[
                direction_rows["protocol"].eq(protocol) & direction_rows[slice_name]
            ]
            values = selected[value_column]
            rows.append(
                {
                    **common,
                    "protocol": protocol,
                    "slice": slice_name,
                    "metric": metric,
                    "k": 10,
                    "aggregation": "query_mean",
                    "value": float(values.mean()) if values.notna().any() else np.nan,
                    "total_queries": len(selected),
                    "scored_queries": int(values.notna().sum()),
                    "excluded_queries": int(values.isna().sum()),
                    "coverage": float(values.notna().mean()) if len(selected) else 0.0,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            *context,
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
        ],
    )


def _index_features_for_ids(index: FeatureIndex, ids: pd.Series) -> np.ndarray:
    numeric_ids = pd.to_numeric(ids, errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.mod(1).eq(0).all():
        raise ValueError("canvas retrieval IDs must be integer-compatible")
    positions = {
        int(product_id): position for position, product_id in enumerate(index.ids)
    }
    requested = numeric_ids.astype(int).tolist()
    missing = sorted(set(requested).difference(positions))
    if missing:
        raise ValueError(f"canvas feature index is missing retrieval IDs: {missing[:10]}")
    return index.features[[positions[product_id] for product_id in requested]]


def _top10_sets(rankings: pd.DataFrame) -> pd.Series:
    return (
        rankings.loc[rankings["rank"].le(10)]
        .groupby("query_id", sort=True)["candidate_id"]
        .agg(lambda values: frozenset(int(value) for value in values))
    )


def evaluate_canvas_stress(
    clean: PairEvaluation,
    canvas_indexes: Mapping[str, FeatureIndex],
    gallery_index: FeatureIndex,
    primary_views: RetrievalViews,
    *,
    fold: int = 1,
) -> CanvasStressEvaluation:
    """Evaluate clean, wide, and tall queries against one fixed gallery."""
    if set(canvas_indexes) != {"wide", "tall"}:
        raise ValueError("canvas stress requires exactly wide and tall indexes")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise ValueError("fold must be an integer")
    for orientation, index in canvas_indexes.items():
        if index.contract != gallery_index.contract:
            raise ValueError(f"{orientation} and gallery indexes must share one contract")
    query_sources = {index.source for index in canvas_indexes.values()}
    if len(query_sources) != 1:
        raise ValueError("wide and tall canvas indexes must share one source")
    query_source = next(iter(query_sources))
    size = f"{gallery_index.contract.width}x{gallery_index.contract.height}"

    clean_values = clean.primary_per_query.loc[
        :, ["query_id", "ndcg_at_10"]
    ].rename(columns={"ndcg_at_10": "clean_ndcg_at_10"})
    clean_values["query_id"] = pd.to_numeric(
        clean_values["query_id"], errors="raise"
    ).astype(int)
    if clean_values["query_id"].duplicated().any():
        raise ValueError("clean evaluation has duplicate primary query metrics")
    clean_sets = _top10_sets(clean.primary_rankings)
    expected_query_ids = set(
        pd.to_numeric(primary_views.queries["id"], errors="raise").astype(int)
    )
    if set(clean_values["query_id"]) != expected_query_ids or set(clean_sets.index) != (
        expected_query_ids
    ):
        raise ValueError("clean evaluation does not match the canvas query view")

    common = {
        "scope": "development",
        "fold": fold,
        "size": size,
        "query_source": query_source,
        "gallery_source": gallery_index.source,
    }
    per_query_frames: list[pd.DataFrame] = [
        clean_values.assign(
            **common,
            query_variant="clean",
            canvas_ndcg_at_10=clean_values["clean_ndcg_at_10"],
            ndcg_change_from_clean=0.0,
            top10_overlap=1.0,
        )
    ]
    rankings: dict[str, pd.DataFrame] = {
        "clean": clean.primary_rankings.copy(),
    }
    query_ids = pd.to_numeric(primary_views.queries["id"], errors="raise").astype(int)
    gallery_ids = pd.to_numeric(primary_views.gallery["id"], errors="raise").astype(int)
    gallery_features = _index_features_for_ids(gallery_index, primary_views.gallery["id"])

    for orientation in ("wide", "tall"):
        canvas_index = canvas_indexes[orientation]
        canvas_rankings = rank_probe_embeddings(
            query_ids=query_ids.to_numpy(dtype=np.int64),
            query_features=_index_features_for_ids(
                canvas_index,
                primary_views.queries["id"],
            ),
            gallery_ids=gallery_ids.to_numpy(dtype=np.int64),
            gallery_features=gallery_features,
            views=primary_views,
            protocol="primary",
            max_k=10,
        )
        canvas_per_query, _ = evaluate_primary_rankings(
            canvas_rankings,
            primary_views,
            k_values=(10,),
        )
        canvas_values = canvas_per_query.loc[
            :, ["query_id", "ndcg_at_10"]
        ].rename(columns={"ndcg_at_10": "canvas_ndcg_at_10"})
        canvas_values["query_id"] = pd.to_numeric(
            canvas_values["query_id"], errors="raise"
        ).astype(int)
        canvas_sets = _top10_sets(canvas_rankings)
        overlaps = pd.DataFrame(
            {
                "query_id": sorted(expected_query_ids),
                "top10_overlap": [
                    len(clean_sets.loc[query_id] & canvas_sets.loc[query_id]) / 10
                    for query_id in sorted(expected_query_ids)
                ],
            }
        )
        labelled = (
            clean_values.merge(canvas_values, on="query_id", validate="one_to_one")
            .merge(overlaps, on="query_id", validate="one_to_one")
            .assign(
                **common,
                query_variant=orientation,
            )
        )
        labelled["ndcg_change_from_clean"] = (
            labelled["canvas_ndcg_at_10"] - labelled["clean_ndcg_at_10"]
        )
        per_query_frames.append(labelled)
        rankings[orientation] = canvas_rankings

    per_query = pd.concat(per_query_frames, ignore_index=True)
    variant_order = pd.Categorical(
        per_query["query_variant"],
        categories=["clean", "wide", "tall"],
        ordered=True,
    )
    per_query = (
        per_query.assign(_variant_order=variant_order)
        .sort_values(["query_id", "_variant_order"], kind="mergesort")
        .drop(columns="_variant_order")
        .reset_index(drop=True)
    )
    per_query = per_query.loc[
        :,
        [
            *common,
            "query_variant",
            "query_id",
            "clean_ndcg_at_10",
            "canvas_ndcg_at_10",
            "ndcg_change_from_clean",
            "top10_overlap",
        ],
    ]

    summary = (
        per_query.groupby("query_variant", sort=False, observed=True)
        .agg(
            queries=("query_id", "size"),
            ndcg_at_10=("canvas_ndcg_at_10", "mean"),
            ndcg_change_from_clean=("ndcg_change_from_clean", "mean"),
            mean_top10_overlap=("top10_overlap", "mean"),
        )
        .reindex(["clean", "wide", "tall"])
        .reset_index()
    )
    for position, (column, value) in enumerate(common.items()):
        summary.insert(position, column, value)
    return CanvasStressEvaluation(
        summary=summary,
        per_query=per_query,
        rankings=rankings,
    )


def select_example_ids(
    query_metrics: pd.DataFrame,
    membership: pd.DataFrame,
    canvas_per_query: pd.DataFrame,
) -> dict[str, int]:
    """Select deterministic V1-to-V1 success, failure, and canvas examples."""
    slice_protocols = (
        ("grayscale", "primary", "ndcg_at_10"),
        ("rare_article_type", "primary", "ndcg_at_10"),
        ("rare_type_colour", "primary", "ndcg_at_10"),
        ("unusual_geometry", "primary", "ndcg_at_10"),
        ("family_unavailable", "family", "recall_at_10"),
        ("weak_family", "family", "recall_at_10"),
    )
    slice_columns = [slice_name for slice_name, _, _ in slice_protocols]
    joined = query_metrics.loc[
        query_metrics["query_source"].eq("v1")
        & query_metrics["gallery_source"].eq("v1")
    ].merge(
        membership.loc[:, ["query_id", *slice_columns]],
        on="query_id",
        validate="many_to_one",
    )
    joined["_numeric_id"] = pd.to_numeric(joined["query_id"], errors="raise").astype(int)

    selected: dict[str, int] = {}
    primary = joined.loc[joined["protocol"].eq("primary")].copy()
    normal = primary.loc[~primary[slice_columns].any(axis=1)].sort_values(
        ["ndcg_at_10", "_numeric_id"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    if not normal.empty:
        selected["normal_success"] = int(normal.iloc[0]["_numeric_id"])

    for slice_name, protocol, score_column in slice_protocols:
        candidates = joined.loc[
            joined["protocol"].eq(protocol) & joined[slice_name]
        ].sort_values(
            [score_column, "_numeric_id"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
        if not candidates.empty:
            selected[slice_name] = int(candidates.iloc[0]["_numeric_id"])

    canvas = canvas_per_query.copy()
    if "query_variant" in canvas:
        canvas = canvas.loc[~canvas["query_variant"].eq("clean")]
    canvas["_numeric_id"] = pd.to_numeric(canvas["query_id"], errors="raise").astype(int)
    canvas = canvas.dropna(subset=["ndcg_change_from_clean"]).sort_values(
        ["ndcg_change_from_clean", "_numeric_id"],
        ascending=[True, True],
        kind="mergesort",
    )
    if not canvas.empty:
        selected["canvas_failure"] = int(canvas.iloc[0]["_numeric_id"])
    return selected
