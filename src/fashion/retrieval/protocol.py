"""Fold-safe views and relevance coverage for Task 4 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from fashion.data.dataset import get_cv_split

FIXED_VALIDATION_FOLD: int = 1
K_VALUES: tuple[int, ...] = (5, 10, 20)
COVERAGE_COLUMNS: tuple[str, ...] = (
    "protocol",
    "k",
    "total_queries",
    "scored_queries",
    "excluded_queries",
    "zero_strict_queries",
    "fewer_than_k_strict_queries",
)


@dataclass(frozen=True)
class RetrievalViews:
    """Query and gallery rows for one retrieval protocol."""

    queries: pd.DataFrame
    gallery: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    if missing := required.difference(frame.columns):
        raise ValueError(f"retrieval data is missing columns: {sorted(missing)}")


def _non_empty_values(series: pd.Series) -> set[object]:
    present = series.notna() & series.astype(str).str.strip().ne("")
    return set(series.loc[present])


def _integer_id_values(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.mod(1).eq(0)
    if not valid.all():
        raise ValueError(f"{label} must contain integer-compatible product IDs")
    return numeric.map(int)


def _canonical_id_map(frame: pd.DataFrame, label: str) -> dict[int, object]:
    numeric_ids = _integer_id_values(frame["id"], label)
    if numeric_ids.duplicated().any():
        raise ValueError(f"{label} contain duplicate product IDs")
    return dict(zip(numeric_ids, frame["id"], strict=True))


def _validate_k_values(k_values: tuple[object, ...]) -> tuple[int, ...]:
    if (
        not k_values
        or any(isinstance(k, bool) or not isinstance(k, Integral) or k <= 0 for k in k_values)
        or len(set(k_values)) != len(k_values)
    ):
        raise ValueError("K values must be positive unique integers")
    return tuple(int(k) for k in k_values)


def _assert_primary_isolation(queries: pd.DataFrame, gallery: pd.DataFrame) -> None:
    for key in ("id", "sha256", "duplicate_group", "product_family_group"):
        overlap = _non_empty_values(queries[key]) & _non_empty_values(gallery[key])
        if overlap:
            raise ValueError(f"Protocol A query and gallery share {key}")


def build_development_views(
    splits: pd.DataFrame,
    validation_fold: int = FIXED_VALIDATION_FOLD,
) -> tuple[RetrievalViews, RetrievalViews]:
    """Build fixed-fold primary and within-fold family retrieval views."""
    gallery, queries = get_cv_split(splits, validation_fold)
    _require_columns(
        queries,
        {
            "id",
            "sha256",
            "duplicate_group",
            "product_family_group",
            "articleType",
            "baseColour",
        },
    )
    _assert_primary_isolation(queries, gallery)
    return RetrievalViews(queries.copy(), gallery.copy()), RetrievalViews(
        queries.copy(), queries.copy()
    )


def primary_relevance(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """Grade candidates by article type, then matching colour."""
    same_type = candidates["articleType"].eq(query["articleType"]).to_numpy()
    same_colour = candidates["baseColour"].eq(query["baseColour"]).to_numpy()
    return same_type.astype(np.int8) + (same_type & same_colour).astype(np.int8)


def family_candidate_mask(query: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    """Keep candidates after removing self and exact duplicates."""
    query_numeric_id = _integer_id_values(
        pd.Series([query["id"]]), "query id"
    ).iloc[0]
    candidate_numeric_ids = _integer_id_values(candidates["id"], "candidate id")
    return (
        candidate_numeric_ids.ne(query_numeric_id)
        & candidates["sha256"].ne(query["sha256"])
        & candidates["duplicate_group"].ne(query["duplicate_group"])
    )


def family_relevance(query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """Mark candidates in the query's product family as relevant."""
    return candidates["product_family_group"].eq(query["product_family_group"]).to_numpy(
        dtype=np.int8
    )


def _coverage_rows(
    protocol: str,
    positive_count: pd.Series,
    strict_count: pd.Series,
    k_values: tuple[int, ...],
) -> list[dict[str, object]]:
    total_queries = len(positive_count)
    excluded_queries = int(positive_count.eq(0).sum())
    return [
        {
            "protocol": protocol,
            "k": k,
            "total_queries": total_queries,
            "scored_queries": total_queries - excluded_queries,
            "excluded_queries": excluded_queries,
            "zero_strict_queries": int(strict_count.eq(0).sum()),
            "fewer_than_k_strict_queries": int(strict_count.lt(k).sum()),
        }
        for k in k_values
    ]


def compute_relevance_coverage(
    primary: RetrievalViews,
    family: RetrievalViews,
    k_values: tuple[int, ...] = K_VALUES,
) -> pd.DataFrame:
    """Count scorable queries and available strict positives by protocol."""
    k_values = _validate_k_values(k_values)
    primary_queries = primary.queries
    primary_gallery = primary.gallery
    type_counts = primary_gallery.groupby("articleType", dropna=False).size()
    primary_positive_count = (
        primary_queries["articleType"].map(type_counts).fillna(0).astype(int)
    )
    strict_counts = primary_gallery.groupby(
        ["articleType", "baseColour"], dropna=False
    ).size()
    strict_keys = pd.Series(
        list(
            primary_queries[["articleType", "baseColour"]].itertuples(
                index=False, name=None
            )
        ),
        index=primary_queries.index,
    )
    primary_strict_count = strict_keys.map(strict_counts).fillna(0).astype(int)

    family_queries = family.queries
    family_size = family_queries.groupby("product_family_group")["id"].transform("size")
    duplicate_size = family_queries.groupby(
        ["product_family_group", "duplicate_group"], dropna=False
    )["id"].transform("size")
    sha_size = family_queries.groupby(
        ["product_family_group", "sha256"], dropna=False
    )["id"].transform("size")
    intersection_size = family_queries.groupby(
        ["product_family_group", "duplicate_group", "sha256"], dropna=False
    )["id"].transform("size")
    family_positive_count = family_size - duplicate_size - sha_size + intersection_size

    rows = _coverage_rows(
        "primary", primary_positive_count, primary_strict_count, k_values
    )
    rows.extend(
        _coverage_rows("family", family_positive_count, family_positive_count, k_values)
    )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def prepare_rankings(
    rankings: pd.DataFrame,
    views: RetrievalViews,
    protocol: str,
    max_k: int = 20,
) -> pd.DataFrame:
    """Collapse image variants and prepare deterministic product rankings."""
    required = {"query_id", "candidate_id", "distance"}
    if missing := required.difference(rankings.columns):
        raise ValueError(f"rankings are missing columns: {sorted(missing)}")
    if protocol not in {"primary", "family"}:
        raise ValueError("protocol must be 'primary' or 'family'")
    (max_k,) = _validate_k_values((max_k,))

    for name, frame in (("queries", views.queries), ("gallery", views.gallery)):
        _require_columns(frame, {"id"})
    query_ids = _canonical_id_map(views.queries, "queries")
    candidate_ids = _canonical_id_map(views.gallery, "gallery")
    ranking_query_ids = _integer_id_values(rankings["query_id"], "query_id")
    ranking_candidate_ids = _integer_id_values(
        rankings["candidate_id"], "candidate_id"
    )
    unknown_queries = set(ranking_query_ids).difference(query_ids)
    if unknown_queries:
        raise ValueError(f"rankings contain unknown query IDs: {sorted(unknown_queries)}")
    unknown_candidates = set(ranking_candidate_ids).difference(candidate_ids)
    if unknown_candidates:
        raise ValueError(
            f"rankings contain unknown candidate IDs: {sorted(unknown_candidates)}"
        )
    numeric_distances = pd.to_numeric(rankings["distance"], errors="coerce")
    if numeric_distances.isna().any() or not np.isfinite(numeric_distances).all():
        raise ValueError("rankings must contain finite numeric distances")

    prepared = rankings.loc[:, ["query_id", "candidate_id", "distance"]].copy()
    prepared["distance"] = numeric_distances.astype(float)
    prepared["_query_numeric_id"] = ranking_query_ids
    prepared["_candidate_numeric_id"] = ranking_candidate_ids
    prepared["query_id"] = prepared["_query_numeric_id"].map(query_ids)
    prepared["candidate_id"] = prepared["_candidate_numeric_id"].map(candidate_ids)
    supplied_query_ids = pd.Index(
        ranking_query_ids.drop_duplicates().map(query_ids)
    )
    prepared = prepared.sort_values(
        ["_query_numeric_id", "distance", "_candidate_numeric_id"], kind="mergesort"
    )
    prepared = prepared.drop_duplicates(
        ["_query_numeric_id", "_candidate_numeric_id"], keep="first"
    )

    if protocol == "family":
        family_columns = {
            "id",
            "sha256",
            "duplicate_group",
            "product_family_group",
        }
        _require_columns(views.queries, family_columns)
        _require_columns(views.gallery, family_columns)
        queries_by_id = views.queries.set_index("id", drop=False)
        gallery_by_id = views.gallery.set_index("id", drop=False)
        keep = pd.Series(False, index=prepared.index)
        for query_id, query_rows in prepared.groupby("query_id", sort=False):
            candidate_rows = gallery_by_id.loc[query_rows["candidate_id"]]
            keep.loc[query_rows.index] = family_candidate_mask(
                queries_by_id.loc[query_id], candidate_rows
            ).to_numpy()
        prepared = prepared.loc[keep]

    prepared["rank"] = prepared.groupby("query_id", sort=False).cumcount() + 1
    prepared = prepared.loc[prepared["rank"].le(max_k)].copy()

    result_counts = prepared.groupby("query_id").size().reindex(supplied_query_ids, fill_value=0)
    if result_counts.lt(max_k).any():
        short_queries = result_counts.index[result_counts.lt(max_k)].tolist()
        raise ValueError(f"queries have fewer than max_k eligible results: {short_queries}")

    return prepared.drop(
        columns=["_query_numeric_id", "_candidate_numeric_id"]
    ).reset_index(drop=True)


def _metric_lookups(
    rankings: pd.DataFrame, views: RetrievalViews
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(rankings, {"query_id", "candidate_id", "distance", "rank"})
    _require_columns(views.queries, {"id"})
    _require_columns(views.gallery, {"id"})
    query_ids = _canonical_id_map(views.queries, "queries")
    candidate_ids = _canonical_id_map(views.gallery, "gallery")
    ranking_query_ids = _integer_id_values(rankings["query_id"], "query_id")
    ranking_candidate_ids = _integer_id_values(
        rankings["candidate_id"], "candidate_id"
    )
    normalized_ids = pd.DataFrame(
        {"query_id": ranking_query_ids, "candidate_id": ranking_candidate_ids}
    )
    if normalized_ids.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("rankings contain duplicate product IDs within a query")

    queries = views.queries.set_index("id", drop=False)
    gallery = views.gallery.set_index("id", drop=False)
    unknown_queries = set(ranking_query_ids).difference(query_ids)
    unknown_candidates = set(ranking_candidate_ids).difference(candidate_ids)
    if unknown_queries:
        raise ValueError(f"rankings contain unknown query IDs: {sorted(unknown_queries)}")
    if unknown_candidates:
        raise ValueError(
            f"rankings contain unknown candidate IDs: {sorted(unknown_candidates)}"
        )
    rank_values = pd.to_numeric(rankings["rank"], errors="coerce")
    integer_ranks = (
        rank_values.notna()
        & np.isfinite(rank_values)
        & rank_values.mod(1).eq(0)
        & rank_values.gt(0)
    )
    valid_ranks = integer_ranks.all()
    if valid_ranks:
        for _, query_ranks in rank_values.groupby(ranking_query_ids, sort=False):
            expected = np.arange(1, len(query_ranks) + 1)
            if not np.array_equal(np.sort(query_ranks.to_numpy()), expected):
                valid_ranks = False
                break
    if not valid_ranks:
        raise ValueError(
            "rankings must have unique consecutive one-based integer ranks per query"
        )

    ordered = rankings.copy()
    ordered["query_id"] = ranking_query_ids.map(query_ids)
    ordered["candidate_id"] = ranking_candidate_ids.map(candidate_ids)
    ordered["rank"] = rank_values.map(int)
    ordered = ordered.sort_values(["query_id", "rank"], kind="mergesort")
    return ordered, queries, gallery


def _tie_rate(query_rankings: pd.DataFrame, k: int) -> float:
    distances = query_rankings.loc[query_rankings["rank"].le(k), "distance"]
    return float(distances.duplicated(keep=False).mean())


def _append_summary_rows(
    rows: list[dict[str, object]],
    per_query: pd.DataFrame,
    metric: str,
    k: int,
    value_column: str,
) -> None:
    scorable = per_query.dropna(subset=[value_column])
    class_means = scorable.groupby("articleType", dropna=False)[value_column].mean()
    common = {
        "metric": metric,
        "k": k,
        "query_count": len(scorable),
        "class_count": len(class_means),
    }
    rows.append(
        {
            **common,
            "aggregation": "query_mean",
            "value": float(scorable[value_column].mean()),
        }
    )
    rows.append(
        {
            **common,
            "aggregation": "article_type_macro",
            "value": float(class_means.mean()),
        }
    )


def evaluate_primary_rankings(
    rankings: pd.DataFrame,
    views: RetrievalViews,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Protocol A with linear-gain nDCG and two precision definitions."""
    k_values = _validate_k_values(k_values)
    _require_columns(views.queries, {"id", "articleType", "baseColour"})
    _require_columns(views.gallery, {"id", "articleType", "baseColour"})
    ordered, queries, gallery = _metric_lookups(rankings, views)

    query_rows: list[dict[str, object]] = []
    for query_id, query in queries.iterrows():
        query_rankings = ordered.loc[ordered["query_id"].eq(query_id)]
        candidates = gallery.loc[query_rankings["candidate_id"]]
        observed_grades = primary_relevance(query, candidates)
        gallery_grades = primary_relevance(query, gallery)
        strict_count = int((gallery_grades == 2).sum())
        broad_count = int((gallery_grades > 0).sum())
        result: dict[str, object] = {
            "query_id": query_id,
            "articleType": query["articleType"],
        }
        for k in k_values:
            discounts = np.log2(np.arange(2, k + 2))
            dcg = float((observed_grades[:k] / discounts[: len(observed_grades[:k])]).sum())
            ideal_grades = np.array(
                [2] * strict_count + [1] * (broad_count - strict_count)
            )
            ideal_at_k = ideal_grades[:k]
            idcg = float((ideal_at_k / discounts[: len(ideal_at_k)]).sum())
            if idcg > 0:
                result[f"ndcg_at_{k}"] = dcg / idcg
                result[f"precision_any_at_{k}"] = float(
                    (observed_grades[:k] > 0).sum() / k
                )
                result[f"precision_strict_at_{k}"] = float(
                    (observed_grades[:k] == 2).sum() / k
                )
            else:
                result[f"ndcg_at_{k}"] = np.nan
                result[f"precision_any_at_{k}"] = np.nan
                result[f"precision_strict_at_{k}"] = np.nan
            result[f"tie_rate_at_{k}"] = _tie_rate(query_rankings, k)
        query_rows.append(result)

    per_query = pd.DataFrame(query_rows)
    summary_rows: list[dict[str, object]] = []
    for k in k_values:
        for metric, column in (
            ("ndcg", f"ndcg_at_{k}"),
            ("precision_any", f"precision_any_at_{k}"),
            ("precision_strict", f"precision_strict_at_{k}"),
            ("tie_rate", f"tie_rate_at_{k}"),
        ):
            _append_summary_rows(summary_rows, per_query, metric, k, column)
    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "k", "aggregation", "value", "query_count", "class_count"],
    )
    return per_query, summary


def evaluate_family_rankings(
    rankings: pd.DataFrame,
    views: RetrievalViews,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Protocol B family recall, hit rate, precision, and coverage."""
    (k,) = _validate_k_values((k,))
    family_columns = {
        "id",
        "sha256",
        "duplicate_group",
        "product_family_group",
    }
    _require_columns(views.queries, family_columns)
    _require_columns(views.gallery, family_columns)
    ordered, queries, gallery = _metric_lookups(rankings, views)
    for query_id, query_rankings in ordered.groupby("query_id", sort=False):
        candidates = gallery.loc[query_rankings["candidate_id"]]
        if not family_candidate_mask(queries.loc[query_id], candidates).all():
            raise ValueError(
                f"rankings contain an ineligible Protocol B candidate for query {query_id}"
            )

    query_rows: list[dict[str, object]] = []
    for query_id, query in queries.iterrows():
        query_rankings = ordered.loc[ordered["query_id"].eq(query_id)]
        top_k = query_rankings.loc[query_rankings["rank"].le(k)]
        candidates = gallery.loc[top_k["candidate_id"]]
        eligible_gallery = gallery.loc[family_candidate_mask(query, gallery)]
        eligible_positive_count = int(
            family_relevance(query, eligible_gallery).sum()
        )
        relevant_retrieved = int(family_relevance(query, candidates).sum())
        result: dict[str, object] = {
            "query_id": query_id,
            f"tie_rate_at_{k}": _tie_rate(query_rankings, k),
        }
        if eligible_positive_count:
            result[f"recall_at_{k}"] = relevant_retrieved / eligible_positive_count
            result[f"hit_rate_at_{k}"] = float(relevant_retrieved > 0)
            result[f"precision_at_{k}"] = relevant_retrieved / k
        else:
            result[f"recall_at_{k}"] = np.nan
            result[f"hit_rate_at_{k}"] = np.nan
            result[f"precision_at_{k}"] = np.nan
        query_rows.append(result)

    per_query = pd.DataFrame(query_rows)
    scored = per_query[f"recall_at_{k}"].notna()
    total_queries = len(per_query)
    scored_queries = int(scored.sum())
    summary_rows: list[dict[str, object]] = [
        {
            "metric": "coverage",
            "k": k,
            "aggregation": "query_mean",
            "value": scored_queries / total_queries if total_queries else np.nan,
            "query_count": total_queries,
            "class_count": pd.NA,
        },
        {
            "metric": "tie_rate",
            "k": k,
            "aggregation": "query_mean",
            "value": float(per_query[f"tie_rate_at_{k}"].mean()),
            "query_count": total_queries,
            "class_count": pd.NA,
        },
    ]
    for metric in ("recall", "hit_rate", "precision"):
        values = per_query.loc[scored, f"{metric}_at_{k}"]
        summary_rows.append(
            {
                "metric": metric,
                "k": k,
                "aggregation": "query_mean",
                "value": float(values.mean()),
                "query_count": scored_queries,
                "class_count": pd.NA,
            }
        )
    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "k", "aggregation", "value", "query_count", "class_count"],
    )
    return per_query, summary
