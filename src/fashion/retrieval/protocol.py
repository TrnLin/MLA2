"""One query/gallery protocol derived only from the canonical split."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import nan

import pandas as pd

from fashion.data.splits import validate_splits


def _valid_supported(frame: pd.DataFrame, target: str) -> pd.Series:
    column = f"has_{target}_supported_label"
    if column not in frame:
        raise ValueError(f"split is missing {column}")
    if pd.api.types.is_bool_dtype(frame[column]):
        return frame[column].fillna(False)
    return frame[column].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_development_retrieval_sets(
    splits: pd.DataFrame,
    relevance_target: str = "articleType",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use train-supported validation queries and a train-only gallery."""
    validate_splits(splits)
    gallery = splits[splits["partition"].eq("train")].copy()
    queries = splits[
        splits["partition"].eq("val") & _valid_supported(splits, relevance_target)
    ].copy()
    if gallery.empty or queries.empty:
        raise ValueError("retrieval development query or gallery set is empty")
    if set(gallery["id"].astype(int)) & set(queries["id"].astype(int)):
        raise ValueError("retrieval query IDs entered the development gallery")
    if set(gallery["sha256"].astype(str)) & set(queries["sha256"].astype(str)):
        raise ValueError("an exact query image entered the development gallery")
    if set(gallery["product_family_group"].astype(str)) & set(
        queries["product_family_group"].astype(str)
    ):
        raise ValueError("a query product family entered the development gallery")
    return queries.reset_index(drop=True), gallery.reset_index(drop=True)


def build_development_retrieval_variant_sets(
    splits: pd.DataFrame,
    variants: pd.DataFrame,
    relevance_target: str = "articleType",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return both image variants for each allowed query and gallery product."""
    queries, gallery = build_development_retrieval_sets(splits, relevance_target)
    required = {"id", "partition", "variant", "variant_key", "path"}
    if missing := required.difference(variants.columns):
        raise ValueError(f"image variant manifest is missing columns: {sorted(missing)}")
    if variants["variant_key"].duplicated().any():
        raise ValueError("image variant keys must be unique")

    query_variants = variants[variants["id"].isin(queries["id"])].copy()
    gallery_variants = variants[variants["id"].isin(gallery["id"])].copy()
    for role, products, rows, expected_partition in (
        ("query", queries, query_variants, "val"),
        ("gallery", gallery, gallery_variants, "train"),
    ):
        if not rows["partition"].eq(expected_partition).all():
            raise ValueError(f"{role} variants escaped the {expected_partition} partition")
        expected_ids = set(products["id"].astype(int))
        if set(rows["id"].astype(int)) != expected_ids:
            raise ValueError(f"{role} variants do not cover every product ID")
        pairs = rows.groupby("id")["variant"].agg(set)
        if not pairs.eq({"original", "high_resolution"}).all():
            raise ValueError(f"{role} products must each have both image variants")
    return (
        query_variants.sort_values(["id", "variant"], ignore_index=True),
        gallery_variants.sort_values(["id", "variant"], ignore_index=True),
    )


def remove_self_match(
    query_id: int,
    ranked_ids: Iterable[int],
    top_k: int | None = None,
) -> list[int]:
    """Remove the query and collapse same-product variants before applying Top-K."""
    seen = {int(query_id)}
    filtered = []
    for item_id in ranked_ids:
        product_id = int(item_id)
        if product_id in seen:
            continue
        seen.add(product_id)
        filtered.append(product_id)
    return filtered if top_k is None else filtered[:top_k]


def rank_products_from_variants(
    variant_scores: pd.DataFrame,
    *,
    top_k: int | None = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Rank each product once using its best-scoring image variant."""
    required = {"id", "variant", "score"}
    if missing := required.difference(variant_scores.columns):
        raise ValueError(f"variant scores are missing columns: {sorted(missing)}")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = variant_scores.copy()
    ranked["id"] = pd.to_numeric(ranked["id"], errors="raise").astype(int)
    ranked["score"] = pd.to_numeric(ranked["score"], errors="raise")
    ranked.sort_values(
        ["score", "id", "variant"],
        ascending=[not higher_is_better, True, True],
        kind="mergesort",
        inplace=True,
    )
    ranked = ranked.drop_duplicates("id", keep="first")
    if top_k is not None:
        ranked = ranked.head(top_k)
    return ranked.reset_index(drop=True)


def recall_at_k(
    ranked_grades: Sequence[int],
    *,
    k: int,
    total_relevant: int | None = None,
    complete_gallery_grades: Sequence[int] | None = None,
    relevant_grade: int = 2,
) -> float:
    """Recall of relevant products against an explicit full-gallery denominator."""
    if k < 1:
        raise ValueError("k must be positive")
    if (total_relevant is None) == (complete_gallery_grades is None):
        raise ValueError(
            "provide exactly one of total_relevant or complete_gallery_grades; "
            "ranked Top-K grades cannot define the recall denominator"
        )
    grades = [int(grade) for grade in ranked_grades]
    if complete_gallery_grades is not None:
        positive_count = sum(int(grade) >= relevant_grade for grade in complete_gallery_grades)
    else:
        if isinstance(total_relevant, bool) or int(total_relevant) != total_relevant:
            raise ValueError("total_relevant must be a non-negative integer")
        positive_count = int(total_relevant)
    if positive_count < 0:
        raise ValueError("total_relevant must be a non-negative integer")
    retrieved_positive_count = sum(grade >= relevant_grade for grade in grades[:k])
    if retrieved_positive_count > positive_count:
        raise ValueError("retrieved relevant products exceed the full-gallery denominator")
    if positive_count == 0:
        return nan
    return retrieved_positive_count / positive_count


def build_final_application_gallery(
    splits: pd.DataFrame,
    evaluation_complete: bool,
    permitted_partitions: Sequence[str] = ("train", "val", "holdout"),
) -> pd.DataFrame:
    """Build a larger catalogue only after the independent evaluation is locked."""
    validate_splits(splits)
    if not evaluation_complete:
        raise ValueError("final application gallery is locked until evaluation is complete")
    if "quarantine" in permitted_partitions:
        raise ValueError("quarantine can never enter a retrieval gallery")
    return splits[splits["partition"].isin(permitted_partitions)].copy().reset_index(drop=True)
