"""Leak-safe visual retrieval contracts."""

from fashion.retrieval.protocol import (
    build_development_retrieval_sets,
    build_development_retrieval_variant_sets,
    build_final_application_gallery,
    rank_products_from_variants,
    recall_at_k,
    remove_self_match,
)

__all__ = [
    "build_development_retrieval_sets",
    "build_development_retrieval_variant_sets",
    "build_final_application_gallery",
    "rank_products_from_variants",
    "recall_at_k",
    "remove_self_match",
]
