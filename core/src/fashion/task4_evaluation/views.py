"""Holdout query and gallery views for the frozen Task 4 retrieval protocols."""

from __future__ import annotations

import pandas as pd

from fashion.data.splits import validate_split_structure
from fashion.task4.protocol import RetrievalViews

DEVELOPMENT_ROWS_EXPECTED = 32_773
HOLDOUT_ROWS_EXPECTED = 5_778
QUARANTINE_ROWS_EXPECTED = 61

_REQUIRED_COLUMNS = frozenset({"articleType", "baseColour"})


def build_holdout_views(splits: pd.DataFrame) -> tuple[RetrievalViews, RetrievalViews]:
    """Build the holdout Protocol A and Protocol B views.

    Protocol A ranks every holdout product against every development product.
    Protocol B ranks the holdout against itself for same-family recovery, matching the
    self-gallery shape the frozen coverage counter assumes.
    """
    validate_split_structure(splits)
    if missing := sorted(_REQUIRED_COLUMNS.difference(splits.columns)):
        raise ValueError(f"split manifest is missing columns: {missing}")
    holdout = splits.loc[splits["partition"].eq("holdout")].copy()
    development = splits.loc[splits["partition"].eq("development")].copy()
    if holdout.empty or development.empty:
        raise ValueError("holdout views require both holdout and development rows")
    holdout = holdout.sort_values("id", kind="mergesort").reset_index(drop=True)
    development = development.sort_values("id", kind="mergesort").reset_index(drop=True)
    primary = RetrievalViews(holdout.copy(), development.copy())
    family = RetrievalViews(holdout.copy(), holdout.copy())
    return primary, family


__all__ = [
    "DEVELOPMENT_ROWS_EXPECTED",
    "HOLDOUT_ROWS_EXPECTED",
    "QUARANTINE_ROWS_EXPECTED",
    "build_holdout_views",
]
