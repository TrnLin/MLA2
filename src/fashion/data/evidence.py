"""Development-only evidence helpers for Notebook 01.

The helpers calculate tables. Plotting stays visible in the notebook so Run All
still tells the complete data-preparation story.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import CV_FOLD_COUNT
from fashion.data.perceptual import NEAR_DUPLICATE_RULE


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def family_profile(
    splits: pd.DataFrame,
    *,
    partition: str = "development",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return conservative split-group sizes, one profile, and overlapping source counts."""
    required = {
        "id",
        "partition",
        "cv_fold",
        "product_family_group",
        "family_group_basis",
    }
    if missing := required.difference(splits.columns):
        raise ValueError(f"family profile is missing columns: {sorted(missing)}")
    rows = splits[splits["partition"].eq(partition)].copy()
    if rows.empty:
        raise ValueError(f"family profile partition {partition!r} is empty")
    family_sizes = (
        rows.groupby("product_family_group", as_index=False)
        .agg(
            family_size=("id", "size"),
            family_group_basis=("family_group_basis", "first"),
        )
        .sort_values(["family_size", "product_family_group"], ascending=[False, True])
    )
    multirow = family_sizes["family_size"].gt(1)
    products_in_multirow = int(family_sizes.loc[multirow, "family_size"].sum())
    partition_crossings = int(
        splits.loc[~splits["partition"].eq("quarantine")]
        .groupby("product_family_group")["partition"]
        .nunique()
        .gt(1)
        .sum()
    )
    fold_crossings = int(rows.groupby("product_family_group")["cv_fold"].nunique().gt(1).sum())
    profile = pd.DataFrame(
        [
            {
                "partition": partition,
                "product_rows": int(len(rows)),
                "conservative_split_groups": int(len(family_sizes)),
                "singleton_family_percent": float(family_sizes["family_size"].eq(1).mean() * 100),
                "products_in_multirow_family_percent": float(
                    products_in_multirow / len(rows) * 100
                ),
                "largest_family_products": int(family_sizes["family_size"].max()),
                "active_partition_crossings": partition_crossings,
                "development_fold_crossings": fold_crossings,
            }
        ]
    )
    source_tokens = {
        "same normalized product name": "normalized_product_name",
        "same SHA-256": "exact_sha256",
        "accepted near-duplicate": "accepted_near_duplicate",
    }
    source_rows = []
    basis = family_sizes["family_group_basis"].astype(str)
    for label, token in source_tokens.items():
        mask = multirow & basis.str.contains(token, regex=False)
        source_rows.append(
            {
                "family_source": label,
                "multirow_families": int(mask.sum()),
                "products": int(family_sizes.loc[mask, "family_size"].sum()),
                "counting_note": "sources overlap when more than one rule joins a family",
            }
        )
    return family_sizes.reset_index(drop=True), profile, pd.DataFrame(source_rows)


def fold_support_tables(
    class_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build scatter points plus validation-count and untrainable heatmap matrices."""
    folds = range(CV_FOLD_COUNT)
    required = {
        "target",
        "class",
        "development_product_count",
        "development_family_count",
        "rare_warning",
        "untrainable_fold_count",
        *(f"fold_{fold}_validation_products" for fold in folds),
        *(f"fold_{fold}_training_products" for fold in folds),
    }
    if missing := required.difference(class_summary.columns):
        raise ValueError(f"fold-support table is missing columns: {sorted(missing)}")
    points = class_summary[
        [
            "target",
            "class",
            "development_product_count",
            "development_family_count",
            "rare_warning",
            "untrainable_fold_count",
        ]
    ].copy()
    warning = class_summary["rare_warning"].astype(str).ne("")
    rare = class_summary.loc[warning].copy()
    rare["row_label"] = rare["target"].astype(str) + ": " + rare["class"].astype(str)
    validation = rare.set_index("row_label")[[f"fold_{fold}_validation_products" for fold in folds]]
    validation.columns = [f"fold {fold}" for fold in folds]
    training = rare.set_index("row_label")[[f"fold_{fold}_training_products" for fold in folds]]
    training.columns = validation.columns
    untrainable = training.apply(pd.to_numeric, errors="raise").eq(0)
    return points, validation.astype(int), untrainable


def _stable_mode(series: pd.Series) -> str:
    counts = series.astype(str).value_counts()
    maximum = int(counts.max())
    return sorted(counts[counts.eq(maximum)].index.astype(str))[0]


def shortcut_benchmarks(
    development: pd.DataFrame,
    *,
    grouping_target: str = "articleType",
    predicted_targets: Sequence[str] = ("season", "usage", "gender"),
) -> pd.DataFrame:
    """Compare global-majority and in-sample group-majority descriptive guesses."""
    output: list[dict[str, Any]] = []
    for target in predicted_targets:
        valid = _as_bool(development[f"has_{grouping_target}_label"]) & _as_bool(
            development[f"has_{target}_label"]
        )
        rows = development.loc[valid, [grouping_target, target]].astype(str).copy()
        if rows.empty:
            raise ValueError(f"no complete rows for {grouping_target} and {target}")
        global_label = _stable_mode(rows[target])
        group_modes = rows.groupby(grouping_target)[target].agg(_stable_mode)
        group_prediction = rows[grouping_target].map(group_modes)
        global_accuracy = float(rows[target].eq(global_label).mean())
        group_accuracy = float(group_prediction.eq(rows[target]).mean())
        output.append(
            {
                "grouping_target": grouping_target,
                "predicted_target": target,
                "complete_development_rows": int(len(rows)),
                "global_majority_label": global_label,
                "global_majority_accuracy": global_accuracy,
                "group_majority_accuracy": group_accuracy,
                "association_lift_percentage_points": (group_accuracy - global_accuracy) * 100,
                "interpretation": (
                    "descriptive association/shortcut risk; not model accuracy or causality"
                ),
            }
        )
    return pd.DataFrame(output)


def article_target_heatmap(
    development: pd.DataFrame,
    target: str,
    *,
    top_article_types: int = 30,
) -> pd.DataFrame:
    """Return row-normalized target shares for the most common article types."""
    valid = _as_bool(development["has_articleType_label"]) & _as_bool(
        development[f"has_{target}_label"]
    )
    rows = development.loc[valid, ["articleType", target]].astype(str)
    support = rows["articleType"].value_counts()
    selected = support.head(top_article_types).index
    table = pd.crosstab(rows["articleType"], rows[target], normalize="index")
    return table.reindex(selected).fillna(0.0)


def near_threshold_review(
    candidates: pd.DataFrame,
    *,
    accepted_pairs: int = 4,
    rejected_pairs: int = 4,
) -> pd.DataFrame:
    """Select automatic non-exact pairs nearest the frozen rule boundary.

    This is evidence only. The returned rows do not feed decisions back into the
    candidate table, family builder, split, or fold assignment.
    """
    required = {
        "id_1",
        "id_2",
        "is_exact_sha256",
        "meets_automatic_rule",
        "accepted_near_duplicate",
        "dhash_distance",
        "ahash_distance",
        "mse",
        "mae",
        "crop_mse",
        "crop_mae",
        "foreground_ratio",
    }
    if missing := required.difference(candidates.columns):
        raise ValueError(f"near-threshold review is missing columns: {sorted(missing)}")
    rows = candidates.loc[~_as_bool(candidates["is_exact_sha256"])].copy()
    if rows.empty:
        raise ValueError("near-threshold review has no non-exact candidate pairs")
    maximum_metrics = {
        "dhash_distance": "maximum_dhash_distance",
        "ahash_distance": "maximum_ahash_distance",
        "mse": "maximum_canvas_mse",
        "mae": "maximum_canvas_mae",
        "crop_mse": "maximum_crop_mse",
        "crop_mae": "maximum_crop_mae",
    }
    margins = []
    for column, rule_name in maximum_metrics.items():
        threshold = float(NEAR_DUPLICATE_RULE[rule_name])
        scale = threshold if threshold else 1.0
        margins.append((threshold - pd.to_numeric(rows[column])) / scale)
    foreground_threshold = float(NEAR_DUPLICATE_RULE["minimum_foreground_ratio"])
    margins.append(
        (pd.to_numeric(rows["foreground_ratio"]) - foreground_threshold) / foreground_threshold
    )
    rows["signed_rule_margin"] = pd.concat(margins, axis=1).min(axis=1)
    rows["boundary_distance"] = rows["signed_rule_margin"].abs()
    rows["review_side"] = np.where(
        _as_bool(rows["meets_automatic_rule"]), "accepted side", "rejected side"
    )
    accepted = rows[_as_bool(rows["meets_automatic_rule"])].nsmallest(
        accepted_pairs, "boundary_distance"
    )
    rejected = rows[~_as_bool(rows["meets_automatic_rule"])].nsmallest(
        rejected_pairs, "boundary_distance"
    )
    selected = pd.concat([accepted, rejected], ignore_index=True)
    selected["pipeline_effect"] = "none; evidence-only sample"
    return selected.sort_values(["review_side", "boundary_distance", "id_1", "id_2"])
