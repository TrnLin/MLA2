"""Official output labels and independently supported development-metric slices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from fashion.config import MIN_SUPPORTED_GROUPS, TARGET_COLUMNS


def _bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def apply_deployment_taxonomy(
    frame: pd.DataFrame,
    group_column: str = "product_family_group",
    targets: Sequence[str] = TARGET_COLUMNS,
    minimum_groups: int = MIN_SUPPORTED_GROUPS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit output and metric classes on train, then apply the frozen mapping."""
    if minimum_groups < 3:
        raise ValueError("supported evaluation needs at least three independent groups")
    if group_column not in frame:
        raise ValueError(f"missing family group column: {group_column}")
    if "partition" not in frame:
        raise ValueError("taxonomy fitting requires allocated partitions")
    fit = frame[frame["partition"].eq("train")]
    if fit.empty:
        raise ValueError("taxonomy fitting requires training rows")
    result = frame.copy()
    policy: dict[str, Any] = {
        "schema_version": "3.0.0",
        "fit_partition": "train",
        "fit_group_column": group_column,
        "minimum_independent_family_groups": minimum_groups,
        "action": (
            "fit model/output classes and the primary comparative slice from training data; "
            "mask validation or protected labels unknown to the frozen training mapping"
        ),
        "semantic_merges": {},
        "targets": {},
    }
    for target in targets:
        raw_mask = _bool_mask(fit[f"has_{target}_label"])
        support = (
            fit.loc[raw_mask, [group_column, target]]
            .drop_duplicates()
            .groupby(target)[group_column]
            .nunique()
            .sort_index()
        )
        supported = set(support[support.ge(minimum_groups)].index.astype(str))
        all_labels = set(support.index.astype(str))
        deployed_column = f"{target}_deployed"
        deployed_mask = f"has_{target}_deployed_label"
        supported_column = f"{target}_supported"
        supported_mask = f"has_{target}_supported_label"
        original_valid = _bool_mask(result[f"has_{target}_label"])
        result[deployed_mask] = original_valid & result[target].astype(str).isin(all_labels)
        result[deployed_column] = result[target].where(result[deployed_mask], "")
        result[f"{target}_deployment_status"] = "missing_raw_label"
        result.loc[original_valid, f"{target}_deployment_status"] = "unknown_to_train"
        result.loc[result[deployed_mask], f"{target}_deployment_status"] = "official_output"
        result[supported_mask] = original_valid & result[target].astype(str).isin(supported)
        result[supported_column] = result[target].where(result[supported_mask], "")
        result[f"{target}_evaluation_status"] = "missing_raw_label"
        result.loc[original_valid, f"{target}_evaluation_status"] = "unknown_to_train"
        result.loc[result[deployed_mask], f"{target}_evaluation_status"] = (
            "official_but_insufficient_for_primary_metrics"
        )
        result.loc[result[supported_mask], f"{target}_evaluation_status"] = (
            "supported_primary_metric"
        )
        limited = [
            {"class": label, "independent_family_groups": int(support.loc[label])}
            for label in sorted(all_labels - supported)
        ]
        policy["targets"][target] = {
            "fit_partition": "train",
            "official_output_classes": sorted(all_labels),
            "official_output_class_count": len(all_labels),
            "supported_classes": sorted(supported),
            "supported_class_count": len(supported),
            "primary_metric_limited_classes": limited,
            "primary_metric_limited_class_count": len(limited),
            "raw_class_count": len(all_labels),
            "official_output_label_column": deployed_column,
            "official_output_validity_column": deployed_mask,
            "supported_metric_label_column": supported_column,
            "supported_metric_validity_column": supported_mask,
        }
    return result, policy


def validate_built_taxonomy_coverage(
    frame: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLUMNS,
    minimum_groups: int = MIN_SUPPORTED_GROUPS,
) -> None:
    """Require every supported class to have enough independent train families."""
    train = frame[frame["partition"].eq("train")]
    for target in targets:
        output_column = f"{target}_deployed"
        output_mask = f"has_{target}_deployed_label"
        label_column = f"{target}_supported"
        mask_column = f"has_{target}_supported_label"
        official = set(train.loc[_bool_mask(train[output_mask]), output_column].astype(str))
        supported = set(train.loc[_bool_mask(train[mask_column]), label_column].astype(str))
        if not supported.issubset(official):
            raise ValueError(f"supported {target} classes are not train-fitted output classes")
        family_support = (
            train.loc[_bool_mask(train[mask_column]), [label_column, "product_family_group"]]
            .drop_duplicates()
            .groupby(label_column)["product_family_group"]
            .nunique()
        )
        weak = family_support[family_support.lt(minimum_groups)]
        if not weak.empty:
            raise ValueError(
                f"supported {target} classes lack {minimum_groups} train families: {weak.to_dict()}"
            )
