"""Training summaries and development-only validation diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fashion.config import TARGET_COLUMNS
from fashion.eda.scope import bool_mask


def build_target_validation_diagnostic(
    splits: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    """Compare Train/Validation using Train-selected classes and local denominators."""
    mask_column = f"has_{target}_label"
    if target not in splits or mask_column not in splits:
        raise ValueError(f"missing target or validity mask for {target}")
    counts: dict[str, pd.Series] = {}
    denominators: dict[str, int] = {}
    for partition in ("train", "val"):
        rows = splits[splits["partition"].eq(partition)]
        valid = rows[bool_mask(rows[mask_column])]
        if valid.empty:
            raise ValueError(f"no valid {partition} labels for {target}")
        denominators[partition] = len(valid)
        counts[partition] = valid[target].value_counts()

    training_classes = counts["train"].index.tolist()
    selected_counts = {
        partition: counts[partition].reindex(training_classes, fill_value=0).astype(int)
        for partition in ("train", "val")
    }
    proportions = {
        partition: selected_counts[partition] / denominators[partition] * 100
        for partition in ("train", "val")
    }
    class_union = counts["train"].index.union(counts["val"].index, sort=False)
    union_proportions = {
        partition: counts[partition].reindex(class_union, fill_value=0)
        / denominators[partition]
        * 100
        for partition in ("train", "val")
    }
    gaps = (proportions["train"] - proportions["val"]).abs()
    union_gaps = (union_proportions["train"] - union_proportions["val"]).abs()
    absent = selected_counts["val"][selected_counts["val"].eq(0)].index.tolist()
    validation_classes = set(counts["val"].index)
    training_class_set = set(training_classes)
    return {
        "classes": [str(label) for label in training_classes],
        "valid_label_denominators": denominators,
        "class_counts": {
            partition: {
                str(label): int(value) for label, value in selected_counts[partition].items()
            }
            for partition in ("train", "val")
        },
        "proportions_percent": {
            partition: {str(label): float(value) for label, value in proportions[partition].items()}
            for partition in ("train", "val")
        },
        "training_class_count": len(training_classes),
        "validation_class_count": len(validation_classes),
        "training_classes_observed_in_validation": int(selected_counts["val"].gt(0).sum()),
        "training_class_coverage_percent": float(selected_counts["val"].gt(0).mean() * 100),
        "training_classes_absent_from_validation": [str(label) for label in absent],
        "validation_classes_not_in_training_count": len(
            validation_classes.difference(training_class_set)
        ),
        "validation_labels_outside_training_classes": int(
            counts["val"].loc[~counts["val"].index.isin(training_classes)].sum()
        ),
        "absolute_percentage_point_gaps": {
            str(label): float(value) for label, value in gaps.items()
        },
        "distribution_gap_summary": {
            "basis": (
                "complete Train/Validation class union with within-partition "
                "valid-label denominators"
            ),
            "max_absolute_percentage_point_gap": float(union_gaps.max()),
            "mean_absolute_percentage_point_gap": float(union_gaps.mean()),
            "median_absolute_percentage_point_gap": float(union_gaps.median()),
            "total_variation_percentage_points": float(union_gaps.sum() / 2),
        },
    }


def build_validation_diagnostics(splits: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build safe development diagnostics for every supervised target."""
    return {target: build_target_validation_diagnostic(splits, target) for target in TARGET_COLUMNS}


def build_split_balance_inputs(
    splits: pd.DataFrame,
    target: str = "articleType",
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a Train-selected table of within-partition percentages."""
    diagnostic = build_target_validation_diagnostic(splits, target)
    classes = diagnostic["classes"][:top_n]
    frame = pd.DataFrame(
        {
            partition: {
                label: diagnostic["proportions_percent"][partition][label] for label in classes
            }
            for partition in ("train", "val")
        },
        index=classes,
    )
    frame.index.name = target
    return frame, diagnostic["valid_label_denominators"]


def summarize_target(train: pd.DataFrame, target: str) -> dict[str, Any]:
    """Summarize one target strictly inside the training partition."""
    valid = bool_mask(train[f"has_{target}_label"])
    counts = train.loc[valid, target].value_counts()
    return {
        "source_partition": "train",
        "valid_labels": int(valid.sum()),
        "missing_labels": int((~valid).sum()),
        "num_classes": len(counts),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "top_5": {str(label): int(count) for label, count in counts.head(5).items()},
        "bottom_5": {str(label): int(count) for label, count in counts.tail(5).items()},
        "singleton_classes": [str(label) for label in counts[counts.eq(1)].index],
        "rare_classes_lt10": [str(label) for label in counts[counts.lt(10)].index],
    }
