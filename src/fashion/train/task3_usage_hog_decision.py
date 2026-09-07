"""The actual frozen Route A/B rules for the Usage U2 screen."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fashion.train.task3_decisions import (
    DECISION_BOOTSTRAP_REPETITIONS,
    check,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    probability_columns,
    robustness_changes,
    validate_oof,
)

SUPPORTED_USAGE_CLASSES = (
    "Casual",
    "Ethnic",
    "Formal",
    "NA",
    "Party",
    "Smart Casual",
    "Sports",
    "Travel",
)
RARE_USAGE_CLASSES = ("NA", "Party", "Smart Casual", "Travel")
U2_GATE_VERSION = "u2_routes_natural_calibration_v2"


def evaluate_usage_u2(
    predictions: pd.DataFrame,
    parent_predictions: pd.DataFrame | None,
    *,
    expected: pd.DataFrame,
    classes: Sequence[str],
    run_ids_by_fold: Mapping[int, str],
    parent_run_ids_by_fold: Mapping[int, str],
    fold_metrics: Mapping[int, Mapping[str, Any]],
    candidate_robustness: pd.DataFrame | None,
    parent_robustness: pd.DataFrame | None,
    artifact_integrity: bool | None,
    parent_clean_gaps: Mapping[int, float] | None = None,
    bootstrap_repetitions: int = DECISION_BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Keep Home in the official score and fail closed on missing safety evidence."""
    if set(classes) != {*SUPPORTED_USAGE_CLASSES, "Home"}:
        raise ValueError("U2 requires the fixed nine Usage classes, including literal NA")
    if set(expected["cv_fold"].astype(int)) != {0, 4}:
        raise ValueError("U2 decisions require the complete folds 0 and 4")
    child = validate_oof(
        predictions, expected, target="usage", classes=classes, run_ids_by_fold=run_ids_by_fold
    )
    metrics = oof_metrics(child, classes)
    checks = [check("canonical_oof_integrity", len(child), "exact canonical folds 0 and 4", True)]
    home_zero = bool(child[probability_columns(classes)[list(classes).index("Home")]].eq(0).all())
    checks.append(check("unsupported_home_zero", home_zero, "P(Home)=0 on every row", home_zero))
    checks.append(check("ece_15", metrics["ece_15"], "<= 0.050", metrics["ece_15"] <= 0.050))
    caps = []
    for scope, rows in [
        ("pooled", child),
        *[(f"fold_{f}", child[child.cv_fold.eq(f)]) for f in (0, 4)],
    ]:
        for label in RARE_USAGE_CLASSES:
            support = int(rows["true_label"].eq(label).sum())
            count = int(rows["predicted_label"].eq(label).sum())
            caps.append(
                {
                    "scope": scope,
                    "class_name": label,
                    "support": support,
                    "predicted_count": count,
                    "maximum": 5 * support,
                    "pass": count <= 5 * support,
                }
            )
    checks.append(
        check(
            "rare_prediction_cap",
            caps,
            "each rare class <= 5x support, pooled and per fold",
            all(c["pass"] for c in caps),
        )
    )
    resource_rows = []
    for fold in (0, 4):
        saved = fold_metrics.get(fold, {})
        seconds = saved.get("fold_wall_seconds")
        memory = saved.get("peak_memory_bytes")
        valid = (
            seconds is not None
            and memory is not None
            and np.isfinite(float(seconds))
            and np.isfinite(float(memory))
            and 0 <= float(seconds) <= 90 * 60
            and 0 <= float(memory) <= 7 * 1024**3
        )
        resource_rows.append(
            {"fold": fold, "seconds": seconds, "peak_memory_bytes": memory, "pass": bool(valid)}
        )
    checks.append(
        check(
            "runtime_and_memory",
            resource_rows,
            "<= 90 min/fold and <= 7 GiB",
            all(r["pass"] for r in resource_rows),
        )
    )
    checks.append(
        check(
            "registry_and_artifact_integrity",
            artifact_integrity,
            "all source and child checks pass",
            artifact_integrity,
        )
    )
    parent_metrics = interval = class_changes = None
    route_a = route_b = None
    gap_improvement = probability_improvement = None
    gap_status = "unavailable_parent_clean_training_not_measured"
    if parent_predictions is not None:
        parent = validate_oof(
            parent_predictions,
            expected,
            target="usage",
            classes=classes,
            run_ids_by_fold=parent_run_ids_by_fold,
            allow_legacy_na=True,
        )
        parent_metrics = oof_metrics(parent, classes)
        interval = paired_family_bootstrap(
            child, parent, classes=classes, repetitions=bootstrap_repetitions
        )
        route_a = metrics["macro_f1"] >= 0.417319 and interval["lower_95"] > 0
        improvements = {
            key: (parent_metrics[key] - metrics[key]) / parent_metrics[key]
            if parent_metrics[key] > 0
            else None
            for key in ("nll", "brier")
        }
        probability_improvement = improvements
        probability_route = any(v is not None and v >= 0.10 for v in improvements.values())
        gap_route = False
        if parent_clean_gaps is not None:
            # Callers must supply clean finished-checkpoint measurements, never epoch curves.
            child_gaps = [
                fold_metrics.get(f, {}).get("final_train_validation_macro_f1_gap") for f in (0, 4)
            ]
            if set(parent_clean_gaps) == {0, 4} and all(
                v is not None and np.isfinite(v) for v in child_gaps
            ):
                parent_gap = float(np.mean([parent_clean_gaps[f] for f in (0, 4)]))
                if np.isfinite(parent_gap) and parent_gap > 0:
                    gap_improvement = float((parent_gap - np.mean(child_gaps)) / parent_gap)
                    gap_route = gap_improvement >= 0.25
                    gap_status = (
                        "evaluated_clean_scores; child_ensemble_has_mixed_fit_calibration_exposure"
                    )
        route_b = metrics["macro_f1"] >= 0.402319 and (gap_route or probability_route)
        child_classes = {r["class_name"]: r for r in metrics["per_class"]}
        parent_classes = {r["class_name"]: r for r in parent_metrics["per_class"]}
        class_changes = [
            {
                "class_name": name,
                "candidate_f1": child_classes[name]["f1"],
                "parent_f1": parent_classes[name]["f1"],
                "delta": child_classes[name]["f1"] - parent_classes[name]["f1"],
            }
            for name in SUPPORTED_USAGE_CLASSES
        ]
        checks.append(
            check(
                "supported_class_no_harm",
                class_changes,
                "all eight non-Home class deltas >= -0.030",
                all(r["delta"] >= -0.030 for r in class_changes),
            )
        )
        if candidate_robustness is not None and parent_robustness is not None:
            child_clean = {
                f: oof_metrics(child[child.cv_fold.eq(f)], classes)["macro_f1"] for f in (0, 4)
            }
            parent_clean = {
                f: oof_metrics(parent[parent.cv_fold.eq(f)], classes)["macro_f1"] for f in (0, 4)
            }
            child_delta = robustness_changes(
                candidate_robustness, clean_by_fold=child_clean, run_ids_by_fold=run_ids_by_fold
            )
            parent_delta = robustness_changes(
                parent_robustness,
                clean_by_fold=parent_clean,
                run_ids_by_fold=parent_run_ids_by_fold,
            )
            for name, delta in (child_delta - parent_delta).items():
                checks.append(
                    check(
                        "robustness." + name,
                        float(delta),
                        "mean induced change vs E2 >= -0.020",
                        delta >= -0.020,
                    )
                )
        else:
            checks.append(
                check("robustness", None, "all five matched corruption checks required", None)
            )
    else:
        checks.extend(
            [
                check("supported_class_no_harm", None, "matched E2 required", None),
                check("robustness", None, "matched E2 required", None),
            ]
        )
    route_pass = None if route_a is None else bool(route_a or route_b)
    checks.append(
        check(
            "route_a_or_b",
            {"route_a": route_a, "route_b": route_b},
            "at least one frozen route passes",
            route_pass,
        )
    )
    home_rows = child[child["true_label"].eq("Home")]
    home_nll = float(-np.log(1e-12) * len(home_rows) / len(child)) if home_zero else None
    return {
        "status": decision(checks),
        "gate_version": U2_GATE_VERSION,
        "checks": checks,
        "candidate_metrics": metrics,
        "matched_parent_metrics": parent_metrics,
        "paired_family_bootstrap": interval,
        "route_a": route_a,
        "route_b": route_b,
        "probability_relative_improvements": probability_improvement,
        "clean_gap_relative_improvement": gap_improvement,
        "clean_gap_status": gap_status,
        "supported_classes": list(SUPPORTED_USAGE_CLASSES),
        "rare_prediction_counts": caps,
        "official_metric": "fixed_nine_class_macro_f1_including_Home",
        "macro_f1_without_home": float(
            np.mean([r["f1"] for r in metrics["per_class"] if r["class_name"] != "Home"])
        ),
        "home_nll_contribution_to_official_mean": home_nll,
        "nll_epsilon": 1e-12,
        "common_four_macro_f1": float(
            np.mean(
                [
                    r["f1"]
                    for r in metrics["per_class"]
                    if r["class_name"] in {"Casual", "Ethnic", "Formal", "Sports"}
                ]
            )
        ),
        "rare_four_macro_f1": float(
            np.mean(
                [r["f1"] for r in metrics["per_class"] if r["class_name"] in RARE_USAGE_CLASSES]
            )
        ),
    }
