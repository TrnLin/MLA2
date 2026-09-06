from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.train.metrics import classification_metrics
from fashion.train.task3_clean_slate import _prediction_frame
from fashion.train.task3_decisions import (
    CORE_CORRUPTIONS,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    validate_oof,
)
from fashion.train.task3_usage_hog_decision import SUPPORTED_USAGE_CLASSES, evaluate_usage_u2


def usage_case():
    classes = sorted([*SUPPORTED_USAGE_CLASSES, "Home"])
    rows = []
    for fold in (0, 4):
        labels = [c for c in SUPPORTED_USAGE_CLASSES for _ in range(8)]
        if fold == 4:
            labels.append("Home")
        for i, label in enumerate(labels):
            rows.append(
                {
                    "id": len(rows),
                    "cv_fold": fold,
                    "usage": label,
                    "product_family_group": f"family-{fold}-{i // 2}",
                    "path": f"data/raw/teacher/train/images_train/{len(rows)}.jpg",
                    "partition": "development",
                }
            )
    expected = pd.DataFrame(rows)
    predictions = {}
    for kind in ("child", "parent"):
        frames = []
        for fold in (0, 4):
            frame = expected[expected.cv_fold.eq(fold)]
            p = np.full((len(frame), len(classes)), 0.02)
            for j, label in enumerate(frame.usage):
                predicted = label
                if kind == "child" and label == "Home":
                    predicted = "Casual"
                if kind == "parent" and j % 3 == 0:
                    predicted = "Formal" if label != "Formal" else "Casual"
                p[j, classes.index(predicted)] = 0.9 if kind == "child" else 0.7
            if kind == "child":
                p[:, classes.index("Home")] = 0
            p /= p.sum(axis=1, keepdims=True)
            frames.append(
                _prediction_frame(
                    frame, target="usage", classes=classes, probabilities=p, run_id=f"{kind}-{fold}"
                )
            )
        predictions[kind] = pd.concat(frames, ignore_index=True)
    # Calibrate top-label confidence to the measured correctness so the test
    # isolates the decision routes rather than the independent ECE guard.
    child = predictions["child"]
    for i in range(len(child)):
        predicted = int(child.loc[i, "predicted_index"])
        p = np.zeros(len(classes))
        supported_indices = [j for j, c in enumerate(classes) if c != "Home"]
        p[supported_indices] = 0.01 / 7
        p[predicted] = 0.99
        for j, name in enumerate(classes):
            child.loc[i, f"probability_{j}_{name}"] = p[j]
        child.loc[i, "confidence"] = 0.99
    robust = {}
    for kind in ("child", "parent"):
        robust[kind] = pd.DataFrame(
            [
                {
                    "run_id": f"{kind}-{f}",
                    "validation_fold": f,
                    "corruption": c,
                    "macro_f1": oof_metrics(
                        predictions[kind][predictions[kind].cv_fold.eq(f)], classes
                    )["macro_f1"]
                    - 0.01,
                    "macro_f1_change": -0.01,
                }
                for f in (0, 4)
                for c in CORE_CORRUPTIONS
            ]
        )
    kwargs = dict(
        expected=expected,
        classes=classes,
        run_ids_by_fold={f: f"child-{f}" for f in (0, 4)},
        parent_run_ids_by_fold={f: f"parent-{f}" for f in (0, 4)},
        fold_metrics={f: {"fold_wall_seconds": 1, "peak_memory_bytes": 100} for f in (0, 4)},
        candidate_robustness=robust["child"],
        parent_robustness=robust["parent"],
        artifact_integrity=True,
        bootstrap_repetitions=40,
    )
    return predictions["child"], predictions["parent"], kwargs


def test_prediction_validation_rejects_missing_rows_and_bad_probabilities():
    child, _, args = usage_case()

    def validate(rows):
        return validate_oof(rows, args["expected"], target="usage", classes=args["classes"])

    validate(child)
    with pytest.raises(ValueError, match="exactly cover"):
        validate(child.iloc[1:])
    corrupted = child.copy()
    corrupted.loc[0, "probability_0_Casual"] = -0.01
    with pytest.raises(ValueError, match="probabilities"):
        validate(corrupted)
    corrupted = child.copy()
    corrupted.loc[0, "cv_fold"] = 4
    with pytest.raises(ValueError, match="canonical split"):
        validate(corrupted)


def test_only_index_verified_literal_na_is_repaired():
    _, parent, args = usage_case()
    parent.loc[parent.true_label.eq("NA"), "true_label"] = ""
    checked = validate_oof(
        parent, args["expected"], target="usage", classes=args["classes"], allow_legacy_na=True
    )
    assert checked.attrs["legacy_na_label_repairs"] == 16
    assert checked.true_label.eq("NA").sum() == 16
    parent.loc[0, "true_label"] = ""
    with pytest.raises(ValueError, match="true_label"):
        validate_oof(
            parent, args["expected"], target="usage", classes=args["classes"], allow_legacy_na=True
        )


def test_family_bootstrap_matches_explicit_whole_family_resampling():
    classes = ["A", "B", "absent"]
    child = pd.DataFrame(
        {
            "id": range(6),
            "cv_fold": [0, 0, 0, 4, 4, 4],
            "product_family_group": ["a", "a", "b", "c", "d", "d"],
            "true_index": [0, 1, 1, 0, 1, 0],
            "predicted_index": [0, 1, 0, 0, 1, 1],
        }
    )
    parent = child.copy()
    parent["predicted_index"] = [1, 1, 0, 1, 0, 0]
    actual = paired_family_bootstrap(child, parent, classes=classes, repetitions=50, seed=23)
    rng = np.random.default_rng(23)
    values = []
    for _ in range(50):
        selected = []
        for fold in (0, 4):
            families = sorted(child.loc[child.cv_fold.eq(fold), "product_family_group"].unique())
            for j in rng.integers(0, len(families), size=len(families)):
                selected.extend(child.index[child.product_family_group.eq(families[j])])
        scores = [
            classification_metrics(
                frame.iloc[selected].true_index.to_numpy(),
                np.eye(3)[frame.iloc[selected].predicted_index],
                classes,
            )["macro_f1"]
            for frame in (child, parent)
        ]
        values.append(scores[0] - scores[1])
    assert actual["lower_95"] == pytest.approx(np.quantile(values, 0.025))
    assert actual["upper_95"] == pytest.approx(np.quantile(values, 0.975))


def test_u2_real_routes_keep_home_and_block_missing_evidence():
    child, parent, args = usage_case()
    passed = evaluate_usage_u2(child, parent, **args)
    assert passed["status"] == "pass"
    assert passed["route_a"] is True
    assert passed["clean_gap_status"].startswith("unavailable")
    assert passed["candidate_metrics"]["support"] == len(child)
    assert passed["home_nll_contribution_to_official_mean"] == pytest.approx(
        -np.log(1e-12) / len(child)
    )
    assert len(passed["supported_classes"]) == 8
    missing = evaluate_usage_u2(child, parent, **{**args, "artifact_integrity": None})
    assert missing["status"] == "not_evaluated"
    missing = evaluate_usage_u2(child, None, **args)
    assert missing["status"] == "not_evaluated"
    bad = args["candidate_robustness"].copy()
    mask = bad.corruption.eq("brightness_085")
    bad.loc[mask, ["macro_f1", "macro_f1_change"]] -= 0.03
    failed = evaluate_usage_u2(child, parent, **{**args, "candidate_robustness": bad})
    assert failed["status"] == "fail"
    assert any(
        c["gate"] == "robustness.brightness_085" and c["status"] == "fail" for c in failed["checks"]
    )


def test_incomplete_corruption_grid_cannot_pass():
    child, parent, args = usage_case()
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_usage_u2(
            child, parent, **{**args, "candidate_robustness": args["candidate_robustness"].iloc[1:]}
        )
    assert decision([]) == "not_evaluated"


def test_noninferiority_alone_does_not_pass_either_u2_route():
    child, _, args = usage_case()
    same_parent = child.copy()
    same_parent["run_id"] = same_parent.cv_fold.map(args["parent_run_ids_by_fold"])
    robust = args["candidate_robustness"].copy()
    robust["run_id"] = robust.validation_fold.map(args["parent_run_ids_by_fold"])
    result = evaluate_usage_u2(child, same_parent, **{**args, "parent_robustness": robust})
    assert result["route_a"] is False
    assert result["route_b"] is False
    assert result["status"] == "fail"


@pytest.mark.parametrize("seconds", [-1, float("nan"), 5401])
def test_u2_rejects_invalid_or_exceeded_resource_budget(seconds):
    child, parent, args = usage_case()
    args["fold_metrics"][0]["fold_wall_seconds"] = seconds
    result = evaluate_usage_u2(child, parent, **args)
    assert result["status"] == "fail"
    assert (
        next(c for c in result["checks"] if c["gate"] == "runtime_and_memory")["status"] == "fail"
    )
