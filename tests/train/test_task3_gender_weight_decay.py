from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_clean_slate import _prediction_frame
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics
from fashion.train.task3_gender_weight_decay import (
    NAME,
    _verify_baseline_controls,
    evaluate_weight_decay_screen,
    weight_decay_config,
)


def test_weight_decay_is_the_only_baseline_control_change():
    ids = [f"g2-{f}" for f in range(5)]
    spec = dataset_v2_spec(NAME, ids)
    config = weight_decay_config(spec, fold=0, device_name="cuda")
    parent = Task3BaselineConfig(target="gender")
    changes = {k for k in config.to_dict() if config.to_dict()[k] != parent.to_dict()[k]}
    assert changes == {"weight_decay"}
    assert config.weight_decay == spec.weight_decay == 0.01
    assert spec.training_augmentation == "translation_uniform_2px_p05"
    assert spec.saved_tensors_on_cpu is False
    assert spec.to_dict()["screen_rule_version"] == "gwd1_gap_and_validation_v1"
    legacy = dataset_v2_spec("gender_v2_translation", ids)
    assert "weight_decay" not in legacy.to_dict()
    with pytest.raises(ValueError, match="predeclared factor"):
        replace(spec, training_augmentation="translation_2px_p05_mild_darkening_p025")
    with pytest.raises(ValueError, match="folds 0 and 4"):
        weight_decay_config(spec, fold=1, device_name="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        weight_decay_config(spec, fold=0, device_name="cpu")
    with pytest.raises(ValueError, match="weight_decay"):
        _verify_baseline_controls(parent.to_dict(), config)


def _case():
    classes = ["Boys", "Girls", "Men", "Unisex", "Women"]
    sources, child = {"G2": {}, "E6": {}}, {}
    for fold in (0, 4):
        frame = pd.DataFrame(
            dict(
                id=fold * 1000 + np.arange(100),
                cv_fold=fold,
                gender=[classes[i % 5] for i in range(100)],
                product_family_group=[f"family-{fold}-{i}" for i in range(100)],
                path=[f"{fold}-{i}.jpg" for i in range(100)],
                partition="development",
            )
        )
        for name, runs in [("child", child), *sources.items()]:
            labels = np.arange(100) % 5
            predicted = labels.copy()
            count = 10 if name == "child" else 30
            predicted[:count] = (predicted[:count] + 1) % 5
            p = np.full((100, 5), 0.025)
            p[np.arange(100), predicted] = 0.9
            predictions = _prediction_frame(
                frame, target="gender", classes=classes, probabilities=p, run_id=f"{name}-{fold}"
            )
            metrics = oof_metrics(predictions, classes)
            training = 0.97 if name == "child" else 0.99
            metrics.update(
                final_train_eval_macro_f1=training,
                final_train_validation_macro_f1_gap=training - metrics["macro_f1"],
                parameter_count=390181,
                peak_memory_bytes=477585408,
                train_seconds=999999,
                latency_ms_batch_1=999,
            )
            robust = []
            for corruption in CORE_CORRUPTIONS:
                delta = -0.10 if name == "E6" else -0.01
                robust.append(
                    dict(
                        run_id=f"{name}-{fold}",
                        validation_fold=fold,
                        corruption=corruption,
                        macro_f1=metrics["macro_f1"] + delta,
                        macro_f1_change=delta,
                    )
                )
            runs[fold] = dict(
                run_id=f"{name}-{fold}",
                predictions=predictions,
                metrics=metrics,
                robustness=pd.DataFrame(robust),
            )
    return child, sources, classes


def test_better_validation_and_smaller_clean_gap_pass_without_speed_cap():
    child, sources, classes = _case()
    result = evaluate_weight_decay_screen(child, sources, classes, repetitions=100)
    assert result["status"] == "pass"
    assert result["speed_cap"] is None
    assert result["folds"][0]["train_seconds"] == 999999
    assert result["required_validation_f1"] == pytest.approx(0.71)
    assert result["maximum_mean_clean_gap"] == pytest.approx(0.26)


def test_weaker_training_alone_cannot_pass():
    child, sources, classes = _case()
    for f in (0, 4):
        parent = sources["G2"][f]
        child[f]["predictions"] = parent["predictions"].assign(run_id=f"child-{f}")
        child[f]["metrics"].update(
            macro_f1=parent["metrics"]["macro_f1"],
            final_train_eval_macro_f1=0.8,
            final_train_validation_macro_f1_gap=0.8 - parent["metrics"]["macro_f1"],
        )
        child[f]["robustness"]["macro_f1"] = (
            child[f]["metrics"]["macro_f1"] + child[f]["robustness"]["macro_f1_change"]
        )
    result = evaluate_weight_decay_screen(child, sources, classes, repetitions=100)
    checks = {c["gate"]: c["status"] for c in result["checks"]}
    assert checks["mean_gap_reduction"] == "pass"
    assert checks["validation_gain"] == checks["validation_ci_lower"] == "fail"
    assert result["status"] == "fail"


def test_validation_gain_alone_cannot_hide_a_worse_gap():
    child, sources, classes = _case()
    for p in sources["G2"].values():
        p["metrics"]["final_train_eval_macro_f1"] = p["metrics"]["macro_f1"] + 0.02
        p["metrics"]["final_train_validation_macro_f1_gap"] = 0.02
    result = evaluate_weight_decay_screen(child, sources, classes, repetitions=100)
    checks = {c["gate"]: c["status"] for c in result["checks"]}
    assert checks["validation_gain"] == "pass"
    assert checks["mean_gap_reduction"] == "fail"
    assert result["status"] == "fail"


def test_memory_boundary_and_wrong_gap_are_rejected():
    child, sources, classes = _case()
    child[0]["metrics"]["peak_memory_bytes"] = 3_000_000_000
    result = evaluate_weight_decay_screen(child, sources, classes, repetitions=40)
    assert (
        next(c for c in result["checks"] if c["gate"] == "fold_0.gpu_memory_bytes")["status"]
        == "fail"
    )
    child[0]["metrics"]["final_train_validation_macro_f1_gap"] = 0
    with pytest.raises(ValueError, match="final checkpoint"):
        evaluate_weight_decay_screen(child, sources, classes, repetitions=40)


def test_notebook_runs_screen_only_and_compiles():
    root = Path(__file__).resolve().parents[2]
    nb = json.loads((root / "notebooks/04u_task3_gender_weight_decay_screen.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "run_gender_weight_decay_screen(" in code
    assert "confirmation" not in code
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "G-WD1-notebook", "exec")
