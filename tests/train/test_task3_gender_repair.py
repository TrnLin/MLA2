from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.train.augmentation import apply_training_augmentation
from fashion.train.task3_clean_slate import _prediction_frame
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics
from fashion.train.task3_gender_repair import NAME, evaluate_gender_repair, require_prerequisites
from fashion.train.task3_gender_repair_preflight import DIAGNOSTICS, SHIFTS

ROOT = Path(__file__).resolve().parents[2]


def test_darkening_does_not_consume_translation_or_sampling_randomness():
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (80, 60, 3), dtype=np.uint8))
    random.seed(23)
    translated = [
        np.asarray(apply_training_augmentation(image, "translation_uniform_2px_p05"))
        for _ in range(100)
    ]
    original_state = random.getstate()
    random.seed(23)
    darkening = random.Random(77)
    repaired = [
        np.asarray(
            apply_training_augmentation(
                image, "translation_2px_p05_mild_darkening_p025", darkening_rng=darkening
            )
        )
        for _ in range(100)
    ]
    assert random.getstate() == original_state
    assert all(np.all(after <= before) for before, after in zip(translated, repaired, strict=True))
    changed = sum(not np.array_equal(a, b) for a, b in zip(translated, repaired, strict=True))
    assert 10 < changed < 40
    random.seed(23)
    replay = random.Random(77)
    assert all(
        np.array_equal(
            expected,
            np.asarray(
                apply_training_augmentation(
                    image, "translation_2px_p05_mild_darkening_p025", darkening_rng=replay
                )
            ),
        )
        for expected in repaired
    )
    with pytest.raises(ValueError, match="own seeded"):
        apply_training_augmentation(image, "translation_2px_p05_mild_darkening_p025")


def test_frozen_spec_keeps_legacy_serialization_and_locks_new_policy():
    parents = [f"g2-parent-{f}" for f in range(5)]
    spec = dataset_v2_spec(NAME, parents)
    assert spec.saved_tensors_on_cpu
    assert spec.parent_artifact_dir == "experiments/t3_gender_v2_g2_translation"
    assert spec.training_augmentation == "translation_2px_p05_mild_darkening_p025"
    assert spec.to_dict()["execution_policy"] == "save_on_cpu_pin_memory_fp32_batch128_v1"
    legacy = dataset_v2_spec("gender_v2_translation", parents)
    assert "saved_tensors_on_cpu" not in legacy.to_dict()
    with pytest.raises(ValueError, match="predeclared factor"):
        replace(spec, classifier_dropout=0.1)
    with pytest.raises(ValueError, match="requires completed"):
        require_prerequisites(None, root=ROOT, spec=spec, device_name="cuda")
    assert len(SHIFTS) == 25 and len(set(SHIFTS)) == 25
    assert set(DIAGNOSTICS) >= {
        "clean",
        "brightness_0.85",
        "brightness_0.90",
        "brightness_0.95",
        "brightness_1.00",
        "shift_0_0",
    }


def gender_case():
    classes = ["Boys", "Girls", "Men", "Unisex", "Women"]
    sources = {"G2": {}, "E6": {}}
    child = {}
    for fold in range(5):
        frame = pd.DataFrame(
            [
                {
                    "id": fold * 100 + i,
                    "cv_fold": fold,
                    "gender": classes[i % 5],
                    "product_family_group": f"{fold}-{i}",
                    "partition": "development",
                    "path": f"{fold}-{i}.jpg",
                }
                for i in range(50)
            ]
        )
        for name, runs in [("child", child), *sources.items()]:
            labels = np.arange(50) % 5
            predicted = labels.copy()
            if name == "E6":
                predicted[::4] = (predicted[::4] + 1) % 5
            probabilities = np.full((50, 5), 0.0025)
            probabilities[np.arange(50), predicted] = 0.99
            predictions = _prediction_frame(
                frame,
                target="gender",
                classes=classes,
                probabilities=probabilities,
                run_id=f"{name}-{fold}",
            )
            metrics = oof_metrics(predictions, classes)
            metrics.update(
                {
                    "final_train_validation_macro_f1_gap": 0.005 if name == "child" else 0.10,
                    "peak_memory_bytes": 400_000_000,
                    "latency_ms_batch_1": 0.5,
                    "train_seconds": 100,
                    "parameter_count": 390181,
                }
            )
            corruption_rows = []
            for corruption in CORE_CORRUPTIONS:
                change = -0.01
                if corruption == "brightness_085":
                    change = -0.12 if name != "child" else -0.03
                if corruption == "translation_003" and name == "E6":
                    change = -0.10
                corruption_rows.append(
                    {
                        "run_id": f"{name}-{fold}",
                        "validation_fold": fold,
                        "corruption": corruption,
                        "macro_f1": metrics["macro_f1"] + change,
                        "macro_f1_change": change,
                    }
                )
            runs[fold] = {
                "run_id": f"{name}-{fold}",
                "predictions": predictions,
                "metrics": metrics,
                "robustness": pd.DataFrame(corruption_rows),
            }
    return child, sources, classes


def test_screen_and_confirmation_apply_real_frozen_rules():
    child, sources, classes = gender_case()
    screen = evaluate_gender_repair(
        {f: child[f] for f in (0, 4)}, sources, classes, phase="screen", repetitions=40
    )
    assert screen["status"] == "pass"
    confirmed = evaluate_gender_repair(
        child, sources, classes, phase="confirmation", repetitions=40
    )
    assert confirmed["status"] == "pass"
    assert "fresh_vs_e6" in confirmed["evidence"]
    assert confirmed["independent_test_evidence"] is False
    child[0]["metrics"]["peak_memory_bytes"] = 476_045_312
    failed = evaluate_gender_repair(
        {f: child[f] for f in (0, 4)}, sources, classes, phase="screen", repetitions=40
    )
    assert failed["status"] == "fail"
    assert (
        next(c for c in failed["checks"] if c["gate"] == "fold_0.gpu_memory_bytes")["status"]
        == "fail"
    )


def test_fresh_fold_ci_cannot_be_replaced_by_pooled_improvement():
    child, sources, classes = gender_case()
    for f in (1, 2, 3):
        reference = sources["E6"][f]
        reference["predictions"] = child[f]["predictions"].assign(run_id=f"E6-{f}")
        old_f1 = reference["metrics"]["macro_f1"]
        reference["metrics"]["macro_f1"] = child[f]["metrics"]["macro_f1"]
        reference["robustness"]["macro_f1"] += reference["metrics"]["macro_f1"] - old_f1
    report = evaluate_gender_repair(child, sources, classes, phase="confirmation", repetitions=40)
    assert report["status"] == "fail"
    fresh = next(c for c in report["checks"] if c["gate"] == "fresh_vs_e6.ci_lower")
    assert fresh["status"] == "fail"
    assert (
        report["evidence"]["five_vs_e6"]["candidate"]["macro_f1"]
        > report["evidence"]["five_vs_e6"]["comparison"]["macro_f1"]
    )


def test_notebook_defaults_to_screen_and_retains_no_outputs():
    nb = json.loads((ROOT / "notebooks/04t_task3_gender_gd1_mild_darkening.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'phase="screen"' in code
    assert 'phase="confirmation"' not in code
    assert code.index("prepare_gender_repair(") < code.index("run_gender_repair(")
    assert not any(c.get("outputs") for c in nb["cells"])
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"04t-cell-{i}", "exec")
