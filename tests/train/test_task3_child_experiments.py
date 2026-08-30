from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from fashion.train.augmentation import apply_training_augmentation
from fashion.train.task3_experiments import (
    effective_number_class_weights,
    gender_brightness_spec,
    latest_completed_baseline_parent_run_ids,
    usage_class_balanced_spec,
)


def test_gender_child_changes_brightness_only() -> None:
    parents = tuple(f"gender-parent-{fold}" for fold in range(5))
    spec = gender_brightness_spec(parents)

    assert spec.target == "gender"
    assert spec.training_augmentation == "brightness_uniform_085_115"
    assert spec.loss_name == "cross_entropy"
    assert spec.class_weight_beta is None
    assert spec.parent_run_ids == parents

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, loss_name="effective_number_cross_entropy")


def test_usage_child_changes_loss_only() -> None:
    parents = tuple(f"usage-parent-{fold}" for fold in range(5))
    spec = usage_class_balanced_spec(parents)

    assert spec.target == "usage"
    assert spec.training_augmentation == "none"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, training_augmentation="brightness_uniform_085_115")


def test_effective_number_weights_are_fold_only_capped_and_zero_for_absent_class() -> None:
    weights = effective_number_class_weights([25_000, 3_000, 100, 10, 0])

    assert weights.shape == (5,)
    assert weights[-1] == 0.0
    assert weights.max() <= 5.0
    assert weights[3] > weights[2] > weights[1] > weights[0]


def test_brightness_augmentation_uses_the_predeclared_range(monkeypatch) -> None:
    image = Image.new("RGB", (2, 2), (100, 100, 100))
    monkeypatch.setattr("fashion.train.augmentation.random.uniform", lambda low, high: low)

    augmented = apply_training_augmentation(image, "brightness_uniform_085_115")

    assert np.asarray(augmented).tolist() == np.full((2, 2, 3), 85).tolist()


def test_latest_parent_lookup_requires_one_complete_baseline_per_fold(tmp_path) -> None:
    target_dir = tmp_path / "baseline/gender"
    expected = []
    for fold in range(5):
        run_id = f"t3_baseline_gender_smallcnn_f{fold}_complete"
        expected.append(run_id)
        run_dir = target_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "final_epoch.pt").write_bytes(b"checkpoint")
        (run_dir / "oof_predictions.csv").write_text("id\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps({"run_id": run_id, "validation_fold": fold}),
            encoding="utf-8",
        )

    found = latest_completed_baseline_parent_run_ids("gender", output_root=tmp_path)

    assert found == tuple(expected)
