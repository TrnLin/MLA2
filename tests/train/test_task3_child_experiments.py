from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from fashion.train.augmentation import apply_training_augmentation
from fashion.train.task3_experiments import (
    audit_completed_registry_rows,
    effective_number_class_weights,
    gender_brightness_spec,
    gender_class_balanced_spec,
    gender_compact_blur_cnn_spec,
    gender_gem_p3_early_stopping_spec,
    gender_gem_p3_spec,
    gender_tinyhrnet20_spec,
    gender_tinyresnet18_pm_spec,
    latest_completed_baseline_parent_run_ids,
    latest_completed_gender_e6_parent_run_ids,
    latest_completed_usage_e2_parent_run_ids,
    usage_class_balanced_spec,
    usage_classifier_dropout_spec,
    usage_focal_gamma1_spec,
    usage_label_smoothing_spec,
    usage_tinyconvnext18_spec,
    usage_tinyresnet18_pm_spec,
    usage_translation_2px_spec,
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


def test_gender_e3_changes_loss_only_from_the_baseline_parent() -> None:
    parents = tuple(f"gender-parent-{fold}" for fold in range(5))
    spec = gender_class_balanced_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "baseline"
    assert spec.training_augmentation == "none"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.classifier_dropout == 0.0

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, classifier_dropout=0.2)


def test_usage_e3_changes_dropout_only_from_the_accepted_e2_parent() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_classifier_dropout_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.training_augmentation == "none"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.classifier_dropout == pytest.approx(0.2)

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, class_weight_cap=4.0)


def test_gender_e4_changes_only_the_architecture_from_e1() -> None:
    parents = tuple(f"gender-e1-parent-{fold}" for fold in range(5))
    spec = gender_tinyresnet18_pm_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "baseline"
    assert spec.loss_name == "cross_entropy"
    assert spec.training_augmentation == "none"
    assert spec.model_family == "task3_tinyresnet18_pm"
    assert spec.run_model_token == "tinyresnet18pm"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, loss_name="effective_number_cross_entropy")


def test_usage_e4_changes_only_the_architecture_from_e2() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_tinyresnet18_pm_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.classifier_dropout == 0.0
    assert spec.model_family == "task3_tinyresnet18_pm"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, classifier_dropout=0.2)


def test_gender_e5_changes_only_capacity_and_downsampling_from_e1() -> None:
    parents = tuple(f"gender-e1-parent-{fold}" for fold in range(5))
    spec = gender_compact_blur_cnn_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "baseline"
    assert spec.loss_name == "cross_entropy"
    assert spec.training_augmentation == "none"
    assert spec.label_smoothing == 0.0
    assert spec.model_family == "task3_compact_blur_cnn"
    assert spec.run_model_token == "compactblurcnn"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, classifier_dropout=0.2)


def test_usage_e5_changes_only_label_smoothing_from_e2() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_label_smoothing_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.loss_name == "effective_number_label_smoothed_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.label_smoothing == pytest.approx(0.05)
    assert spec.model_family == "task3_small_cnn"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, label_smoothing=0.1)


def test_gender_e6_changes_only_global_pooling_from_e1() -> None:
    parents = tuple(f"gender-e1-parent-{fold}" for fold in range(5))
    spec = gender_gem_p3_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "baseline"
    assert spec.changed_factor == "global_pooling"
    assert spec.loss_name == "cross_entropy"
    assert spec.model_family == "task3_small_cnn_gem_p3"
    assert spec.run_model_token == "smallcnngem3"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, focal_gamma=1.0)


def test_usage_e6_changes_only_focal_modulation_from_e2() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_focal_gamma1_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.changed_factor == "loss_modulation"
    assert spec.loss_name == "effective_number_focal_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.focal_gamma == pytest.approx(1.0)
    assert spec.model_family == "task3_small_cnn"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, label_smoothing=0.05)


def test_gender_e7_changes_only_the_architecture_from_e1() -> None:
    parents = tuple(f"gender-e1-parent-{fold}" for fold in range(5))
    spec = gender_tinyhrnet20_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "baseline"
    assert spec.changed_factor == "model_architecture"
    assert spec.loss_name == "cross_entropy"
    assert spec.model_family == "task3_tinyhrnet20"
    assert spec.run_model_token == "tinyhrnet20"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, classifier_dropout=0.2)


def test_usage_e7_changes_only_the_architecture_from_e2() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_tinyconvnext18_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.changed_factor == "model_architecture"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.model_family == "task3_tinyconvnext18"
    assert spec.run_model_token == "tinyconvnext18"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, focal_gamma=1.0)


def test_gender_e8_changes_only_the_e6_checkpoint_policy() -> None:
    parents = tuple(f"gender-e6-parent-{fold}" for fold in range(5))
    spec = gender_gem_p3_early_stopping_spec(parents)

    assert spec.target == "gender"
    assert spec.parent_artifact_dir == "experiments/t3_gender_e6_gem_p3"
    assert spec.changed_factor == "checkpoint_selection"
    assert spec.model_family == "task3_small_cnn_gem_p3"
    assert spec.checkpoint_policy == "best_validation_macro_f1"
    assert spec.early_stopping_min_epoch == 15
    assert spec.early_stopping_patience == 10
    assert spec.early_stopping_min_delta == pytest.approx(0.001)

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, training_augmentation="translation_uniform_2px_p05")


def test_usage_e8_changes_only_training_translation_from_e2() -> None:
    parents = tuple(f"usage-e2-parent-{fold}" for fold in range(5))
    spec = usage_translation_2px_spec(parents)

    assert spec.target == "usage"
    assert spec.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert spec.changed_factor == "training_translation"
    assert spec.training_augmentation == "translation_uniform_2px_p05"
    assert spec.loss_name == "effective_number_cross_entropy"
    assert spec.class_weight_beta == pytest.approx(0.999)
    assert spec.class_weight_cap == pytest.approx(5.0)
    assert spec.checkpoint_policy == "final_epoch"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, checkpoint_policy="best_validation_macro_f1")


def test_registry_audit_requires_complete_hashed_rows(tmp_path) -> None:
    registry = tmp_path / "runs.csv"
    registry.write_text(
        "run_id,status,config_path,history_path,checkpoint_path,checkpoint_sha256,"
        "prediction_path,prediction_sha256,metrics_json\n"
        "fold-0,complete,config.json,history.csv,final.pt,abc,pred.csv,def,{}\n",
        encoding="utf-8",
    )

    audit = audit_completed_registry_rows(registry, ["fold-0"])

    assert audit["ready"] is True
    assert audit["completed_rows"] == 1

    with pytest.raises(RuntimeError, match="found 0"):
        audit_completed_registry_rows(registry, ["missing"])


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


def test_translation_augmentation_uses_the_frozen_probability_and_offsets(monkeypatch) -> None:
    image = Image.new("RGB", (5, 5), (255, 255, 255))
    image.putpixel((2, 2), (0, 0, 0))
    offsets = iter((1, -1))
    monkeypatch.setattr("fashion.train.augmentation.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "fashion.train.augmentation.random.randint", lambda low, high: next(offsets)
    )

    augmented = apply_training_augmentation(image, "translation_uniform_2px_p05")

    assert augmented.size == image.size
    assert augmented.getpixel((3, 1)) == (0, 0, 0)
    assert augmented.getpixel((2, 2)) == (255, 255, 255)


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
        (run_dir / "robustness.csv").write_text(
            "corruption,macro_f1_change\n", encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps({"run_id": run_id, "validation_fold": fold}),
            encoding="utf-8",
        )

    found = latest_completed_baseline_parent_run_ids("gender", output_root=tmp_path)

    assert found == tuple(expected)


def test_latest_usage_e2_lookup_requires_the_accepted_parent_chain(tmp_path) -> None:
    target_dir = tmp_path / "experiments/t3_usage_e2_class_balanced_ce/usage"
    expected = []
    for fold in range(5):
        run_id = f"t3_usage_e2_class_balanced_ce_usage_smallcnn_f{fold}_complete"
        expected.append(run_id)
        run_dir = target_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "final_epoch.pt").write_bytes(b"checkpoint")
        (run_dir / "oof_predictions.csv").write_text("id\n", encoding="utf-8")
        (run_dir / "robustness.csv").write_text(
            "corruption,macro_f1_change\n", encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {"run_id": run_id, "target": "usage", "validation_fold": fold}
            ),
            encoding="utf-8",
        )

    found = latest_completed_usage_e2_parent_run_ids(output_root=tmp_path)

    assert found == tuple(expected)


def test_latest_gender_e6_lookup_uses_the_gem_run_token(tmp_path) -> None:
    target_dir = tmp_path / "experiments/t3_gender_e6_gem_p3/gender"
    expected = []
    for fold in range(5):
        run_id = f"t3_gender_e6_gem_p3_gender_smallcnngem3_f{fold}_complete"
        expected.append(run_id)
        run_dir = target_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "final_epoch.pt").write_bytes(b"checkpoint")
        (run_dir / "oof_predictions.csv").write_text("id\n", encoding="utf-8")
        (run_dir / "robustness.csv").write_text(
            "corruption,macro_f1_change\n", encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {"run_id": run_id, "target": "gender", "validation_fold": fold}
            ),
            encoding="utf-8",
        )

    found = latest_completed_gender_e6_parent_run_ids(output_root=tmp_path)

    assert found == tuple(expected)
