from __future__ import annotations

import csv

import numpy as np
import pytest

from fashion.train.config import (
    Task3BaselineConfig,
    baseline_parameter_count,
    compact_blur_cnn_macs,
    compact_blur_cnn_parameter_count,
    config_digest,
    tinyconvnext18_macs,
    tinyconvnext18_parameter_count,
    tinyhrnet20_macs,
    tinyhrnet20_parameter_count,
    tinyresnet18_pm_macs,
    tinyresnet18_pm_parameter_count,
)
from fashion.train.metrics import classification_metrics
from fashion.train.registry import REGISTRY_COLUMNS, RunRegistry
from fashion.train.task3_experiments import _EarlyStoppingTracker


def test_baseline_configuration_and_parameter_counts_are_frozen() -> None:
    gender = Task3BaselineConfig(target="gender")
    usage = Task3BaselineConfig(target="usage")

    assert gender.to_dict()["channels"] == [32, 64, 128, 256]
    assert gender.num_classes == 5
    assert usage.num_classes == 9
    assert baseline_parameter_count("gender") == 390_181
    assert baseline_parameter_count("usage") == 391_209
    assert tinyresnet18_pm_parameter_count("gender") == 394_865
    assert tinyresnet18_pm_parameter_count("usage") == 395_253
    assert tinyresnet18_pm_macs("gender") == 94_268_640
    assert tinyresnet18_pm_macs("usage") == 94_269_024
    assert compact_blur_cnn_parameter_count("gender") == 67_069
    assert compact_blur_cnn_parameter_count("usage") == 67_585
    assert compact_blur_cnn_macs("gender") == 29_504_320
    assert compact_blur_cnn_macs("usage") == 29_504_832
    assert tinyhrnet20_parameter_count("gender") == 374_445
    assert tinyhrnet20_macs("gender") == 104_064_700
    assert tinyconvnext18_parameter_count("usage") == 384_345
    assert tinyconvnext18_macs("usage") == 95_297_616
    assert config_digest(gender) == config_digest(Task3BaselineConfig(target="gender"))
    assert config_digest(gender) != config_digest(usage)


def test_baseline_configuration_rejects_a_second_changed_factor() -> None:
    with pytest.raises(ValueError, match="channels"):
        Task3BaselineConfig(target="gender", channels=(16, 32, 64, 128))
    with pytest.raises(ValueError, match="augmentation"):
        Task3BaselineConfig(target="gender", augmentation="horizontal_flip")
    with pytest.raises(ValueError, match="80x60"):
        Task3BaselineConfig(target="usage", image_height=128, image_width=96)


def test_early_stopping_tracker_uses_the_frozen_delta_and_patience() -> None:
    tracker = _EarlyStoppingTracker(min_epoch=3, patience=2, min_delta=0.01)

    assert tracker.update(1, 0.50) is True
    assert tracker.update(2, 0.505) is False
    assert tracker.should_stop(2) is False
    assert tracker.update(3, 0.509) is False
    assert tracker.should_stop(3) is True
    assert tracker.best_epoch == 1
    assert tracker.best_score == pytest.approx(0.50)


def test_fixed_class_metrics_include_zero_support_classes() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.6, 0.4, 0.0],
            [0.2, 0.8, 0.0],
            [0.7, 0.3, 0.0],
        ]
    )

    metrics = classification_metrics(labels, probabilities, ["a", "b", "missing"])

    assert metrics["support"] == 4
    assert len(metrics["per_class"]) == 3
    assert metrics["per_class"][2]["support"] == 0
    assert metrics["per_class"][2]["f1"] == 0.0
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 0.0) / 3)


def test_registry_appends_before_completion_and_mirrors_atomically(tmp_path) -> None:
    path = tmp_path / "results/runs.csv"
    mirror = tmp_path / "drive/runs.csv"
    registry = RunRegistry(path, mirrors=[mirror])

    started = registry.start(
        {
            "run_id": "t3_baseline_gender_f0",
            "experiment_id": "t3_primary_baseline_smallcnn",
            "task": "task3",
            "target": "gender",
            "validation_fold": 0,
            "seed": 2753,
            "scratch": True,
            "submission_eligible": True,
        }
    )
    assert started["status"] == "running"
    assert path.read_bytes() == mirror.read_bytes()

    completed = registry.complete(
        "t3_baseline_gender_f0",
        {"metrics_json": {"macro_f1": 0.25}, "last_completed_stage": "complete"},
    )
    assert completed["status"] == "complete"
    assert path.read_bytes() == mirror.read_bytes()

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == REGISTRY_COLUMNS
    assert rows[0]["run_id"] == "t3_baseline_gender_f0"
    assert rows[0]["status"] == "complete"
    assert rows[0]["metrics_json"] == '{"macro_f1":0.25}'
    with pytest.raises(ValueError, match="final registry row"):
        registry.update("t3_baseline_gender_f0", {"status": "running"})


def test_registry_keeps_failed_execution(tmp_path) -> None:
    path = tmp_path / "results/runs.csv"
    registry = RunRegistry(path)
    registry.start({"run_id": "failed-run", "task": "task3", "target": "usage"})

    failed = registry.fail(
        "failed-run",
        RuntimeError("bad batch"),
        last_completed_stage="epoch_2_complete",
    )

    assert failed["status"] == "failed"
    assert failed["exception_type"] == "RuntimeError"
    assert failed["exception_message"] == "bad batch"
    assert failed["last_completed_stage"] == "epoch_2_complete"
