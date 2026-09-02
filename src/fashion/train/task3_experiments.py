"""Strict one-factor child experiments for the Task 3 scratch models."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from fashion.config import ROOT, RUNS_CSV
from fashion.train.config import Task3Target

Task3ChildName = Literal[
    "gender_brightness",
    "usage_class_balanced",
    "gender_class_balanced",
    "usage_classifier_dropout",
    "gender_tinyresnet18_pm",
    "usage_tinyresnet18_pm",
    "gender_compact_blur_cnn",
    "usage_label_smoothing",
    "gender_gem_p3",
    "usage_focal_gamma1",
    "gender_tinyhrnet20",
    "usage_tinyconvnext18",
    "gender_gem_p3_early_stopping",
    "usage_translation_2px",
    "gender_semantic_filter",
    "usage_exception_balance",
    "gender_audience_aux",
]
Task3ModelFamily = Literal[
    "task3_small_cnn",
    "task3_small_cnn_gem_p3",
    "task3_tinyresnet18_pm",
    "task3_compact_blur_cnn",
    "task3_tinyhrnet20",
    "task3_tinyconvnext18",
    "task3_small_cnn_gem_p3_audience_aux",
]
Task3RunModelToken = Literal[
    "smallcnn",
    "smallcnngem3",
    "tinyresnet18pm",
    "compactblurcnn",
    "tinyhrnet20",
    "tinyconvnext18",
    "smallcnngem3aux3",
]

GENDER_E10_PRIMARY_LOSS_WEIGHT = 0.5
GENDER_E10_AUXILIARY_LOSS_WEIGHT = 0.5 * math.log(5.0) / math.log(3.0)
GENDER_E10_PEAK_MEMORY_LIMIT_BYTES = 2 * 1024**3


@dataclass
class _EarlyStoppingTracker:
    """Track one frozen best-validation checkpoint policy."""

    min_epoch: int
    patience: int
    min_delta: float
    best_score: float = float("-inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.min_epoch < 1:
            raise ValueError("early-stopping minimum epoch must be positive")
        if self.patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.min_delta < 0:
            raise ValueError("early-stopping minimum delta cannot be negative")

    def update(self, epoch: int, score: float) -> bool:
        """Record a qualifying improvement and return whether it was selected."""
        if epoch < 1 or not np.isfinite(score):
            raise ValueError("early-stopping epoch must be positive and score must be finite")
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    def should_stop(self, epoch: int) -> bool:
        return epoch >= self.min_epoch and self.epochs_without_improvement >= self.patience


@dataclass(frozen=True)
class Task3ChildSpec:
    """A predeclared child that may differ from the baseline in one factor only."""

    name: Task3ChildName
    target: Task3Target
    experiment_id: str
    hypothesis_id: str
    artifact_dir: str
    run_prefix: str
    changed_factor: str
    training_augmentation: str
    loss_name: str
    parent_artifact_dir: str
    parent_run_ids: tuple[str, str, str, str, str]
    class_weight_beta: float | None = None
    class_weight_cap: float | None = None
    classifier_dropout: float = 0.0
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0
    model_family: Task3ModelFamily = "task3_small_cnn"
    run_model_token: Task3RunModelToken = "smallcnn"
    checkpoint_policy: Literal["final_epoch", "best_validation_macro_f1"] = "final_epoch"
    early_stopping_min_epoch: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    training_selection_strategy: Literal[
        "all",
        "gender_semantic_conflicts_v1",
        "usage_article_type_exception_balance_v1",
    ] = "all"
    auxiliary_target: Literal["none", "gender_audience_3way"] = "none"
    primary_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if len(set(self.parent_run_ids)) != 5 or any(not run_id for run_id in self.parent_run_ids):
            raise ValueError("a Task 3 child requires five distinct completed parent run IDs")
        parent_path = Path(self.parent_artifact_dir)
        if parent_path.is_absolute() or ".." in parent_path.parts:
            raise ValueError("parent artifact directory must stay inside the Task 3 output root")
        expected = {
            "gender_brightness": {
                "target": "gender",
                "experiment_id": "t3_gender_brightness_smallcnn",
                "hypothesis_id": "t3_gender_e2_brightness",
                "artifact_dir": "experiments/t3_gender_e2_brightness",
                "run_prefix": "t3_gender_e2_brightness",
                "changed_factor": "brightness_augmentation",
                "training_augmentation": "brightness_uniform_085_115",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "usage_class_balanced": {
                "target": "usage",
                "experiment_id": "t3_usage_class_balanced_smallcnn",
                "hypothesis_id": "t3_usage_e2_class_balanced_ce",
                "artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "run_prefix": "t3_usage_e2_class_balanced_ce",
                "changed_factor": "class_balanced_loss",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "gender_class_balanced": {
                "target": "gender",
                "experiment_id": "t3_gender_class_balanced_smallcnn",
                "hypothesis_id": "t3_gender_e3_class_balanced_ce",
                "artifact_dir": "experiments/t3_gender_e3_class_balanced_ce",
                "run_prefix": "t3_gender_e3_class_balanced_ce",
                "changed_factor": "class_balanced_loss",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "usage_classifier_dropout": {
                "target": "usage",
                "experiment_id": "t3_usage_classifier_dropout_smallcnn",
                "hypothesis_id": "t3_usage_e3_classifier_dropout",
                "artifact_dir": "experiments/t3_usage_e3_classifier_dropout",
                "run_prefix": "t3_usage_e3_classifier_dropout",
                "changed_factor": "classifier_dropout",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.2,
                "label_smoothing": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "gender_tinyresnet18_pm": {
                "target": "gender",
                "experiment_id": "t3_gender_tinyresnet18_pm",
                "hypothesis_id": "t3_gender_e4_tinyresnet18_pm",
                "artifact_dir": "experiments/t3_gender_e4_tinyresnet18_pm",
                "run_prefix": "t3_gender_e4_tinyresnet18_pm",
                "changed_factor": "model_architecture",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_tinyresnet18_pm",
                "run_model_token": "tinyresnet18pm",
            },
            "usage_tinyresnet18_pm": {
                "target": "usage",
                "experiment_id": "t3_usage_tinyresnet18_pm",
                "hypothesis_id": "t3_usage_e4_tinyresnet18_pm",
                "artifact_dir": "experiments/t3_usage_e4_tinyresnet18_pm",
                "run_prefix": "t3_usage_e4_tinyresnet18_pm",
                "changed_factor": "model_architecture",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_tinyresnet18_pm",
                "run_model_token": "tinyresnet18pm",
            },
            "gender_compact_blur_cnn": {
                "target": "gender",
                "experiment_id": "t3_gender_compact_blur_cnn",
                "hypothesis_id": "t3_gender_e5_compact_blur_cnn",
                "artifact_dir": "experiments/t3_gender_e5_compact_blur_cnn",
                "run_prefix": "t3_gender_e5_compact_blur_cnn",
                "changed_factor": "model_architecture",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "model_family": "task3_compact_blur_cnn",
                "run_model_token": "compactblurcnn",
            },
            "usage_label_smoothing": {
                "target": "usage",
                "experiment_id": "t3_usage_label_smoothing_smallcnn",
                "hypothesis_id": "t3_usage_e5_label_smoothing",
                "artifact_dir": "experiments/t3_usage_e5_label_smoothing",
                "run_prefix": "t3_usage_e5_label_smoothing",
                "changed_factor": "label_smoothing",
                "training_augmentation": "none",
                "loss_name": "effective_number_label_smoothed_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.05,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "gender_gem_p3": {
                "target": "gender",
                "experiment_id": "t3_gender_gem_p3_smallcnn",
                "hypothesis_id": "t3_gender_e6_gem_p3",
                "artifact_dir": "experiments/t3_gender_e6_gem_p3",
                "run_prefix": "t3_gender_e6_gem_p3",
                "changed_factor": "global_pooling",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn_gem_p3",
                "run_model_token": "smallcnngem3",
            },
            "usage_focal_gamma1": {
                "target": "usage",
                "experiment_id": "t3_usage_focal_gamma1_smallcnn",
                "hypothesis_id": "t3_usage_e6_focal_gamma1",
                "artifact_dir": "experiments/t3_usage_e6_focal_gamma1",
                "run_prefix": "t3_usage_e6_focal_gamma1",
                "changed_factor": "loss_modulation",
                "training_augmentation": "none",
                "loss_name": "effective_number_focal_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 1.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
            },
            "gender_tinyhrnet20": {
                "target": "gender",
                "experiment_id": "t3_gender_tinyhrnet20",
                "hypothesis_id": "t3_gender_e7_tinyhrnet20",
                "artifact_dir": "experiments/t3_gender_e7_tinyhrnet20",
                "run_prefix": "t3_gender_e7_tinyhrnet20",
                "changed_factor": "model_architecture",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "baseline",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_tinyhrnet20",
                "run_model_token": "tinyhrnet20",
            },
            "usage_tinyconvnext18": {
                "target": "usage",
                "experiment_id": "t3_usage_tinyconvnext18",
                "hypothesis_id": "t3_usage_e7_tinyconvnext18",
                "artifact_dir": "experiments/t3_usage_e7_tinyconvnext18",
                "run_prefix": "t3_usage_e7_tinyconvnext18",
                "changed_factor": "model_architecture",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_tinyconvnext18",
                "run_model_token": "tinyconvnext18",
            },
            "gender_gem_p3_early_stopping": {
                "target": "gender",
                "experiment_id": "t3_gender_gem_p3_early_stopping",
                "hypothesis_id": "t3_gender_e8_early_stopping",
                "artifact_dir": "experiments/t3_gender_e8_early_stopping",
                "run_prefix": "t3_gender_e8_early_stopping",
                "changed_factor": "checkpoint_selection",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn_gem_p3",
                "run_model_token": "smallcnngem3",
                "checkpoint_policy": "best_validation_macro_f1",
                "early_stopping_min_epoch": 15,
                "early_stopping_patience": 10,
                "early_stopping_min_delta": 0.001,
            },
            "usage_translation_2px": {
                "target": "usage",
                "experiment_id": "t3_usage_translation_2px_smallcnn",
                "hypothesis_id": "t3_usage_e8_translation",
                "artifact_dir": "experiments/t3_usage_e8_translation",
                "run_prefix": "t3_usage_e8_translation",
                "changed_factor": "training_translation",
                "training_augmentation": "translation_uniform_2px_p05",
                "loss_name": "effective_number_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
                "checkpoint_policy": "final_epoch",
                "early_stopping_min_epoch": 0,
                "early_stopping_patience": 0,
                "early_stopping_min_delta": 0.0,
            },
            "gender_semantic_filter": {
                "target": "gender",
                "experiment_id": "t3_gender_semantic_filter_gem_p3",
                "hypothesis_id": "t3_gender_e9_semantic_filter",
                "artifact_dir": "experiments/t3_gender_e9_semantic_filter",
                "run_prefix": "t3_gender_e9_semantic_filter",
                "changed_factor": "fold_training_semantic_filter",
                "training_augmentation": "none",
                "loss_name": "cross_entropy",
                "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn_gem_p3",
                "run_model_token": "smallcnngem3",
                "checkpoint_policy": "final_epoch",
                "early_stopping_min_epoch": 0,
                "early_stopping_patience": 0,
                "early_stopping_min_delta": 0.0,
                "training_selection_strategy": "gender_semantic_conflicts_v1",
            },
            "usage_exception_balance": {
                "target": "usage",
                "experiment_id": "t3_usage_exception_balance_smallcnn",
                "hypothesis_id": "t3_usage_e9_exception_balance",
                "artifact_dir": "experiments/t3_usage_e9_exception_balance",
                "run_prefix": "t3_usage_e9_exception_balance",
                "changed_factor": "fold_training_article_type_group_balance",
                "training_augmentation": "none",
                "loss_name": "effective_number_group_balanced_cross_entropy",
                "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn",
                "run_model_token": "smallcnn",
                "checkpoint_policy": "final_epoch",
                "early_stopping_min_epoch": 0,
                "early_stopping_patience": 0,
                "early_stopping_min_delta": 0.0,
                "training_selection_strategy": "usage_article_type_exception_balance_v1",
            },
            "gender_audience_aux": {
                "target": "gender",
                "experiment_id": "t3_gender_audience_aux_gem_p3",
                "hypothesis_id": "t3_gender_e10_audience_aux",
                "artifact_dir": "experiments/t3_gender_e10_audience_aux",
                "run_prefix": "t3_gender_e10_audience_aux",
                "changed_factor": "training_only_audience_auxiliary_head",
                "training_augmentation": "none",
                "loss_name": "gender_audience_auxiliary_cross_entropy",
                "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
                "class_weight_beta": None,
                "class_weight_cap": None,
                "classifier_dropout": 0.0,
                "label_smoothing": 0.0,
                "focal_gamma": 0.0,
                "model_family": "task3_small_cnn_gem_p3_audience_aux",
                "run_model_token": "smallcnngem3aux3",
                "checkpoint_policy": "final_epoch",
                "early_stopping_min_epoch": 0,
                "early_stopping_patience": 0,
                "early_stopping_min_delta": 0.0,
                "training_selection_strategy": "all",
                "auxiliary_target": "gender_audience_3way",
                "primary_loss_weight": GENDER_E10_PRIMARY_LOSS_WEIGHT,
                "auxiliary_loss_weight": GENDER_E10_AUXILIARY_LOSS_WEIGHT,
            },
        }[self.name]
        expected.setdefault("focal_gamma", 0.0)
        expected.setdefault("checkpoint_policy", "final_epoch")
        expected.setdefault("early_stopping_min_epoch", 0)
        expected.setdefault("early_stopping_patience", 0)
        expected.setdefault("early_stopping_min_delta", 0.0)
        expected.setdefault("training_selection_strategy", "all")
        expected.setdefault("auxiliary_target", "none")
        expected.setdefault("primary_loss_weight", 1.0)
        expected.setdefault("auxiliary_loss_weight", 0.0)
        actual = asdict(self)
        actual.pop("name")
        actual.pop("parent_run_ids")
        mismatches = {
            field: {"expected": value, "actual": actual[field]}
            for field, value in expected.items()
            if actual[field] != value
        }
        if mismatches:
            raise ValueError(
                "Task 3 child changes more than its predeclared factor: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parent_run_ids"] = list(self.parent_run_ids)
        stage_two = {"gender_brightness", "usage_class_balanced"}
        stage_three = {"gender_class_balanced", "usage_classifier_dropout"}
        stage_four = {"gender_tinyresnet18_pm", "usage_tinyresnet18_pm"}
        stage_five = {"gender_compact_blur_cnn", "usage_label_smoothing"}
        stage_six_or_seven = {
            "gender_gem_p3",
            "usage_focal_gamma1",
            "gender_tinyhrnet20",
            "usage_tinyconvnext18",
        }
        checkpoint_fields = {
            "checkpoint_policy",
            "early_stopping_min_epoch",
            "early_stopping_patience",
            "early_stopping_min_delta",
        }
        if self.name in stage_two:
            for field in (
                "parent_artifact_dir",
                "classifier_dropout",
                "label_smoothing",
                "focal_gamma",
                "model_family",
                "run_model_token",
                *checkpoint_fields,
                "training_selection_strategy",
            ):
                payload.pop(field)
        elif self.name in stage_three:
            for field in (
                "label_smoothing",
                "focal_gamma",
                "model_family",
                "run_model_token",
                *checkpoint_fields,
                "training_selection_strategy",
            ):
                payload.pop(field)
        elif self.name in stage_four:
            for field in (
                "label_smoothing",
                "focal_gamma",
                *checkpoint_fields,
                "training_selection_strategy",
            ):
                payload.pop(field)
        elif self.name in stage_five:
            for field in (
                "focal_gamma",
                *checkpoint_fields,
                "training_selection_strategy",
            ):
                payload.pop(field)
        elif self.name in stage_six_or_seven:
            for field in (*checkpoint_fields, "training_selection_strategy"):
                payload.pop(field)
        elif payload["training_selection_strategy"] == "all":
            payload.pop("training_selection_strategy")
        if payload["auxiliary_target"] == "none":
            payload.pop("auxiliary_target")
            payload.pop("primary_loss_weight")
            payload.pop("auxiliary_loss_weight")
        return payload


def gender_brightness_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="gender_brightness",
        target="gender",
        experiment_id="t3_gender_brightness_smallcnn",
        hypothesis_id="t3_gender_e2_brightness",
        artifact_dir="experiments/t3_gender_e2_brightness",
        run_prefix="t3_gender_e2_brightness",
        changed_factor="brightness_augmentation",
        training_augmentation="brightness_uniform_085_115",
        loss_name="cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
    )


def usage_class_balanced_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="usage_class_balanced",
        target="usage",
        experiment_id="t3_usage_class_balanced_smallcnn",
        hypothesis_id="t3_usage_e2_class_balanced_ce",
        artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        run_prefix="t3_usage_e2_class_balanced_ce",
        changed_factor="class_balanced_loss",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
    )


def gender_class_balanced_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="gender_class_balanced",
        target="gender",
        experiment_id="t3_gender_class_balanced_smallcnn",
        hypothesis_id="t3_gender_e3_class_balanced_ce",
        artifact_dir="experiments/t3_gender_e3_class_balanced_ce",
        run_prefix="t3_gender_e3_class_balanced_ce",
        changed_factor="class_balanced_loss",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
    )


def usage_classifier_dropout_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="usage_classifier_dropout",
        target="usage",
        experiment_id="t3_usage_classifier_dropout_smallcnn",
        hypothesis_id="t3_usage_e3_classifier_dropout",
        artifact_dir="experiments/t3_usage_e3_classifier_dropout",
        run_prefix="t3_usage_e3_classifier_dropout",
        changed_factor="classifier_dropout",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        classifier_dropout=0.2,
    )


def gender_tinyresnet18_pm_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="gender_tinyresnet18_pm",
        target="gender",
        experiment_id="t3_gender_tinyresnet18_pm",
        hypothesis_id="t3_gender_e4_tinyresnet18_pm",
        artifact_dir="experiments/t3_gender_e4_tinyresnet18_pm",
        run_prefix="t3_gender_e4_tinyresnet18_pm",
        changed_factor="model_architecture",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_tinyresnet18_pm",
        run_model_token="tinyresnet18pm",
    )


def usage_tinyresnet18_pm_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="usage_tinyresnet18_pm",
        target="usage",
        experiment_id="t3_usage_tinyresnet18_pm",
        hypothesis_id="t3_usage_e4_tinyresnet18_pm",
        artifact_dir="experiments/t3_usage_e4_tinyresnet18_pm",
        run_prefix="t3_usage_e4_tinyresnet18_pm",
        changed_factor="model_architecture",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        model_family="task3_tinyresnet18_pm",
        run_model_token="tinyresnet18pm",
    )


def gender_compact_blur_cnn_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="gender_compact_blur_cnn",
        target="gender",
        experiment_id="t3_gender_compact_blur_cnn",
        hypothesis_id="t3_gender_e5_compact_blur_cnn",
        artifact_dir="experiments/t3_gender_e5_compact_blur_cnn",
        run_prefix="t3_gender_e5_compact_blur_cnn",
        changed_factor="model_architecture",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_compact_blur_cnn",
        run_model_token="compactblurcnn",
    )


def usage_label_smoothing_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="usage_label_smoothing",
        target="usage",
        experiment_id="t3_usage_label_smoothing_smallcnn",
        hypothesis_id="t3_usage_e5_label_smoothing",
        artifact_dir="experiments/t3_usage_e5_label_smoothing",
        run_prefix="t3_usage_e5_label_smoothing",
        changed_factor="label_smoothing",
        training_augmentation="none",
        loss_name="effective_number_label_smoothed_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        label_smoothing=0.05,
    )


def gender_gem_p3_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only Gender E1 global average pooling to fixed GeM p=3."""
    return Task3ChildSpec(
        name="gender_gem_p3",
        target="gender",
        experiment_id="t3_gender_gem_p3_smallcnn",
        hypothesis_id="t3_gender_e6_gem_p3",
        artifact_dir="experiments/t3_gender_e6_gem_p3",
        run_prefix="t3_gender_e6_gem_p3",
        changed_factor="global_pooling",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_small_cnn_gem_p3",
        run_model_token="smallcnngem3",
    )


def usage_focal_gamma1_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only Usage E2 cross-entropy modulation to focal gamma=1."""
    return Task3ChildSpec(
        name="usage_focal_gamma1",
        target="usage",
        experiment_id="t3_usage_focal_gamma1_smallcnn",
        hypothesis_id="t3_usage_e6_focal_gamma1",
        artifact_dir="experiments/t3_usage_e6_focal_gamma1",
        run_prefix="t3_usage_e6_focal_gamma1",
        changed_factor="loss_modulation",
        training_augmentation="none",
        loss_name="effective_number_focal_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        focal_gamma=1.0,
    )


def gender_tinyhrnet20_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only the Gender E1 representation to TinyHRNet-20."""
    return Task3ChildSpec(
        name="gender_tinyhrnet20",
        target="gender",
        experiment_id="t3_gender_tinyhrnet20",
        hypothesis_id="t3_gender_e7_tinyhrnet20",
        artifact_dir="experiments/t3_gender_e7_tinyhrnet20",
        run_prefix="t3_gender_e7_tinyhrnet20",
        changed_factor="model_architecture",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="baseline",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_tinyhrnet20",
        run_model_token="tinyhrnet20",
    )


def usage_tinyconvnext18_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only the Usage E2 representation to TinyConvNeXt-18."""
    return Task3ChildSpec(
        name="usage_tinyconvnext18",
        target="usage",
        experiment_id="t3_usage_tinyconvnext18",
        hypothesis_id="t3_usage_e7_tinyconvnext18",
        artifact_dir="experiments/t3_usage_e7_tinyconvnext18",
        run_prefix="t3_usage_e7_tinyconvnext18",
        changed_factor="model_architecture",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        model_family="task3_tinyconvnext18",
        run_model_token="tinyconvnext18",
    )


def gender_gem_p3_early_stopping_spec(
    parent_run_ids: Sequence[str],
) -> Task3ChildSpec:
    """Change only the Gender E6 checkpoint-selection policy."""
    return Task3ChildSpec(
        name="gender_gem_p3_early_stopping",
        target="gender",
        experiment_id="t3_gender_gem_p3_early_stopping",
        hypothesis_id="t3_gender_e8_early_stopping",
        artifact_dir="experiments/t3_gender_e8_early_stopping",
        run_prefix="t3_gender_e8_early_stopping",
        changed_factor="checkpoint_selection",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="experiments/t3_gender_e6_gem_p3",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_small_cnn_gem_p3",
        run_model_token="smallcnngem3",
        checkpoint_policy="best_validation_macro_f1",
        early_stopping_min_epoch=15,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    )


def usage_translation_2px_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only the accepted Usage E2 training translation."""
    return Task3ChildSpec(
        name="usage_translation_2px",
        target="usage",
        experiment_id="t3_usage_translation_2px_smallcnn",
        hypothesis_id="t3_usage_e8_translation",
        artifact_dir="experiments/t3_usage_e8_translation",
        run_prefix="t3_usage_e8_translation",
        changed_factor="training_translation",
        training_augmentation="translation_uniform_2px_p05",
        loss_name="effective_number_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
    )


def gender_semantic_filter_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only the Gender E6 fold-training row selection."""
    return Task3ChildSpec(
        name="gender_semantic_filter",
        target="gender",
        experiment_id="t3_gender_semantic_filter_gem_p3",
        hypothesis_id="t3_gender_e9_semantic_filter",
        artifact_dir="experiments/t3_gender_e9_semantic_filter",
        run_prefix="t3_gender_e9_semantic_filter",
        changed_factor="fold_training_semantic_filter",
        training_augmentation="none",
        loss_name="cross_entropy",
        parent_artifact_dir="experiments/t3_gender_e6_gem_p3",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_small_cnn_gem_p3",
        run_model_token="smallcnngem3",
        training_selection_strategy="gender_semantic_conflicts_v1",
    )


def usage_exception_balance_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Change only the Usage E2 fold-training example weights."""
    return Task3ChildSpec(
        name="usage_exception_balance",
        target="usage",
        experiment_id="t3_usage_exception_balance_smallcnn",
        hypothesis_id="t3_usage_e9_exception_balance",
        artifact_dir="experiments/t3_usage_e9_exception_balance",
        run_prefix="t3_usage_e9_exception_balance",
        changed_factor="fold_training_article_type_group_balance",
        training_augmentation="none",
        loss_name="effective_number_group_balanced_cross_entropy",
        parent_artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
        training_selection_strategy="usage_article_type_exception_balance_v1",
    )


def gender_audience_aux_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    """Add only three-way audience supervision to the exact Gender E6 parent."""
    return Task3ChildSpec(
        name="gender_audience_aux",
        target="gender",
        experiment_id="t3_gender_audience_aux_gem_p3",
        hypothesis_id="t3_gender_e10_audience_aux",
        artifact_dir="experiments/t3_gender_e10_audience_aux",
        run_prefix="t3_gender_e10_audience_aux",
        changed_factor="training_only_audience_auxiliary_head",
        training_augmentation="none",
        loss_name="gender_audience_auxiliary_cross_entropy",
        parent_artifact_dir="experiments/t3_gender_e6_gem_p3",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        model_family="task3_small_cnn_gem_p3_audience_aux",
        run_model_token="smallcnngem3aux3",
        auxiliary_target="gender_audience_3way",
        primary_loss_weight=GENDER_E10_PRIMARY_LOSS_WEIGHT,
        auxiliary_loss_weight=GENDER_E10_AUXILIARY_LOSS_WEIGHT,
    )


def effective_number_class_weights(
    counts: Sequence[int], *, beta: float = 0.999, cap: float = 5.0
) -> np.ndarray:
    """Return fold-only mean-one effective-number weights with a fixed upper cap."""
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("class counts must be a non-empty one-dimensional sequence")
    if (values < 0).any():
        raise ValueError("class counts cannot be negative")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be between zero and one")
    if cap <= 0:
        raise ValueError("class-weight cap must be positive")
    weights = np.zeros(len(values), dtype=np.float64)
    present = values > 0
    if not present.any():
        raise ValueError("at least one class must be present")
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, values[present]))
    weights[present] /= weights[present].mean()
    weights[present] = np.minimum(weights[present], cap)
    return weights


def latest_completed_parent_run_ids(
    target: Task3Target,
    *,
    output_root: str | Path,
    artifact_dir: str,
    run_prefix: str,
    run_model_token: str = "smallcnn",
) -> tuple[str, str, str, str, str]:
    """Find the newest complete registered parent for each canonical fold."""
    relative_dir = Path(artifact_dir)
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise ValueError("artifact directory must stay inside the Task 3 output root")
    target_dir = Path(output_root) / relative_dir / target
    latest: dict[int, str] = {}
    for run_dir in sorted(target_dir.glob(f"{run_prefix}_{target}_{run_model_token}_f*")):
        required = (
            run_dir / "config.json",
            run_dir / "final_epoch.pt",
            run_dir / "metrics.json",
            run_dir / "oof_predictions.csv",
            run_dir / "robustness.csv",
        )
        if not all(path.is_file() for path in required):
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        fold = int(metrics["validation_fold"])
        run_id = str(metrics["run_id"])
        if run_id != run_dir.name:
            raise ValueError(f"parent run folder and metrics disagree: {run_dir}")
        if str(metrics.get("target", target)) != target:
            raise ValueError(f"parent target metadata disagrees: {run_dir}")
        latest[fold] = run_id
    if set(latest) != set(range(5)):
        raise FileNotFoundError(
            f"expected completed {target} parent folds 0-4; found {sorted(latest)}"
        )
    return tuple(latest[fold] for fold in range(5))  # type: ignore[return-value]


def latest_completed_baseline_parent_run_ids(
    target: Task3Target, *, output_root: str | Path
) -> tuple[str, str, str, str, str]:
    """Find the newest complete baseline checkpoint for each canonical fold."""
    return latest_completed_parent_run_ids(
        target,
        output_root=output_root,
        artifact_dir="baseline",
        run_prefix="t3_baseline",
    )


def latest_completed_usage_e2_parent_run_ids(
    *, output_root: str | Path
) -> tuple[str, str, str, str, str]:
    """Find the accepted five-fold Usage E2 parent chain."""
    return latest_completed_parent_run_ids(
        "usage",
        output_root=output_root,
        artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        run_prefix="t3_usage_e2_class_balanced_ce",
    )


def latest_completed_gender_e6_parent_run_ids(
    *, output_root: str | Path
) -> tuple[str, str, str, str, str]:
    """Find the five completed Gender E6 GeM performance-benchmark folds."""
    return latest_completed_parent_run_ids(
        "gender",
        output_root=output_root,
        artifact_dir="experiments/t3_gender_e6_gem_p3",
        run_prefix="t3_gender_e6_gem_p3",
        run_model_token="smallcnngem3",
    )


def _spec(name: Task3ChildName, parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    if name == "gender_brightness":
        return gender_brightness_spec(parent_run_ids)
    if name == "usage_class_balanced":
        return usage_class_balanced_spec(parent_run_ids)
    if name == "gender_class_balanced":
        return gender_class_balanced_spec(parent_run_ids)
    if name == "usage_classifier_dropout":
        return usage_classifier_dropout_spec(parent_run_ids)
    if name == "gender_tinyresnet18_pm":
        return gender_tinyresnet18_pm_spec(parent_run_ids)
    if name == "usage_tinyresnet18_pm":
        return usage_tinyresnet18_pm_spec(parent_run_ids)
    if name == "gender_compact_blur_cnn":
        return gender_compact_blur_cnn_spec(parent_run_ids)
    if name == "usage_label_smoothing":
        return usage_label_smoothing_spec(parent_run_ids)
    if name == "gender_gem_p3":
        return gender_gem_p3_spec(parent_run_ids)
    if name == "usage_focal_gamma1":
        return usage_focal_gamma1_spec(parent_run_ids)
    if name == "gender_tinyhrnet20":
        return gender_tinyhrnet20_spec(parent_run_ids)
    if name == "usage_tinyconvnext18":
        return usage_tinyconvnext18_spec(parent_run_ids)
    if name == "gender_gem_p3_early_stopping":
        return gender_gem_p3_early_stopping_spec(parent_run_ids)
    if name == "usage_translation_2px":
        return usage_translation_2px_spec(parent_run_ids)
    if name == "gender_semantic_filter":
        return gender_semantic_filter_spec(parent_run_ids)
    if name == "usage_exception_balance":
        return usage_exception_balance_spec(parent_run_ids)
    if name == "gender_audience_aux":
        return gender_audience_aux_spec(parent_run_ids)
    raise ValueError(f"unsupported Task 3 child: {name}")


def check_task3_child_setup(
    name: Task3ChildName,
    *,
    parent_run_ids: Sequence[str],
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Check a locked child and its parent chain without an optimiser step."""
    from fashion.train.task3_baseline import check_task3_baseline_setup

    spec = _spec(name, parent_run_ids)
    baseline = check_task3_baseline_setup(spec.target, root=root, device_name=device_name)
    if spec.model_family in {
        "task3_small_cnn_gem_p3",
        "task3_tinyresnet18_pm",
        "task3_compact_blur_cnn",
        "task3_tinyhrnet20",
        "task3_tinyconvnext18",
        "task3_small_cnn_gem_p3_audience_aux",
    }:
        import torch

        from fashion.train.config import (
            Task3BaselineConfig,
            baseline_parameter_count,
            compact_blur_cnn_macs,
            compact_blur_cnn_parameter_count,
            gender_audience_aux_parameter_count,
            tinyconvnext18_macs,
            tinyconvnext18_parameter_count,
            tinyhrnet20_macs,
            tinyhrnet20_parameter_count,
            tinyresnet18_pm_macs,
            tinyresnet18_pm_parameter_count,
        )
        from fashion.train.model import (
            Task3CompactBlurCNN,
            Task3GeM3AudienceCNN,
            Task3GeM3CNN,
            Task3TinyConvNeXt18,
            Task3TinyHRNet20,
            Task3TinyResNet18PM,
        )

        config = Task3BaselineConfig(target=spec.target)
        device = torch.device(device_name)
        if spec.model_family == "task3_small_cnn_gem_p3":
            model = Task3GeM3CNN(config).to(device)
            parameter_count = baseline_parameter_count(spec.target)
            architecture_macs = None
        elif spec.model_family == "task3_small_cnn_gem_p3_audience_aux":
            model = Task3GeM3AudienceCNN(config).to(device)
            parameter_count = gender_audience_aux_parameter_count()
            architecture_macs = None
        elif spec.model_family == "task3_tinyresnet18_pm":
            model = Task3TinyResNet18PM(config).to(device)
            parameter_count = tinyresnet18_pm_parameter_count(spec.target)
            architecture_macs = tinyresnet18_pm_macs(spec.target)
        elif spec.model_family == "task3_compact_blur_cnn":
            model = Task3CompactBlurCNN(config).to(device)
            parameter_count = compact_blur_cnn_parameter_count(spec.target)
            architecture_macs = compact_blur_cnn_macs(spec.target)
        elif spec.model_family == "task3_tinyhrnet20":
            model = Task3TinyHRNet20(config).to(device)
            parameter_count = tinyhrnet20_parameter_count(spec.target)
            architecture_macs = tinyhrnet20_macs(spec.target)
        else:
            model = Task3TinyConvNeXt18(config).to(device)
            parameter_count = tinyconvnext18_parameter_count(spec.target)
            architecture_macs = tinyconvnext18_macs(spec.target)
        with torch.inference_mode():
            output = model(
                torch.zeros(2, 3, config.image_height, config.image_width, device=device)
            )
        if tuple(output.shape) != (2, config.num_classes):
            raise RuntimeError(f"unexpected child model output shape: {tuple(output.shape)}")
        if spec.auxiliary_target == "gender_audience_3way":
            from fashion.train.loss import GenderAudienceAuxiliaryCrossEntropy

            if device.type != "cuda":
                raise RuntimeError("E10 preflight requires CUDA for its memory check")
            primary_output, auxiliary_output = model.forward_with_auxiliary(  # type: ignore[attr-defined]
                torch.zeros(2, 3, config.image_height, config.image_width, device=device)
            )
            if tuple(primary_output.shape) != (2, 5):
                raise RuntimeError("E10 changed the fixed five-way Gender output shape")
            if tuple(auxiliary_output.shape) != (2, 3):
                raise RuntimeError("E10 audience helper output must have three logits")
            baseline["auxiliary_output_shape"] = list(auxiliary_output.shape)
            torch.cuda.reset_peak_memory_stats(device)
            memory_images = torch.zeros(
                config.batch_size,
                3,
                config.image_height,
                config.image_width,
                device=device,
            )
            memory_labels = torch.arange(config.batch_size, device=device) % 5
            memory_primary, memory_auxiliary = model.forward_with_auxiliary(memory_images)  # type: ignore[attr-defined]
            memory_loss = GenderAudienceAuxiliaryCrossEntropy(
                primary_weight=spec.primary_loss_weight,
                auxiliary_weight=spec.auxiliary_loss_weight,
            )(memory_primary, memory_auxiliary, memory_labels)
            memory_loss.backward()
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            model.zero_grad(set_to_none=True)
            del memory_images, memory_labels, memory_primary, memory_auxiliary, memory_loss
            torch.cuda.empty_cache()
            if peak_memory_bytes >= GENDER_E10_PEAK_MEMORY_LIMIT_BYTES:
                raise RuntimeError(
                    "E10 zero-step memory check exceeded the frozen 2 GiB limit: "
                    f"{peak_memory_bytes:,} bytes"
                )
            baseline["zero_step_peak_memory_bytes"] = peak_memory_bytes
            baseline["peak_memory_limit_bytes"] = GENDER_E10_PEAK_MEMORY_LIMIT_BYTES
        baseline["parameter_count"] = parameter_count
        baseline["architecture_macs"] = architecture_macs
    baseline["model_family"] = spec.model_family
    baseline["label_smoothing"] = spec.label_smoothing
    baseline["focal_gamma"] = spec.focal_gamma
    baseline["training_selection_strategy"] = spec.training_selection_strategy
    baseline["auxiliary_target"] = spec.auxiliary_target
    baseline["primary_loss_weight"] = spec.primary_loss_weight
    baseline["auxiliary_loss_weight"] = spec.auxiliary_loss_weight
    baseline["gender_deterministic_audit_required"] = (
        spec.training_selection_strategy == "gender_semantic_conflicts_v1"
    )
    baseline["gender_human_rating_gate_required"] = False
    return {
        **baseline,
        "child": spec.to_dict(),
        "changed_factor": spec.changed_factor,
        "optimizer_steps": 0,
        "ready": True,
    }


def audit_completed_registry_rows(
    registry_path: str | Path, run_ids: Sequence[str]
) -> dict[str, object]:
    """Require one complete, hashed registry row for every returned fold run."""
    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Task 3 registry was not persisted: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_artifacts = (
        "config_path",
        "history_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "prediction_path",
        "prediction_sha256",
        "metrics_json",
    )
    audited: list[dict[str, str]] = []
    for run_id in run_ids:
        matches = [row for row in rows if row.get("run_id") == run_id]
        if len(matches) != 1:
            raise RuntimeError(f"expected one registry row for {run_id}; found {len(matches)}")
        row = matches[0]
        if row.get("status") != "complete":
            raise RuntimeError(f"registry row is not complete: {run_id}")
        missing = [field for field in required_artifacts if not row.get(field)]
        if missing:
            raise RuntimeError(f"registry row {run_id} lacks {missing}")
        audited.append(row)
    return {
        "registry_path": str(path),
        "run_ids": list(run_ids),
        "completed_rows": len(audited),
        "ready": len(audited) == len(run_ids),
    }


def run_task3_child_cv(
    name: Task3ChildName,
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Sequence[int] = range(5),
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Run one exact five-fold child while preserving its baseline controls."""
    from fashion.train.task3_baseline import run_task3_baseline_cv

    spec = _spec(name, parent_run_ids)
    return run_task3_baseline_cv(
        spec.target,
        folds=folds,
        output_root=output_root,
        registry_path=registry_path,
        registry_mirrors=registry_mirrors,
        root=root,
        device_name=device_name,
        child_spec=spec,
    )
