"""Task 1 article-type classification tools."""

from fashion.task1.dataset import Task1TorchDataset, get_task1_fold_rows
from fashion.task1.experiments import (
    Task1ExperimentResult,
    run_task1_experiment,
    write_task1_comparison_figure,
    write_task1_confusion_figure,
)
from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    build_prediction_frame,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)
from fashion.task1.models import Task1ModelConfig, Task1SmallCNN, count_trainable_parameters
from fashion.task1.preprocessing import (
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_CONTROL_PREPROCESSING,
    Task1ImageTransform,
    Task1Normalization,
    Task1PreprocessingConfig,
    build_task1_training_transform,
    build_task1_validation_transform,
    fit_task1_normalization,
)
from fashion.task1.training import (
    Task1FoldResult,
    Task1TrainConfig,
    select_training_device,
    train_task1_fold,
)

__all__ = [
    "DEFAULT_TASK1_PREPROCESSING",
    "TASK1_CONTROL_PREPROCESSING",
    "Task1ImageTransform",
    "Task1Normalization",
    "Task1PreprocessingConfig",
    "build_task1_training_transform",
    "build_task1_validation_transform",
    "fit_task1_normalization",
    "Task1TorchDataset",
    "get_task1_fold_rows",
    "Task1ExperimentResult",
    "run_task1_experiment",
    "write_task1_comparison_figure",
    "write_task1_confusion_figure",
    "classification_metrics",
    "per_class_metrics",
    "build_prediction_frame",
    "validate_oof_predictions",
    "aggregate_fold_metrics",
    "Task1ModelConfig",
    "Task1SmallCNN",
    "count_trainable_parameters",
    "Task1FoldResult",
    "Task1TrainConfig",
    "select_training_device",
    "train_task1_fold",
]
