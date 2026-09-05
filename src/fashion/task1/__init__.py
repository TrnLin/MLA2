"""Task 1 article-type classification tools."""

from fashion.task1.analysis import (
    Task1ProblemProfile,
    build_task1_confusion_pairs,
    build_task1_decision_evidence,
    build_task1_problem_profile,
    build_task1_weak_class_table,
)
from fashion.task1.classical_experiments import (
    Task1ClassicalExperimentResult,
    Task1ClassicalSelection,
    run_task1_classical_experiment,
)
from fashion.task1.classical_features import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    TASK1_HOG_SPECS,
    Task1HogSpec,
)
from fashion.task1.classical_models import (
    TASK1_KNN_GRID,
    TASK1_SVM_GRID,
    Task1KNNConfig,
    Task1LinearSVMConfig,
)
from fashion.task1.classical_training import (
    Task1ClassicalFoldResult,
    Task1ClassicalRunConfig,
    run_task1_classical_fold,
)
from fashion.task1.cnn_experiments import (
    Task1ExperimentResult,
    run_task1_experiment,
)
from fashion.task1.weighted_experiments import run_task1_weighted_experiment
from fashion.task1.dataset import Task1TorchDataset, get_task1_fold_rows
from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    build_prediction_frame,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)
from fashion.task1.image_contract import (
    TASK1_IMAGE_SIZE,
    TASK1_PAD_COLOR,
    TASK1_TENSOR_SHAPE,
)
from fashion.task1.models import Task1ModelConfig, Task1SmallCNN, count_trainable_parameters
from fashion.task1.plotting import (
    write_task1_comparison_figure,
    write_task1_confusion_figure,
    write_task1_learning_curve_figure,
)
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
from fashion.task1.losses import (
    TASK1_GENTLE_WEIGHTED_LOSS,
    TASK1_UNWEIGHTED_LOSS,
    Task1LossConfig,
    Task1LossWeights,
    build_task1_loss_weights,
)
from fashion.task1.candidates import (
    TASK1_GENTLE_WEIGHTED_CANDIDATE,
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
    Task1CnnCandidate,
)

__all__ = [
    "DEFAULT_TASK1_PREPROCESSING",
    "Task1ProblemProfile",
    "build_task1_problem_profile",
    "build_task1_decision_evidence",
    "build_task1_weak_class_table",
    "build_task1_confusion_pairs",
    "TASK1_CONTROL_PREPROCESSING",
    "TASK1_IMAGE_SIZE",
    "TASK1_PAD_COLOR",
    "TASK1_TENSOR_SHAPE",
    "Task1ImageTransform",
    "Task1Normalization",
    "Task1PreprocessingConfig",
    "build_task1_training_transform",
    "build_task1_validation_transform",
    "fit_task1_normalization",
    "Task1TorchDataset",
    "get_task1_fold_rows",
    "TASK1_HOG_COARSE",
    "TASK1_HOG_FINE",
    "TASK1_HOG_SPECS",
    "TASK1_KNN_GRID",
    "TASK1_SVM_GRID",
    "Task1HogSpec",
    "Task1KNNConfig",
    "Task1LinearSVMConfig",
    "Task1ClassicalRunConfig",
    "Task1ClassicalFoldResult",
    "run_task1_classical_fold",
    "Task1ClassicalExperimentResult",
    "Task1ClassicalSelection",
    "run_task1_classical_experiment",
    "Task1ExperimentResult",
    "run_task1_experiment",
    "run_task1_weighted_experiment",
    "write_task1_comparison_figure",
    "write_task1_confusion_figure",
    "write_task1_learning_curve_figure",
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
    "Task1LossConfig",
    "Task1LossWeights",
    "build_task1_loss_weights",
    "TASK1_UNWEIGHTED_LOSS",
    "TASK1_GENTLE_WEIGHTED_LOSS",
    "Task1CnnCandidate",
    "TASK1_NO_AUG_CANDIDATE",
    "TASK1_MILD_AUG_CANDIDATE",
    "TASK1_GENTLE_WEIGHTED_CANDIDATE",
]
