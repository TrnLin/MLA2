"""Compatibility facade for Task 1 experiment controllers and figures."""

from fashion.task1.classical_experiments import (
    Task1ClassicalExperimentResult,
    Task1ClassicalSelection,
    run_task1_classical_experiment,
)
from fashion.task1.cnn_experiments import Task1ExperimentResult, run_task1_experiment
from fashion.task1.plotting import write_task1_comparison_figure, write_task1_confusion_figure

__all__ = [
    "Task1ExperimentResult",
    "Task1ClassicalSelection",
    "Task1ClassicalExperimentResult",
    "run_task1_experiment",
    "run_task1_classical_experiment",
    "write_task1_comparison_figure",
    "write_task1_confusion_figure",
]
