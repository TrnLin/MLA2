"""Reusable training contracts for the assignment tasks."""

from fashion.train.config import Task3BaselineConfig, baseline_parameter_count, config_digest
from fashion.train.registry import RunRegistry
from fashion.train.task3_experiments import (
    Task3ChildSpec,
    check_task3_child_setup,
    latest_completed_baseline_parent_run_ids,
    run_task3_child_cv,
)

__all__ = [
    "RunRegistry",
    "Task3BaselineConfig",
    "Task3ChildSpec",
    "baseline_parameter_count",
    "check_task3_child_setup",
    "config_digest",
    "latest_completed_baseline_parent_run_ids",
    "run_task3_child_cv",
]
