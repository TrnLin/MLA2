"""Reusable training contracts for the assignment tasks."""

from fashion.train.config import Task3BaselineConfig, baseline_parameter_count, config_digest
from fashion.train.registry import RunRegistry

__all__ = [
    "RunRegistry",
    "Task3BaselineConfig",
    "baseline_parameter_count",
    "config_digest",
]
