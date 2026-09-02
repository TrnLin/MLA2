"""Compatibility contracts for the Task 1 experiment facade."""

from __future__ import annotations

import fashion.task1.classical_experiments as classical_experiments
import fashion.task1.cnn_experiments as cnn_experiments
import fashion.task1.experiments as experiments
import fashion.task1.plotting as plotting


def test_experiment_facade_reexports_public_controllers() -> None:
    """Facade callers must keep the controller and plotting object identities."""
    assert experiments.run_task1_experiment is cnn_experiments.run_task1_experiment
    assert (
        experiments.run_task1_classical_experiment
        is classical_experiments.run_task1_classical_experiment
    )
    assert experiments.write_task1_confusion_figure is plotting.write_task1_confusion_figure
