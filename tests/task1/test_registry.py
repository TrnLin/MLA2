from __future__ import annotations

from pathlib import Path

import pytest

from fashion.task1.registry import Task1RunRegistry
from fashion.train.registry import RunRecord
from fashion.train.registry import RunRegistry as SharedRunRegistry


def _record(run_id: str, *, task: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment_id="registry-isolation",
        fold=0,
        seed=2753,
        config_sha256="a" * 64,
        split_sha256="b" * 64,
        label_map_sha256="c" * 64,
        implementation_sha256="d" * 64,
        task=task,
    )


def test_task1_and_shared_task2_views_do_not_mix_rows(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    task1 = Task1RunRegistry(path)
    task2 = SharedRunRegistry(path)

    task1.append(_record("task1-run", task="task1"))
    task2.append(_record("task2-run", task="task2"))

    assert task1.read().run_id.tolist() == ["task1-run"]
    assert task2.read().run_id.tolist() == ["task2-run"]
    assert {row["task"] for row in task1._read_rows()} == {"task1", "task2"}


@pytest.mark.parametrize("task", ["task2", "task3", "task4", ""])
def test_task1_registry_rejects_other_task_labels(tmp_path: Path, task: str) -> None:
    registry = Task1RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(ValueError, match="task must be task1"):
        registry.append(_record("wrong-task", task=task))

    assert not (tmp_path / "runs.csv").exists()
