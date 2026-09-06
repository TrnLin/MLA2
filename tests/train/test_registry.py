from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from fashion.train import registry as registry_module
from fashion.train.registry import (
    RUN_COLUMNS,
    DuplicateRunError,
    ImmutableRunError,
    RunRecord,
    RunRegistry,
    new_run_id,
    tracked_run,
)

DIGESTS = {
    "config_sha256": "a" * 64,
    "split_sha256": "b" * 64,
    "label_map_sha256": "c" * 64,
    "implementation_sha256": "d" * 64,
}

_ORIGINAL_ATOMIC_WRITE_CSV = registry_module.atomic_write_csv
_WRITE_STARTED: object | None = None


def _write_after_contention_window(*args: object, **kwargs: object) -> Path:
    """Hold the first writer's registry lock while another process attempts append."""
    assert _WRITE_STARTED is not None
    _WRITE_STARTED.set()
    time.sleep(0.5)
    return _ORIGINAL_ATOMIC_WRITE_CSV(*args, **kwargs)


def _record(run_id: str = "c1-f0-s2753-test") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment_id="c1-screen",
        fold=0,
        seed=2753,
        model_family="smallcnn",
        **DIGESTS,
    )


def _append_in_process(
    path: Path,
    run_id: str,
    start_barrier: object,
    write_started: object,
    wait_for_first_write: bool,
) -> None:
    if not wait_for_first_write:
        global _WRITE_STARTED
        _WRITE_STARTED = write_started
        registry_module.atomic_write_csv = _write_after_contention_window
    start_barrier.wait(timeout=5)
    if wait_for_first_write:
        if not write_started.wait(timeout=5):
            raise TimeoutError("first writer did not reach the contention window")
    RunRegistry(path).append(_record(run_id))


def test_tracked_run_records_completed_metrics(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")

    with tracked_run(registry, _record()) as run:
        run.epochs_completed = 2
        run.primary_metric_name = "macro_f1"
        run.primary_metric_value = 0.625
        run.metrics = {"macro_f1": 0.625, "loss": 0.8}

    rows = registry.read()
    assert tuple(rows.columns) == RUN_COLUMNS
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "completed"
    assert rows.loc[0, "primary_metric_value"] == "0.625"
    assert rows.loc[0, "metrics"] == '{"loss":0.8,"macro_f1":0.625}'
    assert rows.loc[0, "git_commit"]
    assert rows.loc[0, "finished_at_utc"].endswith("Z")


def test_tracked_run_records_failure_and_reraises(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(RuntimeError, match="out of memory"):
        with tracked_run(registry, _record()) as run:
            run.epochs_completed = 1
            raise RuntimeError("out of memory")

    rows = registry.read()
    assert rows.loc[0, "status"] == "failed"
    assert rows.loc[0, "error_type"] == "RuntimeError"
    assert rows.loc[0, "error_message"] == "out of memory"


def test_duplicate_ids_and_final_rewrites_are_rejected(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    with tracked_run(registry, _record()):
        pass

    with pytest.raises(DuplicateRunError):
        registry.append(_record())

    finished = _record()
    finished.status = "completed"
    finished.finished_at_utc = "2026-08-26T00:00:00Z"
    with pytest.raises(ImmutableRunError):
        registry.finalize(finished)


def test_concurrent_appends_keep_both_run_rows(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    start_barrier = context.Barrier(3)
    write_started = context.Event()
    path = tmp_path / "runs.csv"
    processes = [
        context.Process(
            target=_append_in_process,
            args=(path, "concurrent-0", start_barrier, write_started, False),
        ),
        context.Process(
            target=_append_in_process,
            args=(path, "concurrent-1", start_barrier, write_started, True),
        )
    ]
    for process in processes:
        process.start()

    start_barrier.wait(timeout=5)
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert set(RunRegistry(path).read()["run_id"]) == {"concurrent-0", "concurrent-1"}


def test_start_identity_cannot_change(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.csv")
    record = _record()
    registry.append(record)
    record.fold = 1
    record.status = "completed"
    record.finished_at_utc = "2026-08-26T00:00:00Z"

    with pytest.raises(ImmutableRunError, match="fold"):
        registry.finalize(record)


def test_new_run_ids_are_readable_and_unique() -> None:
    first = new_run_id("C1 screen", fold=2, seed=2753)
    second = new_run_id("C1 screen", fold=2, seed=2753)

    assert first.startswith("C1-screen-f2-s2753-")
    assert first != second


def test_record_rejects_invalid_provenance_hash() -> None:
    record = _record()
    record.split_sha256 = "not-a-digest"
    with pytest.raises(ValueError, match="split_sha256"):
        record.to_row()
