from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from fashion.train.resource_worker import WorkerResourceError, run_supervised


# Module-level functions are intentional: the production worker uses spawn.
def _return_pid():
    return os.getpid()


def _hang(marker):
    Path(marker).write_text(str(os.getpid()))
    time.sleep(30)
    Path(marker + ".finished").write_text("should never happen")


def _allocate_too_much():
    bytearray(2 * 1024**3)


def _crash():
    os._exit(17)


def test_success_runs_in_a_fresh_process_and_reaps_it():
    pid, seconds = run_supervised(_return_pid, {}, seconds=10, memory_bytes=512 * 1024**2)
    assert pid != os.getpid()
    assert 0 < seconds < 10
    assert pid not in [p.pid for p in multiprocessing.active_children()]


def test_wall_deadline_kills_an_unresponsive_worker(tmp_path):
    marker = tmp_path / "started"
    started = time.monotonic()
    with pytest.raises(WorkerResourceError, match="wall-time limit"):
        run_supervised(_hang, {"marker": str(marker)}, seconds=2, memory_bytes=512 * 1024**2)
    assert time.monotonic() - started < 5
    assert marker.exists(), "test must reach the worker body, not just time out during startup"
    pid = int(marker.read_text())
    assert pid not in [p.pid for p in multiprocessing.active_children()]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not Path(str(marker) + ".finished").exists()


def test_hard_memory_cap_rejects_allocation_before_it_can_succeed():
    with pytest.raises(WorkerResourceError, match="MemoryError"):
        run_supervised(_allocate_too_much, {}, seconds=10, memory_bytes=512 * 1024**2)


def test_abrupt_worker_exit_is_never_success():
    with pytest.raises(WorkerResourceError, match="exit code 17"):
        run_supervised(_crash, {}, seconds=10, memory_bytes=512 * 1024**2)
