"""Run a fold in a disposable process with an externally enforced deadline."""

from __future__ import annotations

import multiprocessing
import os
import resource
import time
import traceback
from typing import Any, Callable, Mapping


class WorkerResourceError(RuntimeError):
    """The worker exceeded its budget, failed, or exited without a result."""


def _worker_entry(connection, function, kwargs, memory_bytes):
    try:
        # RLIMIT_RSS is ignored on Linux. RLIMIT_AS is an actual allocation cap.
        # It also counts mappings, so this is stricter than a resident-memory cap.
        _, inherited_hard = resource.getrlimit(resource.RLIMIT_AS)
        hard = (
            memory_bytes
            if inherited_hard == resource.RLIM_INFINITY
            else min(memory_bytes, inherited_hard)
        )
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
        result = function(**kwargs)
        connection.send(("result", result))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"))
    finally:
        connection.close()


def run_supervised(
    function: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    seconds: float,
    memory_bytes: int,
) -> tuple[Any, float]:
    """Kill and reap a stuck worker; never return partial output as success.

    Spawn gives every fold a fresh peak-memory counter and avoids forking the
    notebook's threaded numerical runtime. The deadline covers startup, fitting,
    diagnostics, artifact writes, and the worker's exit. Linux address-space
    limits stop allocation even while native solver code is running.
    """
    if os.name != "posix" or not hasattr(resource, "RLIMIT_AS"):
        raise WorkerResourceError("enforced U2 memory limits require a POSIX RLIMIT_AS runtime")
    if seconds <= 0 or memory_bytes <= 0:
        raise ValueError("worker budgets must be positive")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry, args=(send, function, dict(kwargs), memory_bytes), daemon=True
    )
    started = time.monotonic()
    try:
        process.start()
        send.close()
        message = None
        while True:
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise WorkerResourceError(f"worker exceeded its {seconds:g}-second wall-time limit")
            if receive.poll(min(0.05, remaining)):
                try:
                    message = receive.recv()
                except EOFError:
                    break
                break
            if not process.is_alive():
                break
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise WorkerResourceError(f"worker exceeded its {seconds:g}-second wall-time limit")
        process.join(timeout=remaining)
        elapsed = time.monotonic() - started
        if process.is_alive() or elapsed > seconds:
            raise WorkerResourceError(f"worker exceeded its {seconds:g}-second wall-time limit")
        if process.exitcode != 0 or message is None:
            raise WorkerResourceError(
                f"worker exited without a valid result (exit code {process.exitcode})"
            )
        kind, payload = message
        if kind != "result":
            raise WorkerResourceError(payload)
        return payload, elapsed
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
        receive.close()
        send.close()
