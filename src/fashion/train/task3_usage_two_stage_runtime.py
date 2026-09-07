"""Watch a disposable GPU fold, including native calls and its data-loader children."""

import os
import signal
import subprocess
import time
from pathlib import Path


class FoldResourceError(RuntimeError):
    """A fold failed, exceeded its deadline, or used too much resident memory."""


def process_group_rss(group):
    """Linux RSS for the worker and all processes in its private process group."""
    total = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Names may contain spaces and parentheses; fields start after the final ')'.
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                total += int((path.parent / "statm").read_text().split()[1]) * page_size
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def run_fold_process(command, *, cwd, log_path, seconds, memory_bytes, env=None):
    """Enforce wall time and monitor process-tree RSS; kill the entire group on exit.

    RSS is sampled every 0.2 seconds, so a brief overshoot is possible. The worker
    separately caps PyTorch's GPU allocator. RLIMIT_AS is intentionally not used:
    CUDA's virtual address mappings are much larger than its resident allocations.
    """
    if os.name != "posix" or not Path("/proc/self/statm").is_file():
        raise FoldResourceError("Use the Linux Colab runtime for the fold watchdog")
    if seconds <= 0 or memory_bytes <= 0:
        raise ValueError("Fold budgets must be positive")
    log_path = Path(log_path)
    started, peak, process = time.monotonic(), 0, None
    with log_path.open("w") as writer, log_path.open() as reader:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=writer,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                output = reader.read()
                if output:
                    print(output, end="", flush=True)
                elapsed = time.monotonic() - started
                peak = max(peak, process_group_rss(process.pid))
                if elapsed > seconds:
                    raise FoldResourceError(f"Fold exceeded its {seconds:g}-second deadline")
                if peak > memory_bytes:
                    raise FoldResourceError(f"Fold process RSS exceeded {memory_bytes} bytes")
                code = process.poll()
                if code is not None:
                    print(reader.read(), end="", flush=True)
                    if code != 0:
                        raise FoldResourceError(f"Fold worker failed (exit {code}); see {log_path}")
                    return {"fold_wall_seconds": elapsed, "peak_host_memory_bytes": peak}
                time.sleep(min(0.2, max(0.001, seconds - elapsed)))
        finally:
            if process is not None:
                # A successful main process can still leave data-loader children behind.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
