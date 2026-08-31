"""Explicit operator recovery for training processes terminated outside Python."""

from __future__ import annotations

from datetime import UTC, datetime

from fashion.train.artifacts import atomic_write_csv
from fashion.train.registry import ImmutableRunError, RegistryError, RunRegistry


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def interrupt_orphaned_run(
    registry: RunRegistry,
    run_id: str,
    *,
    reason: str,
) -> None:
    """Close one externally terminated run without rewriting its provenance."""
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not reason.strip():
        raise ValueError("orphan interruption reason must be non-empty")
    frame = registry.read()
    matches = frame.index[frame["run_id"] == run_id].tolist()
    if not matches:
        raise RegistryError(f"run_id does not exist: {run_id}")
    index = matches[0]
    if frame.loc[index, "status"] != "running":
        raise ImmutableRunError(f"run is already final: {run_id}")
    frame.loc[index, "status"] = "interrupted"
    frame.loc[index, "error_type"] = "ExternalProcessTermination"
    frame.loc[index, "error_message"] = reason.strip()
    frame.loc[index, "finished_at_utc"] = _utc_now()
    atomic_write_csv(registry.path, frame)
