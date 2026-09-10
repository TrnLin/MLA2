"""Explicit operator recovery for training processes terminated outside Python."""

from __future__ import annotations

from fashion.train.registry import RunRegistry


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
    registry.interrupt(run_id, reason=reason.strip())
