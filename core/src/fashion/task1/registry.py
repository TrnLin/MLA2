"""Task 1 registry view over the shared experiment ledger.

The team-owned registry keeps its Task 2 and Task 3 behaviour unchanged.  This
adapter supplies the Task 1 record lifecycle while preserving every other row
in the same ``results/runs.csv`` file.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from fashion.train.registry import (
    IMMUTABLE_START_FIELDS,
    RUN_COLUMNS,
    TASK2_RUN_COLUMNS,
    TERMINAL_STATUSES,
    DuplicateRunError,
    ImmutableRunError,
    RegistryError,
    RunRecord,
)
from fashion.train.registry import RunRegistry as SharedRunRegistry

TASK1 = "task1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Task1RunRegistry(SharedRunRegistry):
    """Read and write only Task 1 records without changing the shared class."""

    def read(self) -> pd.DataFrame:
        """Return the Task 1 view while retaining the shared column contract."""
        with self._locked():
            rows = self._read_rows()
        task_rows = [
            {column: row[column] for column in TASK2_RUN_COLUMNS}
            for row in rows
            if row["task"] == TASK1
        ]
        return pd.DataFrame(task_rows, columns=TASK2_RUN_COLUMNS)

    def append(self, record: RunRecord) -> None:
        """Append one new running Task 1 row without disturbing other tasks."""
        if record.task != TASK1:
            raise ValueError("task must be task1 for append")
        if record.status != "running":
            raise ValueError("new registry rows must start with status='running'")
        record_row = record.to_row()
        with self._locked():
            rows = self._read_rows()
            if any(row["run_id"] == record.run_id for row in rows):
                raise DuplicateRunError(f"run_id already exists: {record.run_id}")
            merged = {column: "" for column in RUN_COLUMNS}
            merged.update(record_row)
            self._write_rows([*rows, merged])

    def finalize(self, record: RunRecord) -> None:
        """Finalize one running Task 1 row while preserving its identity."""
        if record.task != TASK1:
            raise ValueError("task must be task1 for finalize")
        if record.status not in TERMINAL_STATUSES:
            raise ValueError("finalized run must have a terminal status")
        with self._locked():
            rows = self._read_rows()
            matches = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == record.run_id and row["task"] == TASK1
            ]
            if not matches:
                raise RegistryError(f"run_id does not exist: {record.run_id}")
            index = matches[0]
            current = rows[index]
            if current["status"] != "running":
                raise ImmutableRunError(f"run is already final: {record.run_id}")
            new_row = record.to_row()
            changed = [
                name for name in IMMUTABLE_START_FIELDS if current[name] != new_row[name]
            ]
            if changed:
                raise ImmutableRunError(
                    f"cannot change run identity after start: {', '.join(changed)}"
                )
            rows[index].update(new_row)
            self._write_rows(rows)

    def interrupt(self, run_id: str, *, reason: str) -> None:
        """Mark one running Task 1 row interrupted without changing other tasks."""
        with self._locked():
            rows = self._read_rows()
            matches = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == run_id and row["task"] == TASK1
            ]
            if not matches:
                raise RegistryError(f"run_id does not exist: {run_id}")
            index = matches[0]
            if rows[index]["status"] != "running":
                raise ImmutableRunError(f"run is already final: {run_id}")
            rows[index].update(
                {
                    "status": "interrupted",
                    "error_type": "ExternalProcessTermination",
                    "error_message": reason,
                    "finished_at_utc": _utc_now(),
                }
            )
            self._write_rows(rows)
