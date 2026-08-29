"""Append-first, atomic run registry used by every training execution."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the project and Colab run on Linux
    fcntl = None  # type: ignore[assignment]


REGISTRY_COLUMNS = (
    "run_id",
    "experiment_id",
    "hypothesis_id",
    "parent_run_ids",
    "task",
    "target",
    "validation_fold",
    "seed",
    "status",
    "debug",
    "scratch",
    "submission_eligible",
    "timestamp_start",
    "timestamp_end",
    "config_hash",
    "config_path",
    "split_digest",
    "label_map_digest",
    "training_product_count",
    "validation_product_count",
    "training_family_count",
    "validation_family_count",
    "model_family",
    "parameter_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_path",
    "prediction_sha256",
    "history_path",
    "metrics_json",
    "train_seconds",
    "peak_memory_bytes",
    "checkpoint_bytes",
    "environment_json",
    "exception_type",
    "exception_message",
    "last_completed_stage",
)

FINAL_STATUSES = {"complete", "failed"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


class RunRegistry:
    """Keep one durable row per execution, including failed executions."""

    def __init__(self, path: str | Path, mirrors: Sequence[str | Path] = ()) -> None:
        self.path = Path(path)
        self.mirrors = tuple(Path(path) for path in mirrors)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
                raise ValueError("existing run registry has an unexpected schema")
            return [dict(row) for row in reader]

    def _write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {column: _stringify(row.get(column)) for column in REGISTRY_COLUMNS}
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        for mirror in self.mirrors:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            temporary_mirror = mirror.with_suffix(mirror.suffix + ".tmp")
            temporary_mirror.write_bytes(self.path.read_bytes())
            os.replace(temporary_mirror, mirror)

    def start(self, record: Mapping[str, Any]) -> dict[str, str]:
        """Append a running row before the first optimiser step."""
        run_id = _stringify(record.get("run_id"))
        if not run_id:
            raise ValueError("run_id is required")
        if record.get("status") not in (None, "", "running"):
            raise ValueError("a new registry row must start in running status")
        with self._locked():
            rows = self._read_rows()
            if any(row["run_id"] == run_id for row in rows):
                raise ValueError(f"run_id already exists: {run_id}")
            row = {column: "" for column in REGISTRY_COLUMNS}
            row.update({key: _stringify(value) for key, value in record.items() if key in row})
            row["run_id"] = run_id
            row["status"] = "running"
            row["timestamp_start"] = row["timestamp_start"] or _now()
            rows.append(row)
            self._write_rows(rows)
            return row

    def update(self, run_id: str, changes: Mapping[str, Any]) -> dict[str, str]:
        """Atomically replace fields in one existing execution row."""
        forbidden = {"run_id", "timestamp_start"}.intersection(changes)
        if forbidden:
            raise ValueError(f"immutable registry fields cannot change: {sorted(forbidden)}")
        unknown = set(changes).difference(REGISTRY_COLUMNS)
        if unknown:
            raise ValueError(f"unknown registry fields: {sorted(unknown)}")
        with self._locked():
            rows = self._read_rows()
            matches = [index for index, row in enumerate(rows) if row["run_id"] == run_id]
            if len(matches) != 1:
                raise ValueError(f"expected one registry row for {run_id}, found {len(matches)}")
            index = matches[0]
            current = rows[index]
            if current["status"] in FINAL_STATUSES:
                raise ValueError(f"final registry row cannot change: {run_id}")
            current.update({key: _stringify(value) for key, value in changes.items()})
            rows[index] = current
            self._write_rows(rows)
            return current

    def complete(self, run_id: str, changes: Mapping[str, Any]) -> dict[str, str]:
        payload = dict(changes)
        payload.update(status="complete", timestamp_end=_now())
        return self.update(run_id, payload)

    def fail(
        self,
        run_id: str,
        error: BaseException,
        *,
        last_completed_stage: str,
    ) -> dict[str, str]:
        return self.update(
            run_id,
            {
                "status": "failed",
                "timestamp_end": _now(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "last_completed_stage": last_completed_stage,
            },
        )
