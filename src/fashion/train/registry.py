"""Append-once run registry with explicit lifecycle and provenance fields."""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from fashion.config import RUNS_CSV
from fashion.train.artifacts import atomic_write_csv, canonical_json_bytes

RUN_STATUSES = frozenset({"running", "completed", "failed", "interrupted"})
TERMINAL_STATUSES = RUN_STATUSES - {"running"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_TRANSIENT_WRITE_ERRORS = {5, 32}
_REGISTRY_WRITE_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.04, 0.08)

TASK2_RUN_COLUMNS = (
    "run_id",
    "task",
    "stage",
    "experiment_id",
    "model_family",
    "benchmark_only",
    "final_eligible",
    "scratch",
    "fold",
    "seed",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "split_sha256",
    "label_map_sha256",
    "implementation_sha256",
    "transform_id",
    "loss_id",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "primary_metric_name",
    "primary_metric_value",
    "metrics",
    "runtime_seconds",
    "peak_vram_mb",
    "parameter_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_path",
    "prediction_sha256",
    "history_path",
    "history_sha256",
    "status",
    "error_type",
    "error_message",
    "started_at_utc",
    "finished_at_utc",
    "runtime",
)

TASK4_RUN_COLUMNS = (
    "schema_version",
    "run_id",
    "parent_run_id",
    "started_at_utc",
    "completed_at_utc",
    "task",
    "run_kind",
    "status",
    "fold",
    "method",
    "architecture",
    "objective",
    "source_policy",
    "pretrained",
    "weight_origin",
    "deployment_eligibility",
    "seed",
    "embedding_dim",
    "planned_epochs",
    "selected_epoch",
    "config_hash",
    "split_fingerprint",
    "git_commit",
    "dirty_tree",
    "parameter_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "development_winner_score",
    "cross_source_score",
    "source_robustness_ratio",
    "protocol_b_recall_at_10",
    "p95_end_to_end_seconds",
    "index_bytes",
    "evidence_manifest_path",
    "error_type",
    "error_message",
)

RUN_COLUMNS = TASK2_RUN_COLUMNS + tuple(
    column for column in TASK4_RUN_COLUMNS if column not in TASK2_RUN_COLUMNS
)

RUN_KINDS = frozenset({"smoke", "candidate", "benchmark", "stability", "final_refit"})
TASK4_STATUSES = frozenset({"running", "completed", "failed", "cancelled"})
DEPLOYMENT_ELIGIBILITIES = frozenset({"eligible", "comparison_only"})
TASK4_SCHEMA_VERSION = "1"

IMMUTABLE_START_FIELDS = (
    "run_id",
    "task",
    "stage",
    "experiment_id",
    "model_family",
    "benchmark_only",
    "final_eligible",
    "scratch",
    "fold",
    "seed",
    "config_sha256",
    "split_sha256",
    "label_map_sha256",
    "implementation_sha256",
    "transform_id",
    "loss_id",
)

_TASK4_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "parent_run_id",
        "started_at_utc",
        "task",
        "run_kind",
        "fold",
        "method",
        "architecture",
        "objective",
        "source_policy",
        "pretrained",
        "weight_origin",
        "deployment_eligibility",
        "seed",
        "embedding_dim",
        "planned_epochs",
        "config_hash",
        "split_fingerprint",
        "git_commit",
        "dirty_tree",
    }
)
_TASK4_NONEMPTY_TEXT_FIELDS = (
    "run_id",
    "started_at_utc",
    "task",
    "method",
    "architecture",
    "objective",
    "source_policy",
    "weight_origin",
)
_TASK4_OPTIONAL_NONNEGATIVE_INTS = ("selected_epoch", "parameter_count", "index_bytes")
_TASK4_OPTIONAL_FINITE_FLOATS = (
    "development_winner_score",
    "cross_source_score",
    "source_robustness_ratio",
    "protocol_b_recall_at_10",
    "p95_end_to_end_seconds",
)


class RegistryError(RuntimeError):
    """Base error for a malformed or unsafe registry operation."""


class RegistrySchemaError(RegistryError):
    """Raised when an existing registry does not have the fixed schema."""


class DuplicateRunError(RegistryError):
    """Raised when a run ID already exists."""


class ImmutableRunError(RegistryError):
    """Raised when code attempts to rewrite a final run or its identity."""


class RunRegistryError(ValueError):
    """Raised when a Task 4 registry operation breaks its run contract."""


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _registry_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_path(path).open("a+b") as lock_handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _read_union_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = tuple(reader.fieldnames or ())
            if header not in {TASK2_RUN_COLUMNS, TASK4_RUN_COLUMNS, RUN_COLUMNS}:
                raise RegistrySchemaError(
                    f"registry schema mismatch at {path}; expected a supported run schema"
                )
            raw_rows = list(reader)
    except csv.Error as error:
        raise RegistrySchemaError(f"malformed registry CSV at {path}: {error}") from error

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise RegistrySchemaError(f"registry row {index} has the wrong number of columns")
        row = {column: raw.get(column, "") for column in RUN_COLUMNS}
        run_id = row["run_id"]
        if not run_id.strip():
            raise RegistrySchemaError(f"registry row {index} has a blank run_id")
        if run_id in seen_ids:
            raise RegistrySchemaError(f"duplicate run_id values in {path}")
        seen_ids.add(run_id)
        rows.append(row)
    return rows


def _write_union_rows_with_pandas(path: Path, rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    _write_registry_csv(path, frame)


def _write_union_rows_with_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_registry_csv(path: Path, frame: pd.DataFrame) -> None:
    """Retry short Windows locks around this high-frequency shared ledger only."""
    for delay in (*_REGISTRY_WRITE_RETRY_DELAYS_SECONDS, None):
        try:
            atomic_write_csv(path, frame)
            return
        except PermissionError as error:
            transient = getattr(error, "winerror", None) in _WINDOWS_TRANSIENT_WRITE_ERRORS
            if not transient or delay is None:
                raise
            time.sleep(delay)


def new_run_id(experiment_id: str, fold: int | None, seed: int) -> str:
    """Create a readable unique run ID without encoding mutable metrics."""
    safe_experiment = re.sub(r"[^a-zA-Z0-9_-]+", "-", experiment_id).strip("-")
    fold_token = "all" if fold is None else str(fold)
    return f"{safe_experiment}-f{fold_token}-s{seed}-{uuid.uuid4().hex[:12]}"


@dataclass
class RunRecord:
    """One physical training or baseline run and all evidence needed to audit it."""

    run_id: str
    experiment_id: str
    fold: int | None
    seed: int
    config_sha256: str
    split_sha256: str
    label_map_sha256: str
    implementation_sha256: str
    task: str = "task2"
    stage: str = "experiment"
    model_family: str = ""
    benchmark_only: bool = False
    final_eligible: bool = True
    scratch: bool = True
    git_commit: str = ""
    git_dirty: bool | None = None
    transform_id: str = ""
    loss_id: str = ""
    epochs_requested: int | None = None
    epochs_completed: int | None = None
    best_epoch: int | None = None
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    peak_vram_mb: float | None = None
    parameter_count: int | None = None
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    prediction_path: str = ""
    prediction_sha256: str = ""
    history_path: str = ""
    history_sha256: str = ""
    status: str = "running"
    error_type: str = ""
    error_message: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject ambiguous identities, hashes, and lifecycle states."""
        if not self.run_id.strip() or not self.experiment_id.strip():
            raise ValueError("run_id and experiment_id must be non-empty")
        if self.fold is not None and self.fold < 0:
            raise ValueError("fold must be non-negative or None")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status: {self.status}")
        for name in (
            "config_sha256",
            "split_sha256",
            "label_map_sha256",
            "implementation_sha256",
        ):
            if not SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("checkpoint_sha256", "prediction_sha256", "history_sha256"):
            value = getattr(self, name)
            if value and not SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be empty or a lowercase SHA-256 digest")
        if self.git_commit and not GIT_PATTERN.fullmatch(self.git_commit):
            raise ValueError("git_commit must be empty or a lowercase 40-character digest")
        if self.status in TERMINAL_STATUSES and not self.finished_at_utc:
            raise ValueError("terminal runs require finished_at_utc")

    def to_row(self) -> dict[str, str]:
        """Serialize the record without locale-dependent values or implicit nulls."""
        self.validate()
        row: dict[str, str] = {}
        for name in TASK2_RUN_COLUMNS:
            value = getattr(self, name)
            if isinstance(value, bool):
                row[name] = str(value).lower()
            elif isinstance(value, dict):
                row[name] = canonical_json_bytes(value).decode("utf-8")
            elif value is None:
                row[name] = ""
            else:
                row[name] = str(value)
        return row


class RunRegistry:
    """Task 2 DataFrame view over the shared run ledger."""

    def __init__(self, path: str | Path = RUNS_CSV) -> None:
        self.path = Path(path)
        self.lock_path = _lock_path(self.path)

    def read(self) -> pd.DataFrame:
        """Return all rows as strings so identifiers and blank fields stay exact."""
        with _registry_lock(self.path, exclusive=False):
            rows = _read_union_rows(self.path)
        task_rows = [
            {column: row[column] for column in TASK2_RUN_COLUMNS}
            for row in rows
            if row["task"] != "task4"
        ]
        return pd.DataFrame(task_rows, columns=TASK2_RUN_COLUMNS)

    def append(self, record: RunRecord) -> None:
        """Append one new running row; never reuse a run ID."""
        if record.status != "running":
            raise ValueError("new registry rows must start with status='running'")
        task2_row = record.to_row()
        with _registry_lock(self.path, exclusive=True):
            rows = _read_union_rows(self.path)
            if any(row["run_id"] == record.run_id for row in rows):
                raise DuplicateRunError(f"run_id already exists: {record.run_id}")
            merged = {column: "" for column in RUN_COLUMNS}
            merged.update(task2_row)
            _write_union_rows_with_pandas(self.path, [*rows, merged])

    def finalize(self, record: RunRecord) -> None:
        """Replace a running row once, while preserving its starting identity."""
        if record.status not in TERMINAL_STATUSES:
            raise ValueError("finalized run must have a terminal status")
        with _registry_lock(self.path, exclusive=True):
            rows = _read_union_rows(self.path)
            matches = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == record.run_id and row["task"] == record.task
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
            _write_union_rows_with_pandas(self.path, rows)

    def interrupt(self, run_id: str, *, reason: str) -> None:
        """Mark one running non-Task-4 row interrupted without losing other tasks."""
        with _registry_lock(self.path, exclusive=True):
            rows = _read_union_rows(self.path)
            matches = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == run_id and row["task"] != "task4"
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
            _write_union_rows_with_pandas(self.path, rows)

    def find(self, **filters: str | int | bool | None) -> pd.DataFrame:
        """Return rows matching exact serialized field values."""
        unknown = set(filters) - set(TASK2_RUN_COLUMNS)
        if unknown:
            raise KeyError(f"unknown registry fields: {sorted(unknown)}")
        frame = self.read()
        for name, value in filters.items():
            if isinstance(value, bool):
                expected = str(value).lower()
            elif value is None:
                expected = ""
            else:
                expected = str(value)
            frame = frame.loc[frame[name] == expected]
        return frame.reset_index(drop=True)


@contextmanager
def tracked_run(registry: RunRegistry, record: RunRecord) -> Iterator[RunRecord]:
    """Persist success, failure, or interruption while re-raising run errors."""
    from fashion.train.reproducibility import capture_git_state, capture_runtime

    tracked = copy.deepcopy(record)
    tracked.started_at_utc = tracked.started_at_utc or _utc_now()
    git_state = capture_git_state()
    tracked.git_commit = tracked.git_commit or str(git_state["commit"] or "")
    if tracked.git_dirty is None:
        tracked.git_dirty = git_state["dirty"]
    tracked.runtime = tracked.runtime or capture_runtime()
    registry.append(tracked)
    started = time.perf_counter()
    try:
        yield tracked
    except KeyboardInterrupt as error:
        tracked.status = "interrupted"
        tracked.error_type = type(error).__name__
        tracked.error_message = str(error)
        raise
    except BaseException as error:
        tracked.status = "failed"
        tracked.error_type = type(error).__name__
        tracked.error_message = str(error)
        raise
    else:
        tracked.status = "completed"
    finally:
        tracked.runtime_seconds = tracked.runtime_seconds or (time.perf_counter() - started)
        tracked.finished_at_utc = _utc_now()
        registry.finalize(tracked)


class Task4RunRegistry:
    """Task 4 mapping view over the shared run ledger."""

    def __init__(self, path: Path | str, *, project_root: Path | str | None = None) -> None:
        self.path = Path(path)
        self.lock_path = _lock_path(self.path)
        if project_root is None:
            if self.path.parent.name == "results":
                project_root = self.path.parent.parent
            else:
                project_root = self.path.parent
        self.project_root = Path(project_root)

    def append(self, row: Mapping[str, Any]) -> dict[str, str]:
        """Append a new running Task 4 attempt."""
        normalized = self._normalize_row(row)
        if normalized["status"] != "running":
            raise RunRegistryError("new run status must be running")
        self._validate_row(normalized, check_artifacts=False)

        with _registry_lock(self.path, exclusive=True):
            rows = self._read_all_unlocked()
            if any(existing["run_id"] == normalized["run_id"] for existing in rows):
                raise RunRegistryError(f"duplicate run_id: {normalized['run_id']}")
            parent_run_id = normalized["parent_run_id"]
            if parent_run_id and not any(
                existing["run_id"] == parent_run_id and existing["task"] == "task4"
                for existing in rows
            ):
                raise RunRegistryError(f"parent_run_id does not exist: {parent_run_id}")
            merged = {column: "" for column in RUN_COLUMNS}
            merged.update(normalized)
            _write_union_rows_with_csv(self.path, [*rows, merged])
        return dict(normalized)

    def update(self, run_id: str, updates: Mapping[str, Any]) -> dict[str, str]:
        """Atomically replace mutable values on one running Task 4 attempt."""
        unknown = set(updates) - set(TASK4_RUN_COLUMNS)
        if unknown:
            raise RunRegistryError(f"unknown registry fields: {sorted(unknown)}")
        if not updates:
            raise RunRegistryError("updates must not be empty")

        with _registry_lock(self.path, exclusive=True):
            rows = self._read_all_unlocked()
            indexes = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == run_id and row["task"] == "task4"
            ]
            if not indexes:
                raise RunRegistryError(f"run_id does not exist: {run_id}")
            index = indexes[0]
            current = {column: rows[index][column] for column in TASK4_RUN_COLUMNS}
            if current["status"] != "running":
                raise RunRegistryError(
                    f"illegal status transition from {current['status']}: "
                    "terminal runs cannot change"
                )

            candidate_values: dict[str, Any] = dict(current)
            candidate_values.update(updates)
            candidate = self._normalize_row(candidate_values)
            for field in _TASK4_IDENTITY_FIELDS:
                if candidate[field] != current[field]:
                    raise RunRegistryError(f"immutable identity field cannot change: {field}")
            self._validate_row(candidate, check_artifacts=True)
            rows[index].update(candidate)
            _write_union_rows_with_csv(self.path, rows)
        return dict(candidate)

    def recover_failed_evidence(
        self,
        run_id: str,
        updates: Mapping[str, Any],
        *,
        expected_error_type: str,
        expected_error_message: str,
    ) -> dict[str, str]:
        """Complete a failed evidence row after exact failed-attempt validation."""
        unknown = set(updates) - set(TASK4_RUN_COLUMNS)
        if unknown:
            raise RunRegistryError(f"unknown registry fields: {sorted(unknown)}")
        if not expected_error_type.strip() or not expected_error_message.strip():
            raise RunRegistryError("evidence recovery requires exact failed-attempt identity")

        with _registry_lock(self.path, exclusive=True):
            rows = self._read_all_unlocked()
            indexes = [
                index
                for index, row in enumerate(rows)
                if row["run_id"] == run_id and row["task"] == "task4"
            ]
            if not indexes:
                raise RunRegistryError(f"run_id does not exist: {run_id}")
            index = indexes[0]
            current = {column: rows[index][column] for column in TASK4_RUN_COLUMNS}
            if current["status"] != "failed":
                raise RunRegistryError("evidence recovery requires a failed row")
            if (
                current["error_type"] != expected_error_type
                or current["error_message"] != expected_error_message
            ):
                raise RunRegistryError("failed evidence identity does not match recovery request")
            candidate_values: dict[str, Any] = dict(current)
            candidate_values.update(updates)
            candidate_values["status"] = "completed"
            candidate_values.setdefault("error_type", current["error_type"])
            candidate_values.setdefault("error_message", current["error_message"])
            candidate = self._normalize_row(candidate_values)
            for field in _TASK4_IDENTITY_FIELDS:
                if candidate[field] != current[field]:
                    raise RunRegistryError(f"immutable identity field cannot change: {field}")
            self._validate_row(candidate, check_artifacts=True)
            rows[index].update(candidate)
            _write_union_rows_with_csv(self.path, rows)
        return dict(candidate)

    def read(self) -> list[dict[str, str]]:
        """Return validated Task 4 rows in file order."""
        with _registry_lock(self.path, exclusive=False):
            rows = self._read_all_unlocked()
        return [
            {column: row[column] for column in TASK4_RUN_COLUMNS}
            for row in rows
            if row["task"] == "task4"
        ]

    def _read_all_unlocked(self) -> list[dict[str, str]]:
        try:
            rows = _read_union_rows(self.path)
            task4_rows = [
                {column: row[column] for column in TASK4_RUN_COLUMNS}
                for row in rows
                if row["task"] == "task4"
            ]
            seen_ids = {row["run_id"] for row in task4_rows}
            for row in task4_rows:
                self._validate_row(row, check_artifacts=True)
                parent = row["parent_run_id"]
                if parent and parent not in seen_ids:
                    raise RunRegistryError(
                        f"parent_run_id {parent!r} for {row['run_id']!r} does not exist"
                    )
            return rows
        except (OSError, RegistrySchemaError, RunRegistryError) as error:
            if isinstance(error, RunRegistryError) and str(error).startswith(
                "malformed registry:"
            ):
                raise
            raise RunRegistryError(f"malformed registry: {error}") from error

    @staticmethod
    def _normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
        missing = set(TASK4_RUN_COLUMNS) - set(row)
        extra = set(row) - set(TASK4_RUN_COLUMNS)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields: {sorted(missing)}")
            if extra:
                details.append(f"unknown fields: {sorted(extra)}")
            raise RunRegistryError("; ".join(details))

        normalized: dict[str, str] = {}
        for column in TASK4_RUN_COLUMNS:
            value = row[column]
            if value is None:
                normalized[column] = ""
            elif isinstance(value, bool):
                normalized[column] = str(value).lower()
            else:
                normalized[column] = str(value)
        return normalized

    def _validate_row(self, row: Mapping[str, str], *, check_artifacts: bool) -> None:
        if row["schema_version"] != TASK4_SCHEMA_VERSION:
            raise RunRegistryError(
                f"schema_version must be {TASK4_SCHEMA_VERSION}, "
                f"got {row['schema_version']!r}"
            )
        if row["task"] != "task4":
            raise RunRegistryError("task must be task4")
        for field_name in _TASK4_NONEMPTY_TEXT_FIELDS:
            if not row[field_name].strip():
                raise RunRegistryError(f"{field_name} must not be blank")
        if row["run_kind"] not in RUN_KINDS:
            raise RunRegistryError(f"run_kind has invalid value: {row['run_kind']}")
        if row["status"] not in TASK4_STATUSES:
            raise RunRegistryError(f"status has invalid value: {row['status']}")
        pretrained = _parse_task4_bool("pretrained", row["pretrained"])
        if pretrained and row["deployment_eligibility"] != "comparison_only":
            raise RunRegistryError("pretrained runs must be comparison_only")
        if row["deployment_eligibility"] not in DEPLOYMENT_ELIGIBILITIES:
            raise RunRegistryError(
                "deployment_eligibility has invalid value: "
                f"{row['deployment_eligibility']}"
            )

        started_at = _parse_task4_utc("started_at_utc", row["started_at_utc"])
        completed_at = None
        if row["completed_at_utc"]:
            completed_at = _parse_task4_utc("completed_at_utc", row["completed_at_utc"])
            if completed_at < started_at:
                raise RunRegistryError("completed_at_utc cannot be before started_at_utc")

        fold = _parse_task4_int("fold", row["fold"], minimum=0)
        if fold > 4:
            raise RunRegistryError("fold must be between 0 and 4")
        _parse_task4_int("seed", row["seed"], minimum=0)
        _parse_task4_int("embedding_dim", row["embedding_dim"], minimum=1)
        planned_epochs = _parse_task4_int(
            "planned_epochs", row["planned_epochs"], minimum=1
        )
        if row["selected_epoch"]:
            selected_epoch = _parse_task4_int(
                "selected_epoch", row["selected_epoch"], minimum=1
            )
            if selected_epoch > planned_epochs:
                raise RunRegistryError("selected_epoch cannot exceed planned_epochs")

        for field_name in ("config_hash", "split_fingerprint"):
            _validate_task4_hex_digest(field_name, row[field_name], length=64)
        _validate_task4_hex_digest("git_commit", row["git_commit"], length=40)
        _parse_task4_bool("dirty_tree", row["dirty_tree"])

        for field_name in _TASK4_OPTIONAL_NONNEGATIVE_INTS:
            if row[field_name]:
                _parse_task4_int(field_name, row[field_name], minimum=0)
        for field_name in _TASK4_OPTIONAL_FINITE_FLOATS:
            if row[field_name]:
                value = _parse_task4_float(field_name, row[field_name])
                if (
                    field_name in {"p95_end_to_end_seconds", "source_robustness_ratio"}
                    and value < 0
                ):
                    raise RunRegistryError(f"{field_name} must be non-negative")

        status = row["status"]
        if status == "running":
            if completed_at is not None:
                raise RunRegistryError(
                    "completed_at_utc must be blank while status is running"
                )
            if row["error_type"] or row["error_message"]:
                raise RunRegistryError("running rows cannot contain error fields")
        elif completed_at is None:
            raise RunRegistryError(f"completed_at_utc is required for status {status}")

        if status == "completed":
            for field_name in (
                "selected_epoch",
                "checkpoint_path",
                "checkpoint_sha256",
                "evidence_manifest_path",
            ):
                if not row[field_name]:
                    raise RunRegistryError(f"{field_name} is required for completed runs")
            _validate_task4_hex_digest(
                "checkpoint_sha256", row["checkpoint_sha256"], length=64
            )
            if check_artifacts:
                self._validate_completion_artifacts(row)
        elif row["checkpoint_sha256"]:
            _validate_task4_hex_digest(
                "checkpoint_sha256", row["checkpoint_sha256"], length=64
            )

        if status == "failed" and (not row["error_type"] or not row["error_message"]):
            raise RunRegistryError(
                "error_type and error_message are required for failed runs"
            )
        if len(row["error_message"]) > 500:
            raise RunRegistryError("error_message must be at most 500 characters")

    def _validate_completion_artifacts(self, row: Mapping[str, str]) -> None:
        checkpoint = self._artifact_path(row["checkpoint_path"])
        if not checkpoint.is_file():
            raise RunRegistryError(f"checkpoint_path does not exist: {checkpoint}")
        manifest = self._artifact_path(row["evidence_manifest_path"])
        if not manifest.is_file():
            raise RunRegistryError(f"evidence_manifest_path does not exist: {manifest}")
        actual_sha256 = _sha256_file(checkpoint)
        if actual_sha256 != row["checkpoint_sha256"]:
            raise RunRegistryError(
                "checkpoint_sha256 does not match checkpoint_path: "
                f"expected {row['checkpoint_sha256']}, got {actual_sha256}"
            )

    def _artifact_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path


def _parse_task4_utc(field_name: str, value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunRegistryError(
            f"{field_name} must be an ISO-8601 UTC timestamp"
        ) from error
    if not value.endswith("Z") or parsed.tzinfo != timezone.utc:
        raise RunRegistryError(f"{field_name} must use UTC with a Z suffix")
    return parsed


def _parse_task4_int(field_name: str, value: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RunRegistryError(f"{field_name} must be an integer") from error
    if str(parsed) != value or parsed < minimum:
        raise RunRegistryError(f"{field_name} must be an integer >= {minimum}")
    return parsed


def _parse_task4_bool(field_name: str, value: str) -> bool:
    if value not in {"true", "false"}:
        raise RunRegistryError(f"{field_name} must be true or false")
    return value == "true"


def _parse_task4_float(field_name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RunRegistryError(f"{field_name} must be numeric") from error
    if not math.isfinite(parsed):
        raise RunRegistryError(f"{field_name} must be finite")
    return parsed


def _validate_task4_hex_digest(field_name: str, value: str, *, length: int) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RunRegistryError(
            f"{field_name} must be {length} lowercase hexadecimal characters"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
