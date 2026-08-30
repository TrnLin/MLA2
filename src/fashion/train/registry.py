"""Append-once run registry with explicit lifecycle and provenance fields."""

from __future__ import annotations

import copy
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from fashion.config import RUNS_CSV
from fashion.train.artifacts import atomic_write_csv, canonical_json_bytes
from fashion.train.reproducibility import capture_git_state, capture_runtime

RUN_STATUSES = frozenset({"running", "completed", "failed", "interrupted"})
TERMINAL_STATUSES = RUN_STATUSES - {"running"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

RUN_COLUMNS = (
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


class RegistryError(RuntimeError):
    """Base error for a malformed or unsafe registry operation."""


class RegistrySchemaError(RegistryError):
    """Raised when an existing registry does not have the fixed schema."""


class DuplicateRunError(RegistryError):
    """Raised when a run ID already exists."""


class ImmutableRunError(RegistryError):
    """Raised when code attempts to rewrite a final run or its identity."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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
        for name in RUN_COLUMNS:
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
    """Read and atomically update the fixed-schema run ledger."""

    def __init__(self, path: str | Path = RUNS_CSV) -> None:
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        """Return all rows as strings so identifiers and blank fields stay exact."""
        if not self.path.exists():
            return pd.DataFrame(columns=RUN_COLUMNS)
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if tuple(frame.columns) != RUN_COLUMNS:
            raise RegistrySchemaError(
                f"registry schema mismatch at {self.path}; expected {list(RUN_COLUMNS)}"
            )
        if frame["run_id"].duplicated().any():
            raise RegistrySchemaError(f"duplicate run_id values in {self.path}")
        return frame

    def append(self, record: RunRecord) -> None:
        """Append one new running row; never reuse a run ID."""
        if record.status != "running":
            raise ValueError("new registry rows must start with status='running'")
        frame = self.read()
        if record.run_id in set(frame["run_id"]):
            raise DuplicateRunError(f"run_id already exists: {record.run_id}")
        output = pd.concat(
            [frame, pd.DataFrame([record.to_row()], columns=RUN_COLUMNS)],
            ignore_index=True,
        )
        atomic_write_csv(self.path, output)

    def finalize(self, record: RunRecord) -> None:
        """Replace a running row once, while preserving its starting identity."""
        if record.status not in TERMINAL_STATUSES:
            raise ValueError("finalized run must have a terminal status")
        frame = self.read()
        matches = frame.index[frame["run_id"] == record.run_id].tolist()
        if not matches:
            raise RegistryError(f"run_id does not exist: {record.run_id}")
        index = matches[0]
        current = frame.loc[index]
        if current["status"] != "running":
            raise ImmutableRunError(f"run is already final: {record.run_id}")
        new_row = record.to_row()
        changed = [name for name in IMMUTABLE_START_FIELDS if current[name] != new_row[name]]
        if changed:
            raise ImmutableRunError(
                f"cannot change run identity after start: {', '.join(changed)}"
            )
        frame.loc[index, list(RUN_COLUMNS)] = [new_row[name] for name in RUN_COLUMNS]
        atomic_write_csv(self.path, frame)

    def find(self, **filters: str | int | bool | None) -> pd.DataFrame:
        """Return rows matching exact serialized field values."""
        unknown = set(filters) - set(RUN_COLUMNS)
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
