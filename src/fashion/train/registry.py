"""Locked, append-first registry for training attempts."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

RUN_COLUMNS = (
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

RUN_KINDS = frozenset({"smoke", "candidate", "benchmark", "stability", "final_refit"})
STATUSES = frozenset({"running", "completed", "failed", "cancelled"})
DEPLOYMENT_ELIGIBILITIES = frozenset({"eligible", "comparison_only"})
SCHEMA_VERSION = "1"

_IDENTITY_FIELDS = frozenset(
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
_NONEMPTY_TEXT_FIELDS = (
    "run_id",
    "started_at_utc",
    "task",
    "method",
    "architecture",
    "objective",
    "source_policy",
    "weight_origin",
)
_OPTIONAL_NONNEGATIVE_INTS = ("selected_epoch", "parameter_count", "index_bytes")
_OPTIONAL_FINITE_FLOATS = (
    "development_winner_score",
    "cross_source_score",
    "source_robustness_ratio",
    "protocol_b_recall_at_10",
    "p95_end_to_end_seconds",
)


class RunRegistryError(ValueError):
    """Raised when a registry operation would break the run record contract."""


class RunRegistry:
    """Read and mutate one run CSV through a process-safe lock."""

    def __init__(self, path: Path | str, *, project_root: Path | str | None = None) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        if project_root is None:
            if self.path.parent.name == "results":
                project_root = self.path.parent.parent
            else:
                project_root = self.path.parent
        self.project_root = Path(project_root)

    def append(self, row: Mapping[str, Any]) -> dict[str, str]:
        """Append a new running attempt and return its serialized row."""

        normalized = self._normalize_row(row)
        if normalized["status"] != "running":
            raise RunRegistryError("new run status must be running")
        self._validate_row(normalized, check_artifacts=False)

        with self._lock(exclusive=True):
            rows = self._read_unlocked()
            if any(existing["run_id"] == normalized["run_id"] for existing in rows):
                raise RunRegistryError(f"duplicate run_id: {normalized['run_id']}")
            parent_run_id = normalized["parent_run_id"]
            if parent_run_id and not any(row["run_id"] == parent_run_id for row in rows):
                raise RunRegistryError(f"parent_run_id does not exist: {parent_run_id}")
            self._write_unlocked([*rows, normalized])
        return dict(normalized)

    def update(self, run_id: str, updates: Mapping[str, Any]) -> dict[str, str]:
        """Atomically replace mutable values on one running attempt."""

        unknown = set(updates) - set(RUN_COLUMNS)
        if unknown:
            raise RunRegistryError(f"unknown registry fields: {sorted(unknown)}")
        if not updates:
            raise RunRegistryError("updates must not be empty")

        with self._lock(exclusive=True):
            rows = self._read_unlocked()
            indexes = [index for index, row in enumerate(rows) if row["run_id"] == run_id]
            if not indexes:
                raise RunRegistryError(f"run_id does not exist: {run_id}")
            current = rows[indexes[0]]
            if current["status"] != "running":
                raise RunRegistryError(
                    f"illegal status transition from {current['status']}: "
                    "terminal runs cannot change"
                )

            candidate_values: dict[str, Any] = dict(current)
            candidate_values.update(updates)
            candidate = self._normalize_row(candidate_values)
            for field in _IDENTITY_FIELDS:
                if candidate[field] != current[field]:
                    raise RunRegistryError(f"immutable identity field cannot change: {field}")

            target_status = candidate["status"]
            if target_status not in STATUSES:
                raise RunRegistryError(f"status has invalid value: {target_status}")
            self._validate_row(candidate, check_artifacts=True)
            rows[indexes[0]] = candidate
            self._write_unlocked(rows)
        return dict(candidate)

    def recover_failed_evidence(
        self,
        run_id: str,
        updates: Mapping[str, Any],
        *,
        expected_error_type: str,
        expected_error_message: str,
    ) -> dict[str, str]:
        """Complete a failed evidence row only after external attempt audit validation."""

        unknown = set(updates) - set(RUN_COLUMNS)
        if unknown:
            raise RunRegistryError(f"unknown registry fields: {sorted(unknown)}")
        if not expected_error_type.strip() or not expected_error_message.strip():
            raise RunRegistryError("evidence recovery requires exact failed-attempt identity")
        with self._lock(exclusive=True):
            rows = self._read_unlocked()
            indexes = [index for index, row in enumerate(rows) if row["run_id"] == run_id]
            if not indexes:
                raise RunRegistryError(f"run_id does not exist: {run_id}")
            current = rows[indexes[0]]
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
            for field in _IDENTITY_FIELDS:
                if candidate[field] != current[field]:
                    raise RunRegistryError(f"immutable identity field cannot change: {field}")
            self._validate_row(candidate, check_artifacts=True)
            rows[indexes[0]] = candidate
            self._write_unlocked(rows)
        return dict(candidate)

    def read(self) -> list[dict[str, str]]:
        """Return validated rows in file order."""

        with self._lock(exclusive=False):
            return [dict(row) for row in self._read_unlocked()]

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_handle:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, strict=True)
                if tuple(reader.fieldnames or ()) != RUN_COLUMNS:
                    raise RunRegistryError("header does not match registry schema")
                raw_rows = list(reader)

            rows: list[dict[str, str]] = []
            seen_ids: set[str] = set()
            for index, raw_row in enumerate(raw_rows, start=2):
                if None in raw_row or any(value is None for value in raw_row.values()):
                    raise RunRegistryError(f"row {index} has the wrong number of columns")
                row = self._normalize_row(raw_row)
                self._validate_row(row, check_artifacts=True)
                if row["run_id"] in seen_ids:
                    raise RunRegistryError(f"duplicate run_id: {row['run_id']}")
                seen_ids.add(row["run_id"])
                rows.append(row)
            for row in rows:
                parent = row["parent_run_id"]
                if parent and parent not in seen_ids:
                    raise RunRegistryError(
                        f"parent_run_id {parent!r} for {row['run_id']!r} does not exist"
                    )
            return rows
        except (csv.Error, OSError, RunRegistryError) as error:
            if isinstance(error, RunRegistryError) and str(error).startswith(
                "malformed registry:"
            ):
                raise
            raise RunRegistryError(f"malformed registry: {error}") from error

    def _write_unlocked(self, rows: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
        missing = set(RUN_COLUMNS) - set(row)
        extra = set(row) - set(RUN_COLUMNS)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields: {sorted(missing)}")
            if extra:
                details.append(f"unknown fields: {sorted(extra)}")
            raise RunRegistryError("; ".join(details))

        normalized: dict[str, str] = {}
        for field in RUN_COLUMNS:
            value = row[field]
            if value is None:
                normalized[field] = ""
            elif isinstance(value, bool):
                normalized[field] = str(value).lower()
            else:
                normalized[field] = str(value)
        return normalized

    def _validate_row(self, row: Mapping[str, str], *, check_artifacts: bool) -> None:
        if row["schema_version"] != SCHEMA_VERSION:
            raise RunRegistryError(
                f"schema_version must be {SCHEMA_VERSION}, got {row['schema_version']!r}"
            )
        for field in _NONEMPTY_TEXT_FIELDS:
            if not row[field].strip():
                raise RunRegistryError(f"{field} must not be blank")
        if row["run_kind"] not in RUN_KINDS:
            raise RunRegistryError(f"run_kind has invalid value: {row['run_kind']}")
        if row["status"] not in STATUSES:
            raise RunRegistryError(f"status has invalid value: {row['status']}")
        pretrained = _parse_bool("pretrained", row["pretrained"])
        if pretrained and row["deployment_eligibility"] != "comparison_only":
            raise RunRegistryError("pretrained runs must be comparison_only")
        if row["deployment_eligibility"] not in DEPLOYMENT_ELIGIBILITIES:
            raise RunRegistryError(
                "deployment_eligibility has invalid value: "
                f"{row['deployment_eligibility']}"
            )

        started_at = _parse_utc("started_at_utc", row["started_at_utc"])
        completed_at = None
        if row["completed_at_utc"]:
            completed_at = _parse_utc("completed_at_utc", row["completed_at_utc"])
            if completed_at < started_at:
                raise RunRegistryError("completed_at_utc cannot be before started_at_utc")

        fold = _parse_int("fold", row["fold"], minimum=0)
        if fold > 4:
            raise RunRegistryError("fold must be between 0 and 4")
        _parse_int("seed", row["seed"], minimum=0)
        _parse_int("embedding_dim", row["embedding_dim"], minimum=1)
        planned_epochs = _parse_int("planned_epochs", row["planned_epochs"], minimum=1)
        selected_epoch = None
        if row["selected_epoch"]:
            selected_epoch = _parse_int("selected_epoch", row["selected_epoch"], minimum=1)
            if selected_epoch > planned_epochs:
                raise RunRegistryError("selected_epoch cannot exceed planned_epochs")

        for field in ("config_hash", "split_fingerprint"):
            _validate_hex_digest(field, row[field], length=64)
        _validate_hex_digest("git_commit", row["git_commit"], length=40)
        _parse_bool("dirty_tree", row["dirty_tree"])

        for field in _OPTIONAL_NONNEGATIVE_INTS:
            if row[field]:
                _parse_int(field, row[field], minimum=0)
        for field in _OPTIONAL_FINITE_FLOATS:
            if row[field]:
                value = _parse_float(field, row[field])
                if field in {"p95_end_to_end_seconds", "source_robustness_ratio"} and value < 0:
                    raise RunRegistryError(f"{field} must be non-negative")

        status = row["status"]
        if status == "running":
            if completed_at is not None:
                raise RunRegistryError("completed_at_utc must be blank while status is running")
            if row["error_type"] or row["error_message"]:
                raise RunRegistryError("running rows cannot contain error fields")
        else:
            if completed_at is None:
                raise RunRegistryError(f"completed_at_utc is required for status {status}")

        if status == "completed":
            for field in (
                "selected_epoch",
                "checkpoint_path",
                "checkpoint_sha256",
                "evidence_manifest_path",
            ):
                if not row[field]:
                    raise RunRegistryError(f"{field} is required for completed runs")
            _validate_hex_digest("checkpoint_sha256", row["checkpoint_sha256"], length=64)
            if check_artifacts:
                self._validate_completion_artifacts(row)
        elif row["checkpoint_sha256"]:
            _validate_hex_digest("checkpoint_sha256", row["checkpoint_sha256"], length=64)

        if status == "failed" and (not row["error_type"] or not row["error_message"]):
            raise RunRegistryError("error_type and error_message are required for failed runs")
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


def _parse_utc(field: str, value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunRegistryError(f"{field} must be an ISO-8601 UTC timestamp") from error
    if not value.endswith("Z") or parsed.tzinfo != timezone.utc:
        raise RunRegistryError(f"{field} must use UTC with a Z suffix")
    return parsed


def _parse_int(field: str, value: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RunRegistryError(f"{field} must be an integer") from error
    if str(parsed) != value or parsed < minimum:
        raise RunRegistryError(f"{field} must be an integer >= {minimum}")
    return parsed


def _parse_bool(field: str, value: str) -> bool:
    if value not in {"true", "false"}:
        raise RunRegistryError(f"{field} must be true or false")
    return value == "true"


def _parse_float(field: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RunRegistryError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise RunRegistryError(f"{field} must be finite")
    return parsed


def _validate_hex_digest(field: str, value: str, *, length: int) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RunRegistryError(f"{field} must be {length} lowercase hexadecimal characters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
