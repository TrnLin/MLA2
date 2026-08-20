from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from data_helpers import write_id_csv
from fashion.data.quarantine import (
    QuarantinePaths,
    establish_quarantine,
    read_id_column,
)


@dataclass(frozen=True)
class CsvCall:
    path: Path
    usecols: list[str] | None


def make_paths(
    tmp_path: Path,
    *,
    test_ids: list[str | int] = [90, 91],
    train_ids: list[str | int] = [1, 2, 3],
    original_ids: list[str | int] = [1, 2, 3, 90, 91],
) -> QuarantinePaths:
    """Build hand-checked miniature populations for quarantine behavior."""
    teacher_test_csv = write_id_csv(tmp_path / "teacher-test.csv", test_ids)
    teacher_train_csv = write_id_csv(tmp_path / "teacher-train.csv", train_ids)
    original_csv = write_id_csv(tmp_path / "original.csv", original_ids)
    return QuarantinePaths(
        teacher_test_csv=teacher_test_csv,
        teacher_train_csv=teacher_train_csv,
        original_csv=original_csv,
        expected_test_count=len(test_ids),
        expected_train_count=len(train_ids),
        expected_original_count=len(original_ids),
    )


def test_establish_quarantine_reads_official_test_ids_first_and_only_reads_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading a target column or training metadata first breaks the quarantine boundary."""
    paths = make_paths(tmp_path)
    recorded_calls: list[CsvCall] = []
    original_read_csv = pd.read_csv

    def record_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        recorded_calls.append(CsvCall(Path(path), kwargs.get("usecols")))  # type: ignore[arg-type]
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr("fashion.data.quarantine.pd.read_csv", record_read_csv)

    audit = establish_quarantine(paths)

    assert audit.test_ids == (90, 91)
    assert audit.train_ids == (1, 2, 3)
    assert audit.original_ids == (1, 2, 3, 90, 91)
    assert recorded_calls[0].path == paths.teacher_test_csv
    assert recorded_calls[0].usecols == ["id"]
    assert all(call.usecols == ["id"] for call in recorded_calls)


def test_read_id_column_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate IDs make a claimed population count untrustworthy."""
    path = write_id_csv(tmp_path / "duplicates.csv", [1, 1])

    with pytest.raises(ValueError, match="Duplicate IDs in Duplicate metadata"):
        read_id_column(path, "Duplicate metadata")


def test_read_id_column_rejects_non_integer_ids(tmp_path: Path) -> None:
    """A non-integer ID cannot be safely reconciled across metadata files."""
    path = write_id_csv(tmp_path / "non-integers.csv", [1, "1.5"])

    with pytest.raises(ValueError, match="Non-integer ID in Invalid metadata: '1.5'"):
        read_id_column(path, "Invalid metadata")


def test_establish_quarantine_rejects_official_train_test_overlap(tmp_path: Path) -> None:
    """An ID in both populations would leak official evaluation data into training."""
    paths = make_paths(tmp_path, train_ids=[1, 90], original_ids=[1, 90, 91])

    with pytest.raises(ValueError, match=r"Official train/test ID overlap: \[90\]"):
        establish_quarantine(paths)


def test_establish_quarantine_rejects_expected_count_mismatch(tmp_path: Path) -> None:
    """A partial official test file must not become the approved boundary."""
    paths = make_paths(tmp_path)
    paths = QuarantinePaths(
        teacher_test_csv=paths.teacher_test_csv,
        teacher_train_csv=paths.teacher_train_csv,
        original_csv=paths.original_csv,
        expected_test_count=3,
        expected_train_count=3,
        expected_original_count=5,
    )

    with pytest.raises(ValueError, match="Expected 3 official test IDs, found 2"):
        establish_quarantine(paths)


def test_establish_quarantine_rejects_official_ids_missing_from_original_metadata(
    tmp_path: Path,
) -> None:
    """Missing original IDs would invalidate the population reconciliation proof."""
    paths = make_paths(tmp_path, original_ids=[1, 2, 3, 90])

    with pytest.raises(ValueError, match="Official IDs are missing from original metadata"):
        establish_quarantine(paths)
