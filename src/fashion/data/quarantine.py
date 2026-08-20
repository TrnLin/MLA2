"""Establish the ID-only boundary around official test metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fashion.config import ORIGINAL_CSV, TEACHER_TRAIN_CSV, TEST_CSV


@dataclass(frozen=True)
class QuarantinePaths:
    """Locations and expected population sizes for ID reconciliation."""

    teacher_test_csv: Path = TEST_CSV
    teacher_train_csv: Path = TEACHER_TRAIN_CSV
    original_csv: Path = ORIGINAL_CSV
    expected_test_count: int = 5_829
    expected_train_count: int = 38_617
    expected_original_count: int = 44_446


@dataclass(frozen=True)
class QuarantineAudit:
    """Immutable, reconciled source population IDs."""

    test_ids: tuple[int, ...]
    train_ids: tuple[int, ...]
    original_ids: tuple[int, ...]


def read_id_column(path: Path, source: str) -> pd.Index:
    """Read, validate, and sort a metadata file's ID column without labels."""
    frame = pd.read_csv(
        path,
        usecols=["id"],
        dtype={"id": "string"},
        keep_default_na=False,
    )
    values = frame["id"]
    valid_ids = values.str.fullmatch(r"[+-]?\d+")
    if not valid_ids.all():
        invalid = values.loc[~valid_ids].iloc[0]
        raise ValueError(f"Non-integer ID in {source}: {invalid!r}")

    ids = values.astype(int)
    duplicates = ids.loc[ids.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate IDs in {source}: {duplicates[:10]}")

    return pd.Index(ids, dtype="int64").sort_values()


def establish_quarantine(
    paths: QuarantinePaths = QuarantinePaths(),
) -> QuarantineAudit:
    """Prove the official test IDs are disjoint before target labels may load."""
    test_ids = read_id_column(paths.teacher_test_csv, "Official test metadata")
    train_ids = read_id_column(paths.teacher_train_csv, "Teacher training metadata")
    original_ids = read_id_column(paths.original_csv, "Original metadata")

    overlap = test_ids.intersection(train_ids)
    if not overlap.empty:
        raise ValueError(f"Official train/test ID overlap: {overlap.tolist()[:10]}")
    if len(test_ids) != paths.expected_test_count:
        raise ValueError(
            f"Expected {paths.expected_test_count} official test IDs, found {len(test_ids)}"
        )
    if len(train_ids) != paths.expected_train_count:
        raise ValueError(
            "Expected "
            f"{paths.expected_train_count} official training IDs, found {len(train_ids)}"
        )
    if len(original_ids) != paths.expected_original_count:
        raise ValueError(
            f"Expected {paths.expected_original_count} original IDs, found {len(original_ids)}"
        )
    if train_ids.union(test_ids).difference(original_ids).size:
        raise ValueError("Official IDs are missing from original metadata")

    return QuarantineAudit(
        test_ids=tuple(test_ids.tolist()),
        train_ids=tuple(train_ids.tolist()),
        original_ids=tuple(original_ids.tolist()),
    )
