from pathlib import Path

import pandas as pd
import pytest

from fashion.task4.image_safety import (
    reject_protected_image_path,
    reject_protected_image_sha256,
    reject_sealed_image_rows,
)


def test_row_guard_rejects_protected_partition_and_teacher_test_path() -> None:
    protected = pd.DataFrame(
        [{"id": 1, "partition": "holdout", "path": "data/raw/teacher/train/1.jpg"}]
    )
    official = pd.DataFrame(
        [{"id": 2, "partition": "development", "path": "data/raw/teacher/test/2.jpg"}]
    )
    with pytest.raises(ValueError, match="holdout/quarantine"):
        reject_sealed_image_rows(protected)
    with pytest.raises(ValueError, match="teacher-test"):
        reject_sealed_image_rows(official)


def test_path_guard_runs_before_image_decode(tmp_path: Path) -> None:
    protected = tmp_path / "data/raw/teacher/test/query.jpg"
    with pytest.raises(ValueError, match="teacher-test"):
        reject_protected_image_path(protected)


def test_hash_guard_rejects_any_non_development_split_hash() -> None:
    splits = pd.DataFrame(
        [
            {"id": 1, "partition": "development", "sha256": "a" * 64},
            {"id": 2, "partition": "quarantine", "sha256": "b" * 64},
        ]
    )
    reject_protected_image_sha256("a" * 64, splits)
    with pytest.raises(ValueError, match="protected image hash"):
        reject_protected_image_sha256("b" * 64, splits)
