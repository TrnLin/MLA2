"""Canonical guards that keep protected images closed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion.data.splits import PROTECTED_PARTITIONS

TEACHER_TEST_MARKERS = (
    "data/raw/teacher/test",
    "teacher/test",
    "images_test",
    "styles_prediction.csv",
)


def reject_sealed_image_rows(
    frame: pd.DataFrame,
    *,
    require_development: bool = False,
) -> None:
    """Reject rows that could expose sealed or official teacher-test images."""

    if "partition" not in frame:
        raise ValueError("image rows must carry canonical partition values")
    protected = frame["partition"].isin(PROTECTED_PARTITIONS)
    if protected.any():
        ids = frame.loc[protected, "id"].astype(str).head(5).tolist()
        raise ValueError(f"sealed holdout/quarantine rows reached image access: {ids}")
    if require_development and not frame["partition"].eq("development").all():
        raise ValueError("sealed or unknown partition rows reached image access")

    path_columns = [
        column for column in frame.columns if column == "path" or str(column).endswith("_path")
    ]
    for column in path_columns:
        lowered = frame[column].astype(str).str.replace("\\", "/", regex=False).str.lower()
        if lowered.map(
            lambda value: any(marker in value for marker in TEACHER_TEST_MARKERS)
        ).any():
            raise ValueError("official teacher-test path reached image access")


def reject_protected_image_path(path: str | Path) -> Path:
    """Resolve an image path only after refusing official teacher-test markers."""

    resolved = Path(path).expanduser().resolve()
    lowered = resolved.as_posix().lower()
    if any(marker in lowered for marker in TEACHER_TEST_MARKERS):
        raise ValueError("official teacher-test path reached image access")
    return resolved


def reject_protected_image_sha256(digest: str, splits: pd.DataFrame) -> None:
    """Reject a digest belonging to any non-development split row."""

    value = str(digest).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("image SHA-256 must contain 64 hexadecimal characters")
    protected = splits.loc[
        ~splits["partition"].eq("development"), "sha256"
    ].astype(str).str.lower()
    if protected.eq(value).any():
        raise ValueError("protected image hash reached image access")
