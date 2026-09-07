"""Export the pinned fold-1 teacher gallery for Task 4 search."""

from __future__ import annotations

import argparse
from pathlib import Path

from fashion.config import ROOT
from fashion.task4 import export_teacher_gallery_artifact

SELECTED_RUN_ID = "task4-candidate-r5-task9-preexec"
SELECTED_CHECKPOINT_SHA256 = (
    "521e96f3df9e28853309bd607030523f59ef8425326da56dfcf2b754e97631b3"
)
EXPECTED_SOURCE_IDENTITY_SHA256 = (
    "00ea23461b4d5ef35a59d2078be8e7f6e9a24f5089e42a4057b1bfdd057747bc"
)
DEFAULT_SOURCE_CACHE = ROOT / (
    "results/cache/task4/features/candidate/"
    "task4-candidate-r5-task9-preexec/521e96f3df9e2885/"
    "1def6dda1e408413/fold-1/teacher/a1be928e99d60f220b9f"
)
DEFAULT_DESTINATION = ROOT / "models/task4_teacher_gallery"
DEFAULT_SPLITS = ROOT / "data/processed/splits.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the pinned development-only fold-1 teacher gallery."
    )
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_teacher_gallery_artifact(
        args.source_cache,
        args.destination,
        splits_path=args.splits,
        expected_source_identity_sha256=EXPECTED_SOURCE_IDENTITY_SHA256,
        expected_checkpoint_sha256=SELECTED_CHECKPOINT_SHA256,
        expected_run_id=SELECTED_RUN_ID,
    )
    print(f"Teacher gallery: {exported}")


if __name__ == "__main__":
    main()
