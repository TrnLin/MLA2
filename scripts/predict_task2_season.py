"""Predict Season probabilities for explicit image paths with the frozen Task 2 bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fashion.task2.inference import (
    InvalidSeasonImageError,
    load_season_bundle,
    predict_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run verified image-only Season inference. This command accepts only the "
            "image paths you provide; it never opens holdout labels or writes the official CSV."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="One or more image files, returned in the same order.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Auto uses CUDA when available, otherwise CPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the verified bundle once and emit one JSON prediction array."""
    arguments = _parser().parse_args(argv)
    try:
        bundle = load_season_bundle(device=arguments.device)
        predictions = predict_manifest(bundle, arguments.images)
    except (
        FileNotFoundError,
        InvalidSeasonImageError,
        FloatingPointError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            [prediction.to_dict() for prediction in predictions],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
