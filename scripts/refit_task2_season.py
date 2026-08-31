"""Refit the frozen Task 2 winner on valid development rows only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from fashion.task2.refit import run_or_load_development_refit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or verify the frozen I2 winner for exactly 24 epochs on valid "
            "development rows. No validation selection or holdout labels are used."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run", "load", "run_or_load"),
        default="run_or_load",
        help="Run once, verify existing artifacts, or run only when none exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the development-only refit contract."""
    arguments = _parser().parse_args(argv)
    outcome = run_or_load_development_refit(mode=arguments.mode)
    print(json.dumps(asdict(outcome), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
