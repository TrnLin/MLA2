"""Run once or verify the fixed-budget Task 1 development refit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from fashion.task1.refit import run_task1_refit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected scratch Article Type CNN for exactly 20 epochs on "
            "labelled development rows, without validation or early stopping."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run", "load", "run_or_load"),
        default="run_or_load",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    outcome = run_task1_refit(mode=arguments.mode)
    print(json.dumps(asdict(outcome), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
