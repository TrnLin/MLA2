"""Run or load the frozen Task 2 I1 effective-number experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT
from fashion.task2.class_balance import run_or_load_i1_experiment

DEFAULT_CONFIG = ROOT / "configs/task2/g4_i1_effective_number_c1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or load the frozen five-fold I1 effective-number class-balance "
            "experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Frozen I1 JSON declaration "
            "(default: configs/task2/g4_i1_effective_number_c1.json)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("run_or_load", "run", "load"),
        default="run_or_load",
        help="Cache policy. The default verifies all cached artifacts before reuse.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute I1 and print fold IDs, sources, and comparison metrics as JSON."""
    arguments = _parser().parse_args(argv)
    outputs = run_or_load_i1_experiment(arguments.config, mode=arguments.mode)
    print(
        json.dumps(
            [
                {
                    "run_id": output.run_id,
                    "fold": output.fold,
                    "seed": output.seed,
                    "source": output.source,
                    "macro_f1": output.metrics.get("macro_f1"),
                    "spring_f1": output.metrics.get("per_class", {})
                    .get("Spring", {})
                    .get("f1"),
                }
                for output in outputs
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
