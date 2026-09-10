"""Run or load the frozen matched P0S/P* benchmark pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT
from fashion.task2.pretraining import run_pretraining_matrix

DEFAULT_CONFIGS = (
    ROOT / "configs/task2/g4_p0s_resnet18_standard_scratch.json",
    ROOT / "configs/task2/g4_pstar_resnet18_standard_pretrained.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or load the frozen five-fold P0S/P* matched benchmark. Both models "
            "are benchmark-only and never final-eligible."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help=(
            "Frozen benchmark JSON. Repeat exactly twice. The defaults are "
            "g4_p0s_resnet18_standard_scratch.json and "
            "g4_pstar_resnet18_standard_pretrained.json."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("run_or_load", "run", "load"),
        default="run_or_load",
        help="Cache policy. The default verifies every cached artifact before reuse.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the pair and print fold identities and macro-F1 as JSON."""
    arguments = _parser().parse_args(argv)
    config_paths = tuple(arguments.configs or DEFAULT_CONFIGS)
    outputs = run_pretraining_matrix(config_paths, mode=arguments.mode)
    print(
        json.dumps(
            [
                {
                    "experiment_id": output.experiment_id,
                    "run_id": output.run_id,
                    "fold": output.fold,
                    "seed": output.seed,
                    "source": output.source,
                    "macro_f1": output.metrics.get("macro_f1"),
                }
                for output in outputs
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
