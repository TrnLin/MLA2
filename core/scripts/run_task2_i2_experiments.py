"""Run or load the two frozen Task 2 I2 auxiliary-supervision experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT
from fashion.task2.multitask import load_i2_config, run_i2_matrix

DEFAULT_CONFIGS = (
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_1_c1.json",
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or load the two frozen five-fold I2 ArticleType auxiliary-loss "
            "experiments."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help=(
            "Frozen I2 JSON declaration. Repeat to run multiple configs. The default "
            "runs g4_i2_article_type_lambda_0_1_c1.json and "
            "g4_i2_article_type_lambda_0_3_c1.json."
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
    """Execute I2 and print fold identities, sources, and Season metrics as JSON."""
    arguments = _parser().parse_args(argv)
    config_paths = tuple(arguments.configs or DEFAULT_CONFIGS)
    configs = [load_i2_config(path) for path in config_paths]
    auxiliary_weights = {
        config.experiment_id: config.auxiliary.loss_weight for config in configs
    }
    outputs = run_i2_matrix(configs, mode=arguments.mode)
    print(
        json.dumps(
            [
                {
                    "experiment_id": output.experiment_id,
                    "auxiliary_weight": auxiliary_weights[output.experiment_id],
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
