"""Build hash-linked Task 2 I1 evidence from five verified cached folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import (
    ROOT,
    RUNS_CSV,
    TASK2_EVIDENCE_DIR,
    TASK2_FIGURE_DIR,
)
from fashion.data.dataset import get_samples, load_splits
from fashion.task2.class_balance import run_or_load_i1_experiment
from fashion.task2.evidence import (
    build_experiment_evidence,
    build_i1_class_balance_evidence,
)

DEFAULT_CONFIG = ROOT / "configs/task2/g4_i1_effective_number_c1.json"
I1_EXPERIMENT_EVIDENCE = TASK2_EVIDENCE_DIR / "g4_i1_effective_number_c1"
I1_DECISION_EVIDENCE = TASK2_EVIDENCE_DIR / "i1_class_balance"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load five verified I1 folds, then build the pooled OOF pack and frozen "
            "class-balance decision evidence."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Frozen I1 JSON declaration (default: configs/task2/g4_i1_effective_number_c1.json)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build I1 evidence without permitting a new physical training run."""
    arguments = _parser().parse_args(argv)
    outputs = run_or_load_i1_experiment(arguments.config, mode="load")
    splits = load_splits()
    development = get_samples(splits, partition="development")
    season_development = get_samples(development, target="season").reset_index(drop=True)
    protected = splits["partition"].isin(["holdout", "quarantine"])
    protected_ids = set(splits.loc[protected, "id"].astype(int))
    experiment_manifest = build_experiment_evidence(
        outputs,
        registry_path=RUNS_CSV,
        expected_ids=season_development["id"],
        protected_ids=protected_ids,
        probability_note=(
            "Uncalibrated softmax probabilities from the best validation macro-F1 "
            "checkpoint in each fold; calibration metrics are diagnostic only."
        ),
        calibration_claim_allowed=False,
        evidence_directory=I1_EXPERIMENT_EVIDENCE,
        figure_directory=TASK2_FIGURE_DIR,
    )
    decision_manifest = build_i1_class_balance_evidence(
        i1_manifest_path=I1_EXPERIMENT_EVIDENCE / "manifest.json",
        i1_config_path=arguments.config,
        evidence_directory=I1_DECISION_EVIDENCE,
        figure_directory=TASK2_FIGURE_DIR,
    )
    print(
        json.dumps(
            {
                "run_ids": experiment_manifest["run_ids"],
                "experiment_manifest_path": experiment_manifest["manifest_path"],
                "experiment_manifest_sha256": experiment_manifest["manifest_sha256"],
                "decision_manifest_path": decision_manifest["manifest_path"],
                "decision_manifest_sha256": decision_manifest["manifest_sha256"],
                "keep_i1": decision_manifest["keep_i1"],
                "selected_experiment_id": decision_manifest["selected_experiment_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
