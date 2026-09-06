"""Build hash-linked evidence from ten verified cached P0S/P* folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT, RUNS_CSV, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.task2.evidence import build_experiment_evidence
from fashion.task2.experiments import load_experiment_config
from fashion.task2.pretraining import run_pretraining_matrix
from fashion.task2.pretraining_evidence import build_pretraining_benchmark_evidence

DEFAULT_CONFIGS = (
    ROOT / "configs/task2/g4_p0s_resnet18_standard_scratch.json",
    ROOT / "configs/task2/g4_pstar_resnet18_standard_pretrained.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load ten verified P0S/P* folds and build the matched initialisation "
            "comparison, teacher-style learning curves, and non-selection boundary."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help="Frozen benchmark JSON. Repeat twice; defaults use P0S then P*.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build benchmark evidence without permitting a new physical training run."""
    arguments = _parser().parse_args(argv)
    config_paths = tuple(arguments.configs or DEFAULT_CONFIGS)
    configs = [load_experiment_config(path) for path in config_paths]
    outputs = run_pretraining_matrix(config_paths, mode="load")
    splits = load_splits()
    development = get_samples(splits, partition="development")
    season_development = get_samples(development, target="season").reset_index(drop=True)
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"].astype(int)
    )

    experiment_manifests = []
    for config in configs:
        experiment_outputs = [
            output for output in outputs if output.experiment_id == config.experiment_id
        ]
        slug = config.experiment_id.replace("-", "_")
        experiment_manifests.append(
            build_experiment_evidence(
                experiment_outputs,
                registry_path=RUNS_CSV,
                expected_ids=season_development["id"],
                protected_ids=protected_ids,
                probability_note=(
                    "Uncalibrated softmax probabilities from the checkpoint selected by "
                    "validation Season macro-F1."
                ),
                calibration_claim_allowed=False,
                evidence_directory=TASK2_EVIDENCE_DIR / slug,
                figure_directory=TASK2_FIGURE_DIR,
            )
        )
    decision_manifest = build_pretraining_benchmark_evidence(
        experiment_manifest_paths=tuple(
            Path(manifest["manifest_path"]) for manifest in experiment_manifests
        ),
        experiment_config_paths=config_paths,
        expected_row_count=len(season_development),
    )
    print(
        json.dumps(
            {
                "experiment_manifests": {
                    config.experiment_id: {
                        "run_ids": manifest["run_ids"],
                        "path": manifest["manifest_path"],
                        "sha256": manifest["manifest_sha256"],
                    }
                    for config, manifest in zip(configs, experiment_manifests, strict=True)
                },
                "decision_manifest_path": decision_manifest["manifest_path"],
                "decision_manifest_sha256": decision_manifest["manifest_sha256"],
                "pstar_minus_p0s_macro_f1": decision_manifest["observed_pstar_minus_p0s_macro_f1"],
                "candidate_selection_affected": decision_manifest["candidate_selection_affected"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
