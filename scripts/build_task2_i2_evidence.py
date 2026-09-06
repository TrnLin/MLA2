"""Build hash-linked Task 2 I2 evidence from ten verified cached folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT, RUNS_CSV, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.task2.evidence import build_experiment_evidence
from fashion.task2.multitask import load_i2_config, run_or_load_i2_experiment
from fashion.task2.multitask_evidence import build_i2_transfer_evidence

DEFAULT_CONFIGS = (
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_1_c1.json",
    ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load ten verified I2 folds, build both pooled OOF packs, then close the "
            "ArticleType transfer and aligned/conflict gate."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help=(
            "Frozen I2 JSON declaration. Repeat twice. The default uses the declared "
            "lambda 0.1 and lambda 0.3 configs."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build I2 evidence without permitting any new physical training run."""
    arguments = _parser().parse_args(argv)
    config_paths = tuple(arguments.configs or DEFAULT_CONFIGS)
    if len(config_paths) != 2:
        raise ValueError("I2 evidence requires exactly two config paths")
    configs = [load_i2_config(path) for path in config_paths]
    splits = load_splits()
    development = get_samples(splits, partition="development")
    season_development = get_samples(development, target="season").reset_index(drop=True)
    protected_ids = set(
        splits.loc[
            splits["partition"].isin(["holdout", "quarantine"]), "id"
        ].astype(int)
    )

    experiment_manifests = []
    for config in configs:
        outputs = run_or_load_i2_experiment(config, mode="load")
        slug = config.experiment_id.replace("-", "_")
        experiment_manifests.append(
            build_experiment_evidence(
                outputs,
                registry_path=RUNS_CSV,
                expected_ids=season_development["id"],
                protected_ids=protected_ids,
                probability_note=(
                    "Uncalibrated softmax probabilities from the best validation Season "
                    "macro-F1 checkpoint; ArticleType is training-only."
                ),
                calibration_claim_allowed=False,
                evidence_directory=TASK2_EVIDENCE_DIR / slug,
                figure_directory=TASK2_FIGURE_DIR,
            )
        )
    decision_manifest = build_i2_transfer_evidence(
        i2_manifest_paths=tuple(
            Path(manifest["manifest_path"]) for manifest in experiment_manifests
        ),
        i2_config_paths=config_paths,
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
                "keep_i2": decision_manifest["keep_i2"],
                "selected_experiment_id": decision_manifest["selected_experiment_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
