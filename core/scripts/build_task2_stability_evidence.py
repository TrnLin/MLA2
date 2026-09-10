"""Build Task 2 seed-stability evidence from verified cached folds only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT, RUNS_CSV, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.task2.evidence import build_experiment_evidence
from fashion.task2.experiments import (
    load_experiment_config,
    run_or_load_experiment,
)
from fashion.task2.multitask import (
    load_i2_config,
    run_or_load_i2_experiment,
)
from fashion.task2.stability import (
    C2_PRIMARY_CONFIG_PATH,
    C2_STABILITY_CONFIG_PATH,
    I2_PRIMARY_CONFIG_PATH,
    I2_STABILITY_CONFIG_PATH,
    load_stability_pair,
    run_stability_matrix,
)
from fashion.task2.stability_evidence import build_seed_stability_evidence

CONFIG_PATHS = (
    C2_PRIMARY_CONFIG_PATH,
    C2_STABILITY_CONFIG_PATH,
    I2_PRIMARY_CONFIG_PATH,
    I2_STABILITY_CONFIG_PATH,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load four complete OOF packs and build the closed G5 decision."""
    if argv:
        raise ValueError("stability evidence uses only the four frozen configs")
    pair = load_stability_pair()
    primary_c2 = load_experiment_config(C2_PRIMARY_CONFIG_PATH)
    primary_i2 = load_i2_config(I2_PRIMARY_CONFIG_PATH)
    outputs = {
        primary_c2.experiment_id: run_or_load_experiment(
            primary_c2,
            mode="load",
        ),
        primary_i2.experiment_id: run_or_load_i2_experiment(
            primary_i2,
            mode="load",
        ),
    }
    stability_outputs = run_stability_matrix(pair, mode="load")
    for config in (pair.c2, pair.i2):
        outputs[config.experiment_id] = [
            output for output in stability_outputs if output.experiment_id == config.experiment_id
        ]

    splits = load_splits()
    development = get_samples(splits, partition="development")
    season_development = get_samples(
        development,
        target="season",
    ).reset_index(drop=True)
    protected_ids = set(
        splits.loc[
            splits["partition"].isin(["holdout", "quarantine"]),
            "id",
        ].astype(int)
    )
    probability_notes = {
        "C2": (
            "Uncalibrated softmax probabilities from the best validation macro-F1 "
            "checkpoint in each fold; calibration metrics are diagnostic only."
        ),
        "I2": (
            "Uncalibrated softmax probabilities from the best validation Season "
            "macro-F1 checkpoint; ArticleType is training-only."
        ),
    }
    configs = (primary_c2, pair.c2, primary_i2, pair.i2)
    manifest_paths: list[Path] = []
    manifests = {}
    for config in configs:
        candidate = "I2" if hasattr(config, "auxiliary") else "C2"
        slug = config.experiment_id.replace("-", "_")
        manifest = build_experiment_evidence(
            outputs[config.experiment_id],
            registry_path=RUNS_CSV,
            expected_ids=season_development["id"],
            protected_ids=protected_ids,
            probability_note=probability_notes[candidate],
            calibration_claim_allowed=False,
            evidence_directory=TASK2_EVIDENCE_DIR / slug,
            figure_directory=TASK2_FIGURE_DIR,
        )
        manifests[config.experiment_id] = manifest
        manifest_paths.append(Path(manifest["manifest_path"]))

    decision_manifest = build_seed_stability_evidence(
        experiment_manifest_paths=manifest_paths,
        experiment_config_paths=CONFIG_PATHS,
        project_root=ROOT,
        expected_row_count=len(season_development),
        evidence_directory=TASK2_EVIDENCE_DIR / "seed_stability",
        figure_directory=TASK2_FIGURE_DIR,
    )
    print(
        json.dumps(
            {
                "experiment_manifests": {
                    experiment_id: {
                        "run_ids": manifest["run_ids"],
                        "path": manifest["manifest_path"],
                        "sha256": manifest["manifest_sha256"],
                    }
                    for experiment_id, manifest in manifests.items()
                },
                "decision_manifest_path": decision_manifest["manifest_path"],
                "decision_manifest_sha256": decision_manifest["manifest_sha256"],
                "ordering_stable": decision_manifest["ordering_stable"],
                "ultimate_winner_frozen": decision_manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
