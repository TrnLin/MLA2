from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.data.dataset import get_samples, load_splits
from fashion.task2.evidence import build_experiment_evidence
from fashion.task2.experiments import ExperimentConfig, run_or_load_experiment
from fashion.train.registry import RunRegistry


def _project_paths(prepared_project) -> dict[str, Path]:
    root = prepared_project.root
    mappings = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    labels = ["Fall", "Spring", "Summer", "Winter"]
    mappings["season"].update(
        {
            "num_classes": len(labels),
            "classes": labels,
            "label_to_index": {label: index for index, label in enumerate(labels)},
            "index_to_label": {str(index): label for index, label in enumerate(labels)},
        }
    )
    prepared_project.label_maps.write_text(
        json.dumps(mappings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "data_root": root,
        "splits_path": prepared_project.splits,
        "label_map_path": prepared_project.label_maps,
        "registry_path": root / "results/runs.csv",
        "checkpoint_directory": root / "tmp/task2/checkpoints",
        "run_directory": root / "tmp/task2/runs",
    }


def test_five_fold_evidence_is_complete_portable_and_cache_stable(
    prepared_project,
) -> None:
    paths = _project_paths(prepared_project)
    config = ExperimentConfig(
        experiment_id="b0-evidence-test",
        method="majority",
        model_family="majority",
        stage="unit",
        folds=(0, 1, 2, 3, 4),
        seeds=(2753,),
        loss_id="training_fold_prior",
    )
    splits = load_splits(prepared_project.splits)
    valid = get_samples(splits, partition="development", target="season")
    protected_ids = splits.loc[~splits["partition"].eq("development"), "id"]

    outputs = run_or_load_experiment(config, mode="run", **paths)
    manifest = build_experiment_evidence(
        outputs,
        registry_path=paths["registry_path"],
        expected_ids=valid["id"],
        protected_ids=protected_ids,
        probability_note="Training-fold empirical class priors.",
        calibration_claim_allowed=True,
        evidence_directory=prepared_project.root / "results/evidence/task2/b0",
        figure_directory=prepared_project.root / "results/figures/task2",
    )

    assert manifest["folds"] == [0, 1, 2, 3, 4]
    assert manifest["coverage"]["row_count"] == len(valid)
    assert manifest["coverage"]["protected_id_count"] == 0
    assert len(manifest["run_ids"]) == 5
    assert len(set(manifest["run_ids"])) == 5
    assert manifest["calibration_claim_allowed"]
    assert len(manifest["oof_artifact_set_sha256"]) == 64
    assert Path(manifest["manifest_path"]).is_file()
    assert not Path(manifest["artifacts"]["figure"]["path"]).is_absolute()
    result_root = prepared_project.root / "results"
    figure_path = result_root / manifest["artifacts"]["figure"]["path"]
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    fold_metrics_path = result_root / manifest["artifacts"]["fold_metrics"]["path"]
    assert len(pd.read_csv(fold_metrics_path)) == 5

    cached = run_or_load_experiment(config, mode="run_or_load", **paths)
    cached_manifest = build_experiment_evidence(
        cached,
        registry_path=paths["registry_path"],
        expected_ids=valid["id"],
        protected_ids=protected_ids,
        probability_note="Training-fold empirical class priors.",
        calibration_claim_allowed=True,
        evidence_directory=prepared_project.root / "results/evidence/task2/b0",
        figure_directory=prepared_project.root / "results/figures/task2",
    )
    assert {output.source for output in cached} == {"cache"}
    assert cached_manifest["run_ids"] == manifest["run_ids"]
    assert cached_manifest["oof_artifact_set_sha256"] == manifest["oof_artifact_set_sha256"]
    assert len(RunRegistry(paths["registry_path"]).read()) == 5


def test_experiment_evidence_rejects_a_missing_fold(prepared_project) -> None:
    paths = _project_paths(prepared_project)
    config = ExperimentConfig(
        experiment_id="incomplete-evidence-test",
        method="majority",
        model_family="majority",
        stage="unit",
        folds=(0, 1, 2, 3),
        seeds=(2753,),
    )
    outputs = run_or_load_experiment(config, mode="run", **paths)

    with pytest.raises(ValueError, match="each expected fold exactly once"):
        build_experiment_evidence(
            outputs,
            registry_path=paths["registry_path"],
            expected_ids=[],
            probability_note="Training-fold empirical class priors.",
            calibration_claim_allowed=True,
            evidence_directory=prepared_project.root / "results/evidence/task2/incomplete",
            figure_directory=prepared_project.root / "results/figures/task2",
        )
