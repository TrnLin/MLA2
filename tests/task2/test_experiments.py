from __future__ import annotations

import json
from pathlib import Path

import pytest

from fashion.task2.experiments import (
    DataRunConfig,
    ExperimentConfig,
    OptimisationRunConfig,
    load_experiment_config,
    run_matrix,
    run_or_load_experiment,
)
from fashion.train.registry import RunRegistry


def _paths(prepared_project) -> dict[str, Path]:
    root = prepared_project.root
    return {
        "data_root": root,
        "splits_path": prepared_project.splits,
        "label_map_path": prepared_project.label_maps,
        "registry_path": root / "results/runs.csv",
        "checkpoint_directory": root / "tmp/task2/checkpoints",
        "run_directory": root / "tmp/task2/runs",
    }


def _majority_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="b0-test",
        method="majority",
        model_family="majority",
        stage="unit",
        folds=(0,),
        seeds=(2753,),
    )


def _deep_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="c1-test",
        method="deep",
        model_family="smallcnn",
        stage="unit",
        folds=(0,),
        seeds=(2753,),
        data=DataRunConfig(
            image_size=(80, 60),
            augmentation="none",
            batch_size=2,
            validation_batch_size=2,
            num_workers=0,
            pin_memory=False,
        ),
        optimisation=OptimisationRunConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            effective_batch_size=2,
            warmup_epochs=0,
            patience=1,
            use_amp=False,
            device="cpu",
        ),
    )


def test_json_config_parser_is_strict_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "experiment_id": "c1-screen",
                "method": "deep",
                "model_family": "smallcnn",
                "stage": "g1",
                "folds": [0, 1],
                "seeds": [2753],
                "data": {"image_size": [80, 60], "augmentation": "a0"},
                "optimisation": {"epochs": 8},
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.folds == (0, 1)
    assert config.data.image_size == (80, 60)
    assert config.to_dict()["experiment_id"] == "c1-screen"

    invalid = json.loads(path.read_text(encoding="utf-8"))
    invalid["optimisation"]["learn_rate"] = 0.1
    with pytest.raises(ValueError, match="unknown optimisation fields"):
        ExperimentConfig.from_dict(invalid)


def test_majority_run_writes_registry_and_reuses_verified_cache(prepared_project) -> None:
    paths = _paths(prepared_project)

    first = run_or_load_experiment(_majority_config(), **paths)
    second = run_or_load_experiment(_majority_config(), **paths)

    assert len(first) == len(second) == 1
    assert first[0].source == "run"
    assert second[0].source == "cache"
    assert first[0].run_id == second[0].run_id
    assert len(RunRegistry(paths["registry_path"]).read()) == 1
    assert first[0].oof["id"].nunique() == len(first[0].oof)
    assert (prepared_project.root / "tmp/task2/runs" / first[0].run_id / "oof.csv").is_file()


def test_deep_run_records_checkpoint_oof_history_and_metrics(prepared_project) -> None:
    paths = _paths(prepared_project)

    output = run_or_load_experiment(_deep_config(), mode="run", **paths)[0]

    rows = RunRegistry(paths["registry_path"]).read()
    row = rows.loc[rows["run_id"].eq(output.run_id)].iloc[0]
    assert output.source == "run"
    assert row["status"] == "completed"
    assert row["scratch"] == "true"
    assert row["benchmark_only"] == "false"
    assert row["checkpoint_sha256"] == output.artifacts["checkpoint"]
    assert row["prediction_sha256"] == output.artifacts["prediction"]
    assert row["history_sha256"] == output.artifacts["history"]
    assert row["primary_metric_name"] == "macro_f1"
    assert int(row["epochs_completed"]) == 1
    assert len(output.cache_key.digest) == 64


def test_load_mode_fails_when_no_matching_verified_run(prepared_project) -> None:
    with pytest.raises(FileNotFoundError, match="no valid cached run"):
        run_or_load_experiment(_majority_config(), mode="load", **_paths(prepared_project))


def test_failed_execution_remains_in_registry(prepared_project, monkeypatch) -> None:
    def fail_training(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("fashion.task2.experiments.train_fold", fail_training)
    paths = _paths(prepared_project)

    with pytest.raises(RuntimeError, match="synthetic"):
        run_or_load_experiment(_deep_config(), mode="run", **paths)

    rows = RunRegistry(paths["registry_path"]).read()
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "failed"
    assert rows.loc[0, "error_type"] == "RuntimeError"
    assert rows.loc[0, "error_message"] == "synthetic training failure"


def test_matrix_rejects_duplicate_experiment_ids() -> None:
    config = _majority_config()
    with pytest.raises(ValueError, match="duplicate"):
        run_matrix([config, config])


def test_config_rejects_pretrained_family_on_final_method_boundary() -> None:
    config = ExperimentConfig(
        experiment_id="invalid",
        method="majority",
        model_family="resnet18_standard_pretrained",
        stage="unit",
    )
    with pytest.raises(ValueError, match="invalid for method"):
        config.validate()
