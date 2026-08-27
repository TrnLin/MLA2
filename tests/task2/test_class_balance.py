from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fashion.data.dataset import get_cv_split, get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2 import class_balance
from fashion.task2.class_balance import (
    I1_DATA_CONFIG,
    I1_EXPERIMENT_ID,
    I1_IMPLEMENTATION_PATHS,
    I1_OPTIMISATION_CONFIG,
    I1_STAGE,
    fit_i1_fold_balance,
    run_or_load_i1_experiment,
    validate_i1_config,
)
from fashion.task2.experiments import ExperimentConfig, _implementation_paths
from fashion.train.artifacts import canonical_sha256
from fashion.train.engine import FoldResult
from fashion.train.losses import EFFECTIVE_NUMBER_LOSS_ID
from fashion.train.metrics import SEASON_LABELS
from fashion.train.registry import RunRegistry


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=I1_EXPERIMENT_ID,
        method="deep",
        model_family="smallcnn",
        stage=I1_STAGE,
        folds=tuple(range(5)),
        seeds=(2753,),
        loss_id=EFFECTIVE_NUMBER_LOSS_ID,
        data=I1_DATA_CONFIG,
        optimisation=I1_OPTIMISATION_CONFIG,
    )


def _paths(prepared_project) -> dict[str, Path]:
    mappings = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    mappings["season"].update(
        {
            "num_classes": len(SEASON_LABELS),
            "classes": list(SEASON_LABELS),
            "label_to_index": {
                label: index for index, label in enumerate(SEASON_LABELS)
            },
            "index_to_label": {
                str(index): label for index, label in enumerate(SEASON_LABELS)
            },
        }
    )
    prepared_project.label_maps.write_text(
        json.dumps(mappings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root = prepared_project.root
    return {
        "data_root": root,
        "source_root": Path(class_balance.__file__).resolve().parents[3],
        "splits_path": prepared_project.splits,
        "label_map_path": prepared_project.label_maps,
        "registry_path": root / "results/runs.csv",
        "checkpoint_directory": root / "tmp/task2/checkpoints",
        "run_directory": root / "tmp/task2/runs",
    }


def _fake_execute_i1(
    *,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    checkpoint_path: Path,
    data_root: Path,
    splits_path: Path,
    label_map_path: Path,
) -> tuple[FoldResult, dict]:
    del data_root, label_map_path
    splits = load_splits(splits_path)
    _, validation = get_cv_split(splits, fold)
    validation = get_samples(validation, target=config.target).reset_index(drop=True)
    labels = SEASON_LABELS
    targets = np.asarray([labels.index(str(value)) for value in validation["season"]])
    probabilities = np.full((len(validation), len(labels)), 0.02, dtype=np.float64)
    probabilities[np.arange(len(validation)), targets] = 0.94
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(f"fold={fold};seed={seed}".encode())
    result = FoldResult(
        fold=fold,
        seed=seed,
        labels=labels,
        best_epoch=1,
        epochs_completed=1,
        best_macro_f1=1.0,
        best_metrics={"macro_f1": 1.0},
        history=[
            {
                "epoch": 1,
                "learning_rate": 1e-3,
                "train_loss": 0.1,
                "validation_loss": 0.1,
                "validation_accuracy": 1.0,
                "validation_macro_f1": 1.0,
            }
        ],
        validation_ids=[int(value) for value in validation["id"]],
        targets=targets,
        probabilities=probabilities,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=compute_sha256(checkpoint_path),
        parameter_count=1_174_244,
        runtime_seconds=0.1,
        peak_vram_mb=1.0,
        stopped_early=False,
        device="cpu",
    )
    return result, {
        "class_balance": {"loss_id": EFFECTIVE_NUMBER_LOSS_ID},
        "epoch_history": result.history,
    }


def test_i1_implementation_paths_extend_without_changing_legacy_deep_paths() -> None:
    legacy = _implementation_paths("deep")

    assert I1_IMPLEMENTATION_PATHS[: len(legacy)] == legacy
    assert I1_IMPLEMENTATION_PATHS[len(legacy) :] == (
        "src/fashion/train/losses.py",
        "src/fashion/task2/class_balance.py",
    )


def test_validate_i1_config_freezes_the_full_protocol() -> None:
    config = _config()

    validate_i1_config(config)
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_i1_config(replace(config, loss_id="cross_entropy"))
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_i1_config(replace(config, folds=(0,)))
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_i1_config(
            replace(config, optimisation=replace(config.optimisation, epochs=29))
        )


def test_fold_balance_uses_only_canonical_training_rows(tmp_path: Path) -> None:
    splits = load_splits()
    training, validation = get_cv_split(splits, 0)
    training = get_samples(training, target="season").reset_index(drop=True)
    training_ids = tuple(int(value) for value in training["id"])
    stats = SimpleNamespace(training_id_sha256=canonical_sha256(sorted(training_ids)))
    loaders = SimpleNamespace(
        training_ids=training_ids,
        labels=SEASON_LABELS,
        stats=stats,
    )
    baseline = fit_i1_fold_balance(loaders, validation_fold=0)

    changed_validation = splits.copy()
    validation_ids = set(int(value) for value in validation["id"])
    changed_validation.loc[
        changed_validation["id"].isin(validation_ids), "season"
    ] = "Spring"
    validation_path = tmp_path / "validation_changed.csv"
    changed_validation.to_csv(validation_path, index=False)
    validation_only = fit_i1_fold_balance(
        loaders,
        validation_fold=0,
        splits_path=validation_path,
    )

    changed_training = splits.copy()
    summer_index = training.index[training["season"].eq("Summer")][0]
    training_id = int(training.loc[summer_index, "id"])
    changed_training.loc[changed_training["id"].eq(training_id), "season"] = "Spring"
    training_path = tmp_path / "training_changed.csv"
    changed_training.to_csv(training_path, index=False)
    training_changed = fit_i1_fold_balance(
        loaders,
        validation_fold=0,
        splits_path=training_path,
    )

    assert validation_only.class_counts == baseline.class_counts
    assert validation_only.class_weights == baseline.class_weights
    assert training_changed.class_counts != baseline.class_counts
    assert training_changed.class_weights != baseline.class_weights
    assert baseline.class_weights[SEASON_LABELS.index("Spring")] == max(
        baseline.class_weights
    )


def test_fold_balance_rejects_loader_training_id_drift() -> None:
    splits = load_splits()
    training, _ = get_cv_split(splits, 0)
    training = get_samples(training, target="season").reset_index(drop=True)
    training_ids = tuple(int(value) for value in training["id"])
    loaders = SimpleNamespace(
        training_ids=training_ids[:-1],
        labels=SEASON_LABELS,
        stats=SimpleNamespace(
            training_id_sha256=canonical_sha256(sorted(training_ids[:-1]))
        ),
    )

    with pytest.raises(ValueError, match="do not match the training loader"):
        fit_i1_fold_balance(loaders, validation_fold=0)


def test_i1_runner_records_five_runs_and_reuses_verified_cache(
    prepared_project,
    monkeypatch,
) -> None:
    monkeypatch.setattr(class_balance, "_execute_i1", _fake_execute_i1)
    paths = _paths(prepared_project)

    first = run_or_load_i1_experiment(_config(), **paths)
    second = run_or_load_i1_experiment(_config(), **paths)
    rows = RunRegistry(paths["registry_path"]).read()

    assert len(first) == len(second) == len(rows) == 5
    assert {output.source for output in first} == {"run"}
    assert {output.source for output in second} == {"cache"}
    assert [output.run_id for output in first] == [output.run_id for output in second]
    assert set(rows["fold"].astype(int)) == set(range(5))
    assert set(rows["loss_id"]) == {EFFECTIVE_NUMBER_LOSS_ID}
    assert set(rows["status"]) == {"completed"}
    assert set(rows["scratch"]) == {"true"}
    assert set(rows["benchmark_only"]) == {"false"}
    assert set(rows["final_eligible"]) == {"true"}
    assert all(
        {"checkpoint", "prediction", "history"} == set(output.artifacts)
        for output in second
    )


def test_i1_failed_run_remains_in_registry(prepared_project, monkeypatch) -> None:
    def fail_i1(**kwargs):
        del kwargs
        raise RuntimeError("synthetic I1 failure")

    monkeypatch.setattr(class_balance, "_execute_i1", fail_i1)
    paths = _paths(prepared_project)

    with pytest.raises(RuntimeError, match="synthetic I1 failure"):
        run_or_load_i1_experiment(_config(), mode="run", **paths)

    rows = RunRegistry(paths["registry_path"]).read()
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "failed"
    assert rows.loc[0, "error_type"] == "RuntimeError"


def test_i1_load_mode_fails_without_a_verified_cache(prepared_project) -> None:
    with pytest.raises(FileNotFoundError, match="no valid cached run"):
        run_or_load_i1_experiment(
            _config(),
            mode="load",
            **_paths(prepared_project),
        )
