from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fashion.data.dataset import get_cv_split, get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2 import multitask
from fashion.task2.experiments import ExperimentConfig
from fashion.task2.multitask import (
    I2_AUXILIARY_TARGET,
    I2_DATA_CONFIG,
    I2_IMPLEMENTATION_PATHS,
    I2_MASK_POLICY,
    I2_OPTIMISATION_CONFIG,
    I2_STAGE,
    AuxiliaryRunConfig,
    I2ExperimentConfig,
    load_i2_config,
    run_i2_matrix,
    run_or_load_i2_experiment,
    validate_i2_config,
)
from fashion.train.engine import FoldResult
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics
from fashion.train.registry import RunRegistry

EXPERIMENT_ID = "g4-i2-article-type-lambda-0-1-c1"
LOSS_ID = "season_ce_plus_masked_article_type_ce_lambda_0_1"


def _config() -> I2ExperimentConfig:
    return I2ExperimentConfig(
        base=ExperimentConfig(
            experiment_id=EXPERIMENT_ID,
            method="deep",
            model_family="smallcnn",
            stage=I2_STAGE,
            folds=tuple(range(5)),
            seeds=(2753,),
            loss_id=LOSS_ID,
            data=I2_DATA_CONFIG,
            optimisation=I2_OPTIMISATION_CONFIG,
        ),
        auxiliary=AuxiliaryRunConfig(
            target=I2_AUXILIARY_TARGET,
            loss_weight=0.1,
            mask_policy=I2_MASK_POLICY,
        ),
    )


def _paths(prepared_project) -> dict[str, Path]:
    mappings = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    season_labels = list(SEASON_LABELS)
    mappings["season"].update(
        {
            "num_classes": len(season_labels),
            "classes": season_labels,
            "label_to_index": {
                label: index for index, label in enumerate(season_labels)
            },
            "index_to_label": {
                str(index): label for index, label in enumerate(season_labels)
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
        "source_root": Path(multitask.__file__).resolve().parents[3],
        "splits_path": prepared_project.splits,
        "label_map_path": prepared_project.label_maps,
        "registry_path": root / "results/runs.csv",
        "checkpoint_directory": root / "tmp/task2/checkpoints",
        "run_directory": root / "tmp/task2/runs",
    }


def _fake_execute_i2(
    *,
    config: I2ExperimentConfig,
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
    targets = np.asarray([SEASON_LABELS.index(str(value)) for value in validation["season"]])
    probabilities = np.full((len(validation), len(SEASON_LABELS)), 0.02, dtype=np.float64)
    probabilities[np.arange(len(validation)), targets] = 0.94
    metrics = multiclass_metrics(
        validation["season"].astype(str),
        probabilities=probabilities,
        labels=SEASON_LABELS,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(f"fold={fold};seed={seed}".encode())
    result = FoldResult(
        fold=fold,
        seed=seed,
        labels=SEASON_LABELS,
        best_epoch=1,
        epochs_completed=1,
        best_macro_f1=1.0,
        best_metrics=metrics,
        history=[
            {
                "epoch": 1,
                "train_loss": 0.1,
                "validation_loss": 0.1,
                "validation_macro_f1": 1.0,
                "validation_accuracy": 1.0,
                "learning_rate": 1e-3,
            }
        ],
        validation_ids=[int(value) for value in validation["id"]],
        targets=targets,
        probabilities=probabilities,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=compute_sha256(checkpoint_path),
        parameter_count=1_200_000,
        runtime_seconds=0.1,
        peak_vram_mb=1.0,
        stopped_early=False,
        device="cpu",
    )
    return result, {
        "auxiliary": {"target": I2_AUXILIARY_TARGET, "loss_weight": 0.1},
        "epoch_history": result.history,
    }


def test_validate_i2_config_freezes_identity_weight_and_training_protocol() -> None:
    config = _config()

    validate_i2_config(config)
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_i2_config(
            replace(
                config,
                auxiliary=replace(config.auxiliary, loss_weight=0.3),
            )
        )
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_i2_config(
            replace(config, base=replace(config.base, folds=(0,)))
        )
    with pytest.raises(ValueError, match="auxiliary target"):
        validate_i2_config(
            replace(
                config,
                auxiliary=replace(config.auxiliary, target="gender"),
            )
        )


def test_repository_i2_configs_change_only_declared_multitask_fields() -> None:
    paths = (
        Path("configs/task2/g4_i2_article_type_lambda_0_1_c1.json"),
        Path("configs/task2/g4_i2_article_type_lambda_0_3_c1.json"),
    )
    configs = [load_i2_config(path) for path in paths]
    reference = ExperimentConfig.from_dict(
        json.loads(
            Path("configs/task2/g3_c1_t1_smallcnn.json").read_text(encoding="utf-8")
        )
    )

    assert [config.experiment_id for config in configs] == [
        "g4-i2-article-type-lambda-0-1-c1",
        "g4-i2-article-type-lambda-0-3-c1",
    ]
    assert [config.auxiliary.loss_weight for config in configs] == [0.1, 0.3]
    assert {config.auxiliary.target for config in configs} == {"articleType"}
    assert {config.auxiliary.mask_policy for config in configs} == {
        "available_labels_only"
    }

    reference_matched = reference.to_dict()
    for field in ("experiment_id", "stage", "loss_id"):
        reference_matched.pop(field)
    for config in configs:
        candidate = config.base.to_dict()
        for field in ("experiment_id", "stage", "loss_id"):
            candidate.pop(field)
        assert candidate == reference_matched


def test_i2_config_parser_rejects_unknown_auxiliary_field(tmp_path: Path) -> None:
    source = Path("configs/task2/g4_i2_article_type_lambda_0_1_c1.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["auxiliary"]["inference_input"] = True
    invalid = tmp_path / "invalid-i2.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown I2 auxiliary fields"):
        load_i2_config(invalid)


def test_i2_cache_hash_covers_only_shared_and_multitask_implementation() -> None:
    required = {
        "src/fashion/data/multitask.py",
        "src/fashion/train/multitask.py",
        "src/fashion/task2/multitask.py",
        "src/fashion/models/season.py",
        "src/fashion/data/torch.py",
    }

    assert required <= set(I2_IMPLEMENTATION_PATHS)
    assert "src/fashion/task2/evidence.py" not in I2_IMPLEMENTATION_PATHS
    assert "src/fashion/task2/class_balance.py" not in I2_IMPLEMENTATION_PATHS


def test_execute_i2_wires_declared_auxiliary_head_and_weight(
    prepared_project,
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(prepared_project)
    splits = load_splits(paths["splits_path"])
    _, validation = get_cv_split(splits, 0)
    validation = get_samples(validation, target="season").reset_index(drop=True)
    validation_ids = tuple(int(value) for value in validation["id"])
    loaders = SimpleNamespace(
        train=("train-loader",),
        validation=("validation-loader",),
        labels=SEASON_LABELS,
        auxiliary_labels=("A", "C", "D"),
        validation_ids=validation_ids,
        audit=lambda: {"auxiliary_target": I2_AUXILIARY_TARGET},
    )
    monkeypatch.setattr(multitask, "build_multitask_loaders", lambda **kwargs: loaders)
    captured: dict[str, object] = {}
    fake_model = object()

    def fake_build(spec, *, article_type_classes):
        captured["model_family"] = spec.family
        captured["season_classes"] = spec.num_classes
        captured["article_type_classes"] = article_type_classes
        return fake_model

    monkeypatch.setattr(multitask, "build_multitask_season_model", fake_build)
    monkeypatch.setattr(multitask, "assert_final_model", lambda model: {})
    monkeypatch.setattr(
        multitask,
        "model_boundary_audit",
        lambda model: {"training_origin": "scratch"},
    )

    def fake_train(model, train_loader, validation_loader, **kwargs):
        captured["model"] = model
        captured["train_loader"] = train_loader
        captured["validation_loader"] = validation_loader
        captured["auxiliary_weight"] = kwargs["auxiliary_weight"]
        targets = np.asarray(
            [SEASON_LABELS.index(str(value)) for value in validation["season"]]
        )
        probabilities = np.full(
            (len(validation), len(SEASON_LABELS)),
            0.02,
            dtype=np.float64,
        )
        probabilities[np.arange(len(validation)), targets] = 0.94
        metrics = multiclass_metrics(
            validation["season"].astype(str),
            probabilities=probabilities,
            labels=SEASON_LABELS,
        )
        return FoldResult(
            fold=0,
            seed=2753,
            labels=SEASON_LABELS,
            best_epoch=1,
            epochs_completed=1,
            best_macro_f1=float(metrics["macro_f1"]),
            best_metrics=metrics,
            history=[],
            validation_ids=list(validation_ids),
            targets=targets,
            probabilities=probabilities,
            checkpoint_path=str(kwargs["checkpoint_path"]),
            checkpoint_sha256="a" * 64,
            parameter_count=1,
            runtime_seconds=0.1,
            peak_vram_mb=None,
            stopped_early=False,
            device="cpu",
            metadata={"selection_metric": "season_macro_f1"},
        )

    monkeypatch.setattr(multitask, "train_masked_multitask_fold", fake_train)

    _, history = multitask._execute_i2(
        config=_config(),
        fold=0,
        seed=2753,
        checkpoint_path=tmp_path / "i2.pt",
        data_root=prepared_project.root,
        splits_path=paths["splits_path"],
        label_map_path=paths["label_map_path"],
    )

    assert captured == {
        "model_family": "smallcnn",
        "season_classes": 4,
        "article_type_classes": 3,
        "model": fake_model,
        "train_loader": loaders.train,
        "validation_loader": loaders.validation,
        "auxiliary_weight": 0.1,
    }
    assert history["auxiliary"]["target"] == I2_AUXILIARY_TARGET
    assert history["inference_boundary"] == "image_only_predict_season_logits"


def test_i2_runner_records_five_runs_and_reuses_verified_cache(
    prepared_project,
    monkeypatch,
) -> None:
    monkeypatch.setattr(multitask, "_execute_i2", _fake_execute_i2)
    paths = _paths(prepared_project)

    first = run_or_load_i2_experiment(_config(), **paths)
    second = run_or_load_i2_experiment(_config(), **paths)
    rows = RunRegistry(paths["registry_path"]).read()

    assert len(first) == len(second) == len(rows) == 5
    assert {output.source for output in first} == {"run"}
    assert {output.source for output in second} == {"cache"}
    assert [output.run_id for output in first] == [output.run_id for output in second]
    assert set(rows["fold"].astype(int)) == set(range(5))
    assert set(rows["loss_id"]) == {LOSS_ID}
    assert set(rows["status"]) == {"completed"}
    assert set(rows["scratch"]) == {"true"}
    assert set(rows["benchmark_only"]) == {"false"}
    assert set(rows["final_eligible"]) == {"true"}


def test_i2_failed_run_remains_in_registry(prepared_project, monkeypatch) -> None:
    def fail_i2(**kwargs):
        del kwargs
        raise RuntimeError("synthetic I2 failure")

    monkeypatch.setattr(multitask, "_execute_i2", fail_i2)
    paths = _paths(prepared_project)

    with pytest.raises(RuntimeError, match="synthetic I2 failure"):
        run_or_load_i2_experiment(_config(), mode="run", **paths)

    rows = RunRegistry(paths["registry_path"]).read()
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "failed"
    assert rows.loc[0, "error_type"] == "RuntimeError"


def test_i2_matrix_rejects_duplicate_experiment_ids() -> None:
    config = _config()
    with pytest.raises(ValueError, match="duplicate"):
        run_i2_matrix([config, config])
