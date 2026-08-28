from __future__ import annotations

import json
from pathlib import Path

import pytest

from fashion.config import ROOT
from fashion.task2.class_balance import validate_i1_config
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
    mappings = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    season_labels = ["Fall", "Spring", "Summer", "Winter"]
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


def test_repository_b0_config_declares_all_canonical_folds_before_execution() -> None:
    config = load_experiment_config(ROOT / "configs/task2/b0_majority.json")

    assert config.experiment_id == "b0-majority"
    assert config.method == "majority"
    assert config.model_family == "majority"
    assert config.stage == "b0_baseline"
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.seeds == (2753,)
    assert config.loss_id == "training_fold_prior"


def test_repository_b1_config_freezes_unweighted_hog_hsv_svm_before_execution() -> None:
    config = load_experiment_config(ROOT / "configs/task2/b1_hog_hsv_svm.json")

    assert config.experiment_id == "b1-hog-hsv-svm"
    assert config.method == "hog_hsv_svm"
    assert config.model_family == "hog_hsv_svm"
    assert config.stage == "b1_baseline"
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.seeds == (2753,)
    assert config.loss_id == "linear_svc_hinge_unweighted"
    assert config.hog_hsv.image_size == (80, 60)
    assert config.hog_hsv.hog_orientations == 9
    assert config.hog_hsv.hog_pixels_per_cell == (8, 8)
    assert config.hog_hsv.hog_cells_per_block == (2, 2)
    assert config.hog_hsv.hsv_bins == (18, 8, 8)
    assert config.hog_hsv.svm_c == 1.0
    assert config.hog_hsv.max_iterations == 5_000


def test_repository_g1_configs_differ_only_by_family_and_experiment_id() -> None:
    paths = (
        ROOT / "configs/task2/g1_c1_smallcnn.json",
        ROOT / "configs/task2/g1_c2_resnet18.json",
        ROOT / "configs/task2/g1_c3_mobilenetv3.json",
    )
    configs = [load_experiment_config(path) for path in paths]
    expected_families = {
        "smallcnn",
        "resnet18_small_stem",
        "mobilenet_v3_small",
    }

    assert {config.model_family for config in configs} == expected_families
    assert {config.experiment_id for config in configs} == {
        "g1-c1-smallcnn",
        "g1-c2-resnet18",
        "g1-c3-mobilenetv3",
    }
    for config in configs:
        assert config.method == "deep"
        assert config.stage == "g1_family_screen"
        assert config.folds == (0, 1, 2, 3, 4)
        assert config.seeds == (2753,)
        assert config.loss_id == "cross_entropy"
        assert config.data.image_size == (80, 60)
        assert config.data.augmentation == "a0"
        assert config.data.batch_size == 32
        assert config.data.validation_batch_size == 128
        assert config.data.num_workers == 4
        assert config.optimisation.epochs == 8
        assert config.optimisation.patience == 8
        assert config.optimisation.learning_rate == 3e-4
        assert config.optimisation.weight_decay == 1e-4
        assert config.optimisation.effective_batch_size == 128
        assert config.optimisation.gradient_clip_norm == 1.0
        assert config.optimisation.warmup_epochs == 1.0
        assert config.optimisation.use_amp

    reference = configs[0].to_dict()
    reference.pop("experiment_id")
    reference.pop("model_family")
    for config in configs[1:]:
        candidate = config.to_dict()
        candidate.pop("experiment_id")
        candidate.pop("model_family")
        assert candidate == reference


def test_repository_g2_p1_changes_only_identity_stage_and_input_size() -> None:
    p0 = load_experiment_config(ROOT / "configs/task2/g1_c2_resnet18.json")
    p1 = load_experiment_config(ROOT / "configs/task2/g2_p1_c2_resnet18.json")

    assert p0.experiment_id == "g1-c2-resnet18"
    assert p0.stage == "g1_family_screen"
    assert p0.data.image_size == (80, 60)
    assert p1.experiment_id == "g2-p1-c2-resnet18"
    assert p1.stage == "g2_input_size_ablation"
    assert p1.data.image_size == (128, 96)

    p0_matched = p0.to_dict()
    p1_matched = p1.to_dict()
    for config in (p0_matched, p1_matched):
        config.pop("experiment_id")
        config.pop("stage")
        config["data"].pop("image_size")
    assert p1_matched == p0_matched


def test_repository_g2_a1_changes_only_identity_stage_and_augmentation() -> None:
    decision = json.loads(
        (ROOT / "results/evidence/task2/g2_input_size_ablation/decision.json").read_text(
            encoding="utf-8"
        )
    )
    a0 = load_experiment_config(ROOT / "configs/task2/g1_c2_resnet18.json")
    a1 = load_experiment_config(ROOT / "configs/task2/g2_a1_c2_resnet18.json")

    assert decision["selected_variant"] == "P0"
    assert decision["selected_experiment_id"] == a0.experiment_id
    assert a0.data.image_size == a1.data.image_size == (80, 60)
    assert a0.data.augmentation == "a0"
    assert a1.data.augmentation == "a1"
    assert a1.experiment_id == "g2-a1-c2-resnet18"
    assert a1.stage == "g2_augmentation_ablation"

    a0_matched = a0.to_dict()
    a1_matched = a1.to_dict()
    for config in (a0_matched, a1_matched):
        config.pop("experiment_id")
        config.pop("stage")
        config["data"].pop("augmentation")
    assert a1_matched == a0_matched


def test_repository_g2_tuning_changes_only_identity_stage_and_optimizer_pair() -> None:
    decision = json.loads(
        (ROOT / "results/evidence/task2/g2_augmentation_ablation/decision.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "smallcnn": {
            "t0": ("g1_c1_smallcnn.json", "g1-c1-smallcnn", 3e-4, 1e-4),
            "t1": ("g2_t1_c1_smallcnn.json", "g2-t1-c1-smallcnn", 1e-3, 1e-4),
            "t2": ("g2_t2_c1_smallcnn.json", "g2-t2-c1-smallcnn", 3e-4, 1e-3),
        },
        "resnet18_small_stem": {
            "t0": ("g1_c2_resnet18.json", "g1-c2-resnet18", 3e-4, 1e-4),
            "t1": ("g2_t1_c2_resnet18.json", "g2-t1-c2-resnet18", 1e-3, 1e-4),
            "t2": ("g2_t2_c2_resnet18.json", "g2-t2-c2-resnet18", 3e-4, 1e-3),
        },
    }

    assert decision["selected_variant"] == "A0"

    for family, variants in expected.items():
        loaded: dict[str, ExperimentConfig] = {}
        for tuning_id, (filename, experiment_id, learning_rate, weight_decay) in (
            variants.items()
        ):
            config = load_experiment_config(ROOT / "configs/task2" / filename)
            loaded[tuning_id] = config

            assert config.experiment_id == experiment_id
            assert config.model_family == family
            assert config.method == "deep"
            assert config.folds == (0, 1, 2, 3, 4)
            assert config.seeds == (2753,)
            assert config.loss_id == "cross_entropy"
            assert config.data.image_size == (80, 60)
            assert config.data.augmentation == "a0"
            assert config.optimisation.epochs == 8
            assert config.optimisation.patience == 8
            assert config.optimisation.learning_rate == learning_rate
            assert config.optimisation.weight_decay == weight_decay
            assert config.stage == (
                "g1_family_screen" if tuning_id == "t0" else "g2_compact_tuning"
            )

        reference = loaded["t0"].to_dict()
        reference.pop("experiment_id")
        reference.pop("stage")
        reference["optimisation"].pop("learning_rate")
        reference["optimisation"].pop("weight_decay")
        for config in loaded.values():
            candidate = config.to_dict()
            candidate.pop("experiment_id")
            candidate.pop("stage")
            candidate["optimisation"].pop("learning_rate")
            candidate["optimisation"].pop("weight_decay")
            assert candidate == reference


def test_repository_g3_full_budget_preserves_selected_g2_settings() -> None:
    decision = json.loads(
        (ROOT / "results/evidence/task2/g2_compact_tuning/decision.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "C1": (
            "g2_t1_c1_smallcnn.json",
            "g3_c1_t1_smallcnn.json",
            "g3-c1-t1-smallcnn",
            "smallcnn",
            1e-3,
        ),
        "C2": (
            "g1_c2_resnet18.json",
            "g3_c2_t0_resnet18.json",
            "g3-c2-t0-resnet18",
            "resnet18_small_stem",
            3e-4,
        ),
    }

    assert decision["families"]["C1"]["selected_tuning_id"] == "T1"
    assert decision["families"]["C2"]["selected_tuning_id"] == "T0"
    full_configs: list[ExperimentConfig] = []
    for family, (
        reference_name,
        full_name,
        experiment_id,
        model_family,
        learning_rate,
    ) in expected.items():
        reference = load_experiment_config(ROOT / "configs/task2" / reference_name)
        full = load_experiment_config(ROOT / "configs/task2" / full_name)
        full_configs.append(full)

        assert decision["families"][family]["selected_experiment_id"] == (
            reference.experiment_id
        )
        assert full.experiment_id == experiment_id
        assert full.stage == "g3_full_budget"
        assert full.model_family == model_family
        assert full.optimisation.learning_rate == learning_rate
        assert full.optimisation.weight_decay == 1e-4
        assert full.optimisation.epochs == 30
        assert full.optimisation.patience == 5
        assert full.folds == (0, 1, 2, 3, 4)
        assert full.seeds == (2753,)
        assert full.data.image_size == (80, 60)
        assert full.data.augmentation == "a0"

        reference_matched = reference.to_dict()
        full_matched = full.to_dict()
        for config in (reference_matched, full_matched):
            config.pop("experiment_id")
            config.pop("stage")
            config["optimisation"].pop("epochs")
            config["optimisation"].pop("patience")
        assert full_matched == reference_matched

    c1_matched, c2_matched = (config.to_dict() for config in full_configs)
    for config in (c1_matched, c2_matched):
        config.pop("experiment_id")
        config.pop("model_family")
        config["optimisation"].pop("learning_rate")
    assert c1_matched == c2_matched


def test_repository_i1_changes_only_identity_stage_and_loss_from_g3_c1() -> None:
    reference = load_experiment_config(
        ROOT / "configs/task2/g3_c1_t1_smallcnn.json"
    )
    i1 = load_experiment_config(
        ROOT / "configs/task2/g4_i1_effective_number_c1.json"
    )

    validate_i1_config(i1)
    assert i1.experiment_id == "g4-i1-effective-number-c1"
    assert i1.stage == "g4_i1_class_balanced"
    assert i1.loss_id == "effective_number_beta_0.9999"

    reference_matched = reference.to_dict()
    i1_matched = i1.to_dict()
    for config in (reference_matched, i1_matched):
        config.pop("experiment_id")
        config.pop("stage")
        config.pop("loss_id")
    assert i1_matched == reference_matched


def test_repository_pretraining_pair_changes_only_identity_and_initial_weights() -> None:
    small_stem = load_experiment_config(ROOT / "configs/task2/g3_c2_t0_resnet18.json")
    scratch = load_experiment_config(ROOT / "configs/task2/g4_p0s_resnet18_standard_scratch.json")
    pretrained = load_experiment_config(
        ROOT / "configs/task2/g4_pstar_resnet18_standard_pretrained.json"
    )

    assert scratch.experiment_id == "g4-p0s-resnet18-standard-scratch"
    assert scratch.model_family == "resnet18_standard_scratch"
    assert pretrained.experiment_id == "g4-pstar-resnet18-standard-pretrained"
    assert pretrained.model_family == "resnet18_standard_pretrained"
    for config in (scratch, pretrained):
        assert config.stage == "g4_pretraining_benchmark"
        assert config.method == "deep"
        assert config.folds == (0, 1, 2, 3, 4)
        assert config.seeds == (2753,)
        assert config.loss_id == "cross_entropy"
        assert config.data.image_size == (80, 60)
        assert config.data.augmentation == "a0"
        assert config.optimisation.epochs == 30
        assert config.optimisation.patience == 5
        assert config.optimisation.learning_rate == 3e-4
        assert config.optimisation.weight_decay == 1e-4

    matched = []
    for config in (small_stem, scratch, pretrained):
        payload = config.to_dict()
        payload.pop("experiment_id")
        payload.pop("stage")
        payload.pop("model_family")
        matched.append(payload)
    assert matched[0] == matched[1] == matched[2]
