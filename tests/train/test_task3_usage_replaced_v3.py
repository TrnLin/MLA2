"""The v3 data change retains fresh MixUp + SAM fits and matched teacher controls."""

import copy
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as previous
from fashion.train import task3_usage_replaced_v3 as screen
from fashion.train.config import Task3BaselineConfig

pytest_plugins = ["test_task3_usage_expanded_v2"]


def test_recipe_changes_only_data_identity_and_resolves_sparse_parents():
    pytest.importorskip("torch")
    spec = screen.screen_spec()
    identity = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
    }
    assert {k: v for k, v in asdict(spec).items() if k not in identity} == {
        k: v for k, v in asdict(previous.screen_spec()).items() if k not in identity
    }
    assert spec.to_dict()["mixup_policy"] == previous.screen_spec().to_dict()["mixup_policy"]
    assert spec.to_dict()["sam_policy"] == previous.screen_spec().to_dict()["sam_policy"]
    assert spec.to_dict()["dataset"]["split_sha256"] != v2.SPLIT_SHA256
    assert spec.parent_run_id_for_fold(4) == screen.BASELINE_RUN_IDS[1]
    assert screen.screen_config(spec, fold=0, device_name="cuda").epochs == 30
    for fold in (1, 2, 3, False, True):
        with pytest.raises(ValueError, match="only folds 0 and 4"):
            screen.screen_config(spec, fold=fold, device_name="cuda")
    with pytest.raises(ValueError, match="CUDA GPU"):
        screen.screen_config(spec, fold=0, device_name="cpu")
    with pytest.raises(ValueError, match="frozen"):
        replace(spec, class_weight_cap=3)


@pytest.fixture
def trained_v3(tiny_v2, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from fashion.train import sam as sam_module
    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="usage"), epochs=1, batch_size=4, num_workers=0)
    for module in (engine, v2, previous, screen):
        monkeypatch.setattr(module, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(screen, "screen_config", lambda *args, **kwargs: config)
    policy = dict(sam_module.usage_policy(), diagnostic_epochs=[1])
    monkeypatch.setattr(sam_module, "usage_policy", lambda: dict(policy))
    monkeypatch.setattr(screen, "validate_dataset", lambda **kwargs: (tiny_v2, {"fixture": "v3"}))
    destination = tmp_path / screen.DATA_DIRECTORY
    shutil.copytree(tmp_path / v2.DATA_DIRECTORY, destination)
    monkeypatch.setattr(screen, "SPLIT_SHA256", compute_sha256(destination / "splits.csv"))
    monkeypatch.setattr(v2, "MAP_SHA256", compute_sha256(destination / "label_maps.json"))
    output, parents = tmp_path / "new_output", tmp_path / "parents"
    built_models = []
    builder = engine._build_task3_model

    def build(*args):
        model = builder(*args)
        built_models.append(model)
        return model

    def reject_load(*args, **kwargs):
        raise AssertionError("v3 must not load any old model weights")

    monkeypatch.setattr(engine, "_build_task3_model", build)
    monkeypatch.setattr(torch, "load", reject_load)
    runs = []
    for fold in screen.FOLDS:
        parent = parents / screen.screen_spec().parent_run_id_for_fold(fold)
        parent.mkdir(parents=True)
        v2.write_json({"run_id": parent.name, "validation_fold": fold}, parent / "metrics.json")
        for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
            (parent / name).write_text("Reference only; never load weights")
        runs.append(
            engine.run_task3_baseline_fold(
                "usage",
                fold,
                root=tmp_path,
                output_root=output,
                registry_path=screen.usage_registry_path(output),
                device_name="cpu",
                child_spec=screen.screen_spec(),
                parent_run_directory=parent,
            )
        )
    assert len(built_models) == 2 and built_models[0] is not built_models[1]
    return {"runs": runs, "output": output, "root": tmp_path, "splits": tiny_v2}


def test_fresh_fits_use_v3_paths_and_verified_sam_receipts(trained_v3):
    fixture = trained_v3
    for fold, run in zip(screen.FOLDS, fixture["runs"], strict=True):
        checked = screen.completed_fold(
            fold=fold,
            output_root=fixture["output"],
            registry_path=screen.usage_registry_path(fixture["output"]),
            splits=fixture["splits"],
            spec=screen.screen_spec(),
        )
        assert checked["run_id"] == run["run_id"]
        config = json.loads((Path(run["run_dir"]) / "config.json").read_text())
        assert config["scratch"] is True
        assert config["expanded_dataset"] == {"fixture": "v3"}
        assert config["mixup_contract"]["target"] == "usage"
        assert config["parent_run_id"] == screen.screen_spec().parent_run_id_for_fold(fold)
        assert "gender_label_variant" not in config
        assert sum(config["class_counts"]) == 16
    path = Path(fixture["runs"][0]["run_dir"]) / "sam_training.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        screen.completed_fold(
            fold=0,
            output_root=fixture["output"],
            registry_path=screen.usage_registry_path(fixture["output"]),
            splits=fixture["splits"],
            spec=screen.screen_spec(),
        )


def test_compare_only_same_teacher_ids_when_outside_ids_change(trained_v3, monkeypatch):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine

    fixture = trained_v3
    references = {}
    for fold, run in zip(screen.FOLDS, fixture["runs"], strict=True):
        predictions = v2.read_predictions(run["prediction_path"])
        predictions.loc[predictions.source_dataset.ne("teacher"), "id"] += 10000
        references[fold] = dict(run, predictions=predictions)
    monkeypatch.setattr(screen, "check_reference", lambda **kwargs: references)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_fit(*args, **kwargs):
        raise AssertionError("Fully verified completed v3 folds must be reused")

    monkeypatch.setattr(engine, "run_task3_baseline_fold", reject_fit)
    result = screen.run_usage_replaced_v3(
        root=fixture["root"],
        output_root=fixture["output"],
        baseline_directory="unused",
        baseline_registry_path="unused/results/runs.csv",
    )
    assert result["comparison"]["teacher_macro_f1_change"] == pytest.approx(0)
    assert result["comparison"]["sources"]["teacher"]["rows"] == 4
    assert len(result["comparison"]["teacher_per_class"]) == 9
    bad = copy.deepcopy(references)
    bad[0]["predictions"] = bad[0]["predictions"].iloc[1:]
    with pytest.raises(ValueError, match="same validation IDs"):
        screen.build_comparison(
            fixture["runs"], splits=fixture["splits"], references=bad, contract={}
        )
    for folds in ((0,), (0, 1, 2, 3, 4), (4, 0), (False, 4)):
        with pytest.raises(ValueError, match="exactly folds 0 and 4"):
            screen.run_usage_replaced_v3(
                root=fixture["root"],
                output_root=fixture["output"],
                baseline_directory="unused",
                baseline_registry_path="unused/results/runs.csv",
                folds=folds,
            )


def test_frozen_dataset_rejects_changed_split_before_reading_images(tmp_path):
    path = tmp_path / "data/processed/splits.csv"
    path.parent.mkdir(parents=True)
    path.write_text("id,path\n1,tampered.png\n")
    with pytest.raises(ValueError, match="Reviewed Usage v3 data changed"):
        screen.validate_dataset(root=tmp_path)
