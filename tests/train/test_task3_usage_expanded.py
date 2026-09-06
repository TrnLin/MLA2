"""Combined-data training boundaries, E8 controls and persisted diagnostics."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.config import ROOT, TARGET_COLUMNS
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded as expanded
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_decisions import probability_columns
from fashion.train.task3_experiments import usage_translation_2px_spec


@pytest.fixture
def tiny_data(tmp_path):
    rows = []
    rng = np.random.default_rng(2753)
    for fold in range(5):
        for source, label in (("teacher", "NA"), ("added", "Home"), ("teacher", "Casual")):
            image_id = len(rows) + 1
            path = tmp_path / f"image_{image_id}.png"
            Image.fromarray(rng.integers(0, 220, (80, 60, 3), dtype=np.uint8)).save(path)
            row = {
                "id": image_id,
                "path": path.name,
                "sha256": compute_sha256(path),
                "product_family_group": f"family_{image_id}",
                "duplicate_group": f"image_{image_id}",
                "product_name_key": f"item {image_id}",
                "partition": "development",
                "cv_fold": fold,
                "source_dataset": source,
                "is_cross_role_exact_duplicate": False,
                "is_cross_role_near_duplicate": False,
                "has_conflicting_target_labels": False,
                "conflicting_targets": "",
                "quarantine_reason": "",
            }
            for target in TARGET_COLUMNS:
                row[target] = label if target == "usage" else ""
                row[f"has_{target}_label"] = target == "usage"
            rows.append(row)
    protected = dict(
        rows[0],
        id=999,
        partition="holdout",
        cv_fold="",
        product_family_group="protected",
        duplicate_group="protected",
        product_name_key="protected",
        sha256="protected",
        path="sealed_must_not_be_read.png",
    )
    protected.update({f"has_{t}_label": False for t in TARGET_COLUMNS})
    protected.update({t: "" for t in TARGET_COLUMNS})
    rows.append(protected)
    path = tmp_path / expanded.DATA_DIRECTORY
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "splits.csv", index=False)
    maps = {
        "usage": {
            "source_scope": "development",
            "classes": list(expanded.CLASSES),
            "label_to_index": {c: i for i, c in enumerate(expanded.CLASSES)},
        }
    }
    expanded.write_json(maps, path / "label_maps.json")
    # Aggregation uses the canonical map; both versions must use identical classes.
    expanded.write_json(maps, tmp_path / "data/processed/label_maps.json")
    return load_splits(path / "splits.csv")


def predictions_for(frame, run_id="fixture"):
    result = frame[["id", "cv_fold", "product_family_group", "path"]].copy()
    labels = frame.usage.map({c: i for i, c in enumerate(expanded.CLASSES)})
    result["run_id"] = run_id
    result["true_index"] = labels
    result["true_label"] = frame.usage
    result["predicted_index"] = labels
    result["predicted_label"] = frame.usage
    result["confidence"] = 0.92
    probabilities = np.full((len(frame), 9), 0.01)
    probabilities[np.arange(len(frame)), labels.to_numpy()] = 0.92
    result[probability_columns(expanded.CLASSES)] = probabilities
    return result


def test_recipe_reuses_e8_and_is_frozen():
    spec = expanded.expanded_usage_spec()
    parent = asdict(usage_translation_2px_spec(expanded.E8_RUN_IDS))
    identity = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
    }
    assert {k: v for k, v in asdict(spec).items() if k not in identity} == {
        k: v for k, v in parent.items() if k not in identity
    }
    assert spec.to_dict()["dataset"]["split_sha256"] == expanded.SPLIT_SHA256
    with pytest.raises(FrozenInstanceError):
        spec.classifier_dropout = 0.2
    with pytest.raises(ValueError, match="frozen E8"):
        replace(spec, class_weight_beta=0.9)


def test_saved_folds_include_added_images_and_seal_holdout(tiny_data):
    for fold in range(5):
        training, validation = expanded.training_scope(tiny_data, fold)
        assert len(training) == 12 and len(validation) == 3
        assert training.source_dataset.eq("added").sum() == 4
        assert validation.source_dataset.eq("added").sum() == 1
        assert 999 not in set(training.id) | set(validation.id)
        assert not set(training.product_family_group) & set(validation.product_family_group)


def test_source_scores_preserve_na_and_all_nine_classes(tiny_data):
    _, validation = expanded.training_scope(tiny_data, 0)
    predictions = expanded.source_predictions(predictions_for(validation), validation)
    scores = expanded.source_metrics(predictions)
    assert [scores[c]["rows"] for c in ("combined", "teacher", "added")] == [3, 2, 1]
    assert scores["teacher"]["metrics"]["macro_f1"] == pytest.approx(2 / 9)
    assert predictions.true_label.eq("NA").sum() == 1
    with pytest.raises(ValueError, match="exactly cover"):
        expanded.source_predictions(predictions.iloc[:-1], validation)


def test_real_dataset_and_e8_sources_are_intact():
    if not (ROOT / expanded.DATA_DIRECTORY / "splits.csv").is_file():
        pytest.skip("Reviewed local dataset is not bundled in a source-only checkout")
    splits, contract = expanded.validate_dataset(check_images=False)
    assert len(splits) == 38732
    assert sum(row["added_validation_rows"] for row in contract["folds"]) == 120
    directory = ROOT / "results/evidence/task3" / expanded.E8_DIRECTORY
    if not directory.is_dir():
        pytest.skip("Saved GPU evidence is not bundled in a source-only checkout")
    sources = expanded.check_e8_sources(
        directory=directory, registry_path=ROOT / "results/runs.csv"
    )
    pooled = pd.concat([s["predictions"] for s in sources.values()])
    assert expanded.oof_metrics(pooled, expanded.CLASSES)["macro_f1"] == pytest.approx(
        0.41939320264210234
    )


@pytest.fixture
def trained_usage_fold(tiny_data, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="usage"), epochs=1, batch_size=4, num_workers=0)
    monkeypatch.setattr(engine, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(expanded, "validate_dataset", lambda **kwargs: (tiny_data, {"test": True}))
    parent = tmp_path / "parent"
    parent.mkdir()
    expanded.write_json(
        {"run_id": expanded.E8_RUN_IDS[0], "validation_fold": 0}, parent / "metrics.json"
    )
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (parent / name).write_text("Parent weights must never be loaded")
    result = engine.run_task3_baseline_fold(
        "usage",
        0,
        root=tmp_path,
        output_root=tmp_path / "output",
        registry_path=tmp_path / "runs.csv",
        device_name="cpu",
        child_spec=expanded.expanded_usage_spec(),
        parent_run_directory=parent,
    )
    return result, config


def test_real_training_step_and_registry_use_combined_data(trained_usage_fold, tmp_path):
    import torch

    result, _ = trained_usage_fold
    path = tmp_path / "output" / expanded.ARTIFACT_DIRECTORY / "usage" / result["run_id"]
    registry = pd.read_csv(tmp_path / "runs.csv", keep_default_na=False)
    assert registry.status.tolist() == ["complete"]
    assert registry.split_digest.item() == compute_sha256(
        tmp_path / expanded.DATA_DIRECTORY / "splits.csv"
    )
    assert registry.training_product_count.item() == 12
    assert result["metrics"]["selected_epoch"] == 1
    assert result["metrics"]["class_counts"][3] == 4  # Home includes all added training rows.
    scores = json.loads((path / "source_metrics.json").read_text())
    assert scores["validation"]["teacher"]["rows"] == 2
    assert scores["validation"]["added"]["rows"] == 1
    assert scores["clean_training"]["added"]["rows"] == 4
    predictions = expanded.read_predictions(path / "oof_predictions.csv")
    assert predictions.source_dataset.eq("added").sum() == 1
    assert set(expanded.read_predictions(path / "training_predictions.csv").id).isdisjoint(
        predictions.id
    )
    checkpoint = torch.load(path / "final_epoch.pt", map_location="cpu", weights_only=False)
    assert checkpoint["class_names"] == list(expanded.CLASSES)
    assert all(
        v.item() > 0
        for k, v in checkpoint["model_state_dict"].items()
        if k.endswith("num_batches_tracked")
    )
    for name, digest in result["metrics"]["expanded_artifact_sha256"].items():
        assert compute_sha256(path / name) == digest


def test_prepared_padding_does_not_enter_training_statistics(tmp_path):
    pytest.importorskip("torch")
    from fashion.train.data import fit_fold_rgb_stats

    pixels = np.full((80, 60, 3), 255, dtype=np.uint8)
    pixels[20:60, 10:50] = np.indices((40, 40))[0][:, :, None] + 40
    Image.fromarray(pixels).save(tmp_path / "prepared.png")
    training = pd.DataFrame(
        [
            {
                "path": "prepared.png",
                "content_left": 10,
                "content_top": 20,
                "content_width": 40,
                "content_height": 40,
            }
        ]
    )
    stats = fit_fold_rgb_stats(training, root=tmp_path)
    assert stats["mean"] == pytest.approx([59.5 / 255] * 3)
    training.loc[0, "content_width"] = 60
    with pytest.raises(ValueError, match="outside"):
        fit_fold_rgb_stats(training, root=tmp_path)


def test_resume_requires_intact_complete_matching_artifacts(tiny_data, tmp_path):
    from fashion.train.registry import RunRegistry

    spec = expanded.expanded_usage_spec()
    run_id = "complete_fixture"
    path = tmp_path / expanded.ARTIFACT_DIRECTORY / "usage" / run_id
    path.mkdir(parents=True)
    _, validation = expanded.training_scope(tiny_data, 0)
    predictions = expanded.source_predictions(predictions_for(validation, run_id), validation)
    predictions.to_csv(path / "oof_predictions.csv", index=False)
    (path / "final_epoch.pt").write_bytes(b"fixture weights")
    expanded.write_json(
        {**Task3BaselineConfig(target="usage").to_dict(), "child_experiment": spec.to_dict()},
        path / "config.json",
    )
    (path / "source_metrics.json").write_text("{}")
    metrics = {
        "selected_epoch": 30,
        "expanded_artifact_sha256": {
            "source_metrics.json": compute_sha256(path / "source_metrics.json")
        },
    }
    expanded.write_json(metrics, path / "metrics.json")
    registry = RunRegistry(tmp_path / "runs.csv")
    registry.start(
        {
            "run_id": run_id,
            "experiment_id": expanded.EXPERIMENT,
            "validation_fold": 0,
            "split_digest": expanded.SPLIT_SHA256,
            "label_map_digest": expanded.MAP_SHA256,
        }
    )
    arguments = dict(
        fold=0,
        output_root=tmp_path,
        registry_path=tmp_path / "runs.csv",
        splits=tiny_data,
        spec=spec,
    )
    assert expanded.reusable_fold(**arguments) is None
    registry.complete(
        run_id,
        {
            "metrics_json": metrics,
            "checkpoint_sha256": compute_sha256(path / "final_epoch.pt"),
            "prediction_sha256": compute_sha256(path / "oof_predictions.csv"),
        },
    )
    assert expanded.reusable_fold(**arguments)["run_id"] == run_id
    (path / "source_metrics.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="diagnostic changed"):
        expanded.reusable_fold(**arguments)


@pytest.mark.parametrize("shared_registry", [False, True])
def test_five_fold_summary_compares_only_matching_teacher_ids(
    tiny_data, tmp_path, monkeypatch, shared_registry
):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine
    from fashion.train import task3_usage_registry as recovery

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(expanded, "validate_dataset", lambda **kwargs: (tiny_data, {"test": True}))
    sources = {}
    for fold in range(5):
        _, validation = expanded.training_scope(tiny_data, fold)
        teacher = validation.loc[validation.source_dataset.eq("teacher")]
        sources[fold] = {
            "run_id": expanded.E8_RUN_IDS[fold],
            "sha256": {},
            "metrics": {},
            "predictions": predictions_for(teacher),
        }
    monkeypatch.setattr(expanded, "check_e8_sources", lambda **kwargs: sources)
    private = recovery.usage_registry_path(tmp_path / "output")
    migration_calls = []

    def migrate(**kwargs):
        migration_calls.append(kwargs)
        return {"registry_path": str(private)}

    monkeypatch.setattr(recovery, "prepare_usage_registry", migrate)

    def completed_fold(target, fold, **kwargs):
        _, validation = expanded.training_scope(tiny_data, fold)
        path = tmp_path / f"fold_{fold}.csv"
        expanded.source_predictions(predictions_for(validation), validation).to_csv(
            path, index=False
        )
        return {"run_id": f"run_{fold}", "prediction_path": str(path)}

    monkeypatch.setattr(engine, "run_task3_baseline_fold", completed_fold)
    result = expanded.run_expanded_usage(
        root=tmp_path,
        output_root=tmp_path / "output",
        registry_path=(
            tmp_path / "output/results/runs.csv" if shared_registry else tmp_path / "runs.csv"
        ),
        e8_directory=tmp_path,
        source_registry_path=tmp_path / "sources.csv",
        resume=False,
    )
    comparison = result["comparison"]
    assert comparison["sources"]["combined"]["rows"] == 15
    assert comparison["sources"]["teacher"]["rows"] == 10
    assert comparison["teacher_macro_f1_change"] == 0
    assert len(comparison["teacher_per_class"]) == 9
    assert bool(migration_calls) == shared_registry
    if shared_registry:
        assert result["registry_path"] == str(private)


@pytest.mark.parametrize("keep_mirror", [True, False])
def test_recovery_preserves_complete_fold_after_shared_csv_overwrite(
    trained_usage_fold, tiny_data, tmp_path, monkeypatch, keep_mirror
):
    from fashion.train import task3_usage_registry as recovery
    from fashion.train.registry import RunRegistry

    result, config = trained_usage_fold
    monkeypatch.setattr(recovery, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(expanded, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(recovery, "validate_dataset", lambda **kwargs: (tiny_data, {}))
    shared = tmp_path / "runs.csv"
    mirror = tmp_path / "local/results/runs.csv"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(shared.read_bytes())
    bundle = {p.name: p.read_bytes() for p in Path(result["run_dir"]).iterdir() if p.is_file()}
    other = RunRegistry(shared)
    other._write_rows([])
    other.start({"run_id": "other-notebook", "experiment_id": "gender"})
    shared_before = shared.read_bytes()
    with pytest.raises(ValueError, match="found 0"):
        other.update(result["run_id"], {"last_completed_stage": "example_lost_update"})
    if not keep_mirror:
        mirror.unlink()
    namespace = {
        "REPO_DIR": tmp_path,
        "DRIVE_TASK_DIR": tmp_path / "output",
        "LOCAL_REGISTRY": mirror,
        "DRIVE_REGISTRY": shared,
    }
    receipt = recovery.repair_connected_usage_session(namespace, expected_completed_folds=(0,))
    assert namespace["DRIVE_REGISTRY"] == recovery.usage_registry_path(tmp_path / "output")
    assert shared.read_bytes() == shared_before
    assert receipt["verified_complete_runs"] == [
        {"run_id": result["run_id"], "fold": 0, "original_complete_row_found": keep_mirror}
    ]
    assert bundle == {
        p.name: p.read_bytes() for p in Path(result["run_dir"]).iterdir() if p.is_file()
    }
    assert (
        expanded.reusable_fold(
            fold=0,
            output_root=tmp_path / "output",
            registry_path=receipt["registry_path"],
            splits=tiny_data,
            spec=expanded.expanded_usage_spec(),
        )["run_id"]
        == result["run_id"]
    )
    private = RunRegistry(receipt["registry_path"])
    private.start({"run_id": "next-usage-fold", "experiment_id": expanded.EXPERIMENT})
    for epoch in range(1, 4):
        other._write_rows([])
        private.update("next-usage-fold", {"last_completed_stage": f"epoch_{epoch}_complete"})
    assert private._read_rows()[-1]["last_completed_stage"] == "epoch_3_complete"


def test_recovery_rejects_changed_complete_artifacts(
    trained_usage_fold, tiny_data, tmp_path, monkeypatch
):
    from fashion.train import task3_usage_registry as recovery

    result, config = trained_usage_fold
    monkeypatch.setattr(recovery, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(recovery, "validate_dataset", lambda **kwargs: (tiny_data, {}))
    path = Path(result["run_dir"]) / "history.csv"
    path.write_text(path.read_text() + "\n")
    shared_before = (tmp_path / "runs.csv").read_bytes()
    with pytest.raises(ValueError, match="artifact changed"):
        recovery.prepare_usage_registry(
            root=tmp_path, output_root=tmp_path / "output", source_paths=(tmp_path / "runs.csv",)
        )
    assert (tmp_path / "runs.csv").read_bytes() == shared_before
    assert not recovery.usage_registry_path(tmp_path / "output").exists()


def test_original_training_error_survives_registry_failure(
    trained_usage_fold, tmp_path, monkeypatch
):
    from fashion.train import task3_baseline as engine
    from fashion.train.registry import RunRegistry

    def fail_training(*args, **kwargs):
        raise RuntimeError("original training error")

    def fail_logging(*args, **kwargs):
        raise ValueError("registry row missing")

    monkeypatch.setattr(engine, "_pass", fail_training)
    monkeypatch.setattr(RunRegistry, "fail", fail_logging)
    with pytest.raises(RuntimeError, match="original training error") as caught:
        engine.run_task3_baseline_fold(
            "usage",
            0,
            root=tmp_path,
            output_root=tmp_path / "output",
            registry_path=tmp_path / "runs.csv",
            device_name="cpu",
            child_spec=expanded.expanded_usage_spec(),
            parent_run_directory=tmp_path / "parent",
        )
    assert any("registry row missing" in note for note in caught.value.__notes__)


def test_recovery_records_interrupted_fold_without_promoting_it(
    trained_usage_fold, tiny_data, tmp_path, monkeypatch
):
    from fashion.train import task3_usage_registry as recovery

    result, config = trained_usage_fold
    monkeypatch.setattr(recovery, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(recovery, "validate_dataset", lambda **kwargs: (tiny_data, {}))
    run_id = "t3_usage_expanded_e8_usage_smallcnn_f2_s2753_interrupted"
    path = Path(result["run_dir"]).parent / run_id
    path.mkdir()
    saved_config = json.loads((Path(result["run_dir"]) / "config.json").read_text())
    saved_config["parent_run_id"] = expanded.E8_RUN_IDS[2]
    expanded.write_json(saved_config, path / "config.json")
    (path / "history.csv").write_text("epoch,train_loss\n1,1.2\n")
    receipt = recovery.prepare_usage_registry(
        root=tmp_path,
        output_root=tmp_path / "output",
        source_paths=(tmp_path / "runs.csv",),
        interrupted_run_ids=(run_id,),
    )
    rows = pd.read_csv(receipt["registry_path"], keep_default_na=False)
    interrupted = rows.loc[rows.run_id.eq(run_id)].iloc[0]
    assert interrupted.status == "failed"
    assert int(interrupted.validation_fold) == 2
    assert interrupted.checkpoint_sha256 == ""
    assert interrupted.exception_type == "RegistryRowLost"
    assert receipt["incomplete_runs"] == [run_id]
