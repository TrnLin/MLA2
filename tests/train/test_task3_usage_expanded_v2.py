"""Versioned data routing, scratch fitting, resume and matched-source comparisons."""

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
from fashion.train import task3_usage_expanded as previous
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_decisions import probability_columns


@pytest.fixture
def tiny_v2(tmp_path):
    rows = []
    rng = np.random.default_rng(2753)
    for fold in range(5):
        for cohort, label in (
            ("teacher", "NA"),
            ("teacher", "Casual"),
            ("previous_added", "Home"),
            ("new_added", "Party"),
        ):
            image_id = len(rows) + 1
            pixels = rng.integers(20, 180, (80, 60, 3), dtype=np.uint8)
            # Bright validation images and white external borders expose scope mistakes.
            if fold == 0:
                pixels = np.minimum(pixels.astype(int) + 65, 254).astype(np.uint8)
            if cohort != "teacher":
                pixels[:10] = 255
                pixels[70:] = 255
            path = tmp_path / f"image_{image_id}.png"
            Image.fromarray(pixels).save(path)
            row = {
                "id": image_id,
                "path": path.name,
                "sha256": compute_sha256(path),
                "product_family_group": f"family_{image_id}",
                "duplicate_group": f"image_{image_id}",
                "product_name_key": f"item {image_id}",
                "extension_family_group": "" if cohort == "teacher" else f"external_{image_id}",
                "partition": "development",
                "cv_fold": fold,
                "source_dataset": "teacher" if cohort == "teacher" else "retailer",
                v2.COHORT_COLUMN: cohort,
                "content_left": np.nan if cohort == "teacher" else 0,
                "content_top": np.nan if cohort == "teacher" else 10,
                "content_width": np.nan if cohort == "teacher" else 60,
                "content_height": np.nan if cohort == "teacher" else 60,
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
    protected.update({f"has_{target}_label": False for target in TARGET_COLUMNS})
    protected.update({target: "" for target in TARGET_COLUMNS})
    rows.append(protected)
    directory = tmp_path / v2.DATA_DIRECTORY
    directory.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(directory / "splits.csv", index=False)
    maps = {
        "usage": {
            "source_scope": "development",
            "classes": list(v2.CLASSES),
            "label_to_index": {label: index for index, label in enumerate(v2.CLASSES)},
        }
    }
    v2.write_json(maps, directory / "label_maps.json")
    v2.write_json(maps, tmp_path / "data/processed/label_maps.json")
    splits = load_splits(directory / "splits.csv")
    for column in v2.GEOMETRY_COLUMNS:
        splits[column] = pd.to_numeric(splits[column].replace("", np.nan))
    return splits


def predictions_for(frame, run_id="fixture"):
    result = frame[["id", "cv_fold", "product_family_group", "path"]].copy()
    labels = frame.usage.map({label: index for index, label in enumerate(v2.CLASSES)})
    result["run_id"] = run_id
    result["true_index"] = labels
    result["true_label"] = frame.usage
    result["predicted_index"] = labels
    result["predicted_label"] = frame.usage
    result["confidence"] = 0.92
    probabilities = np.full((len(frame), 9), 0.01)
    probabilities[np.arange(len(frame)), labels.to_numpy()] = 0.92
    result[probability_columns(v2.CLASSES)] = probabilities
    return result


def references_for(splits):
    references = {"teacher_e8": {}, "previous_expansion": {}}
    for fold in range(5):
        _, validation = v2.training_scope(splits, fold)
        teacher = validation.loc[validation.source_dataset.eq("teacher")]
        older = validation.loc[validation[v2.COHORT_COLUMN].ne("new_added")]
        references["teacher_e8"][fold] = {
            "run_id": v2.E8_RUN_IDS[fold],
            "predictions": predictions_for(teacher),
        }
        references["previous_expansion"][fold] = {
            "run_id": v2.PREVIOUS_RUN_IDS[fold],
            "predictions": v2.source_predictions(predictions_for(older), older),
        }
    return references


def test_recipe_is_frozen_and_only_dataset_identity_changes():
    spec = v2.expanded_usage_spec()
    identity = {"name", "experiment_id", "hypothesis_id", "artifact_dir", "run_prefix"}
    assert {k: v for k, v in asdict(spec).items() if k not in identity} == {
        k: v for k, v in asdict(previous.expanded_usage_spec()).items() if k not in identity
    }
    assert spec.to_dict()["dataset"]["split_sha256"] == v2.SPLIT_SHA256
    assert spec.to_dict()["dataset"]["previous_split_sha256"] == previous.SPLIT_SHA256
    assert spec.checkpoint_policy == "final_epoch"
    with pytest.raises(FrozenInstanceError):
        spec.classifier_dropout = 0.2
    with pytest.raises(ValueError, match="frozen E8"):
        replace(spec, class_weight_beta=0.9)


def test_saved_folds_keep_all_three_sources_and_never_open_holdout(tiny_v2):
    for fold in range(5):
        training, validation = v2.training_scope(tiny_v2, fold)
        assert (len(training), len(validation)) == (16, 4)
        assert validation[v2.COHORT_COLUMN].value_counts().to_dict() == {
            "teacher": 2,
            "previous_added": 1,
            "new_added": 1,
        }
        assert 999 not in set(training.id) | set(validation.id)
        assert not set(training.id) & set(validation.id)
    crossing = tiny_v2.copy()
    crossing.loc[crossing.id.isin([4, 8]), "extension_family_group"] = "crossing"
    with pytest.raises(ValueError, match="extension_family_group"):
        v2.training_scope(crossing, 0)


def test_comparison_is_on_same_teacher_ids_and_splits_source_diagnostics(tiny_v2):
    expected = tiny_v2.loc[tiny_v2.partition.eq("development")]
    predictions = v2.source_predictions(predictions_for(expected), expected)
    refs = references_for(tiny_v2)
    comparison = v2.build_comparison(
        predictions, splits=tiny_v2, folds=tuple(range(5)), references=refs, contract={}
    )
    assert {k: v["rows"] for k, v in comparison["sources"].items()} == {
        "combined": 20,
        "teacher": 10,
        "previous_added": 5,
        "new_added": 5,
        "added": 10,
    }
    assert comparison["teacher_macro_f1_changes"] == {
        "teacher_e8": 0,
        "previous_expansion": 0,
    }
    assert len(comparison["teacher_per_class"]) == 9
    assert comparison["teacher_per_class"][4]["class_name"] == "NA"
    with pytest.raises(ValueError, match="exactly cover"):
        v2.build_comparison(
            predictions.iloc[:-1],
            splits=tiny_v2,
            folds=tuple(range(5)),
            references=refs,
            contract={},
        )
    refs["previous_expansion"][0]["predictions"].loc[0, "id"] = 10000
    with pytest.raises(ValueError, match="exactly the same"):
        v2.build_comparison(
            predictions, splits=tiny_v2, folds=tuple(range(5)), references=refs, contract={}
        )


@pytest.fixture
def trained_v2(tiny_v2, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="usage"), epochs=1, batch_size=4, num_workers=0)
    monkeypatch.setattr(engine, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(v2, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(v2, "validate_dataset", lambda **kwargs: (tiny_v2, {"test_fixture": True}))
    monkeypatch.setattr(
        v2, "SPLIT_SHA256", compute_sha256(tmp_path / v2.DATA_DIRECTORY / "splits.csv")
    )
    monkeypatch.setattr(
        v2, "MAP_SHA256", compute_sha256(tmp_path / v2.DATA_DIRECTORY / "label_maps.json")
    )
    parent = tmp_path / "parent"
    parent.mkdir()
    v2.write_json({"run_id": v2.E8_RUN_IDS[0], "validation_fold": 0}, parent / "metrics.json")
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (parent / name).write_text("Reference weights must never be loaded")
    output = tmp_path / "output"
    result = engine.run_task3_baseline_fold(
        "usage",
        0,
        root=tmp_path,
        output_root=output,
        registry_path=v2.usage_registry_path(output),
        registry_mirrors=(tmp_path / "local/results/runs.csv",),
        device_name="cpu",
        child_spec=v2.expanded_usage_spec(),
        parent_run_directory=parent,
    )
    return result


def test_real_optimizer_step_uses_v2_images_weights_geometry_and_registry(
    trained_v2, tiny_v2, tmp_path
):
    import torch

    result = trained_v2
    directory = Path(result["run_dir"])
    log = v2.usage_registry_path(tmp_path / "output")
    registry = pd.read_csv(log, keep_default_na=False)
    assert registry.status.tolist() == ["complete"]
    assert registry.experiment_id.item() == v2.EXPERIMENT
    assert registry.split_digest.item() == v2.SPLIT_SHA256
    assert (registry.training_product_count.item(), registry.validation_product_count.item()) == (
        16,
        4,
    )
    assert log.read_bytes() == (tmp_path / "local/results/runs.csv").read_bytes()
    assert result["metrics"]["class_counts"][3] == 4  # Previous Home images.
    assert result["metrics"]["class_counts"][5] == 4  # Newly added Party images.
    assert result["metrics"]["source_metrics"]["validation"]["new_added"]["rows"] == 1
    training, validation = v2.training_scope(tiny_v2, 0)
    pixels = []
    for row in training.itertuples():
        values = np.array(Image.open(tmp_path / row.path), dtype=np.float64)
        if row.source_dataset != "teacher":
            values = values[10:70]
        pixels.append(values.reshape(-1, 3) / 255)
    normalization = json.loads((directory / "normalization.json").read_text())
    assert normalization["mean"] == pytest.approx(np.concatenate(pixels).mean(axis=0))
    assert set(v2.read_predictions(directory / "training_predictions.csv").id) == set(training.id)
    assert set(v2.read_predictions(directory / "oof_predictions.csv").id) == set(validation.id)
    checkpoint = torch.load(directory / "final_epoch.pt", map_location="cpu", weights_only=True)
    assert checkpoint["config"]["child_experiment"]["dataset"]["name"] == v2.DATASET
    assert checkpoint["class_names"] == list(v2.CLASSES)
    assert any(
        v.item() > 0
        for k, v in checkpoint["model_state_dict"].items()
        if k.endswith("num_batches_tracked")
    )


def test_resume_reuses_verified_v2_only_and_rejects_changed_checkpoint(
    trained_v2, tiny_v2, tmp_path
):
    arguments = dict(
        fold=0,
        output_root=tmp_path / "output",
        registry_path=v2.usage_registry_path(tmp_path / "output"),
        splits=tiny_v2,
        spec=v2.expanded_usage_spec(),
    )
    assert v2.reusable_fold(**arguments)["run_id"] == trained_v2["run_id"]
    assert v2.reusable_fold(**{**arguments, "fold": 1}) is None
    path = Path(trained_v2["run_dir"]) / "final_epoch.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        v2.reusable_fold(**arguments)


def test_real_five_fold_orchestration_uses_v2_routes_and_source_comparison(
    tiny_v2, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(v2, "validate_dataset", lambda **kwargs: (tiny_v2, {}))
    monkeypatch.setattr(v2, "check_references", lambda **kwargs: references_for(tiny_v2))
    calls = []

    def completed_fold(target, fold, **kwargs):
        calls.append((target, fold, kwargs))
        _, validation = v2.training_scope(tiny_v2, fold)
        path = tmp_path / f"fold_{fold}.csv"
        v2.source_predictions(predictions_for(validation), validation).to_csv(path, index=False)
        return {"run_id": f"run_{fold}", "prediction_path": str(path)}

    monkeypatch.setattr(engine, "run_task3_baseline_fold", completed_fold)
    args = dict(
        root=tmp_path,
        output_root=tmp_path / "output",
        e8_directory=tmp_path,
        source_registry_path=tmp_path / "sources.csv",
        previous_directory=tmp_path,
        previous_registry_path=tmp_path / "previous.csv",
        resume=False,
    )
    result = v2.run_expanded_usage_v2(**args)
    assert [fold for _, fold, _ in calls] == list(range(5))
    assert all(kwargs["child_spec"].name == "usage_expanded_v2_e8" for _, _, kwargs in calls)
    assert result["comparison"]["sources"]["combined"]["rows"] == 20
    assert result["comparison"]["teacher_macro_f1_changes"]["previous_expansion"] == 0
    assert "aggregate/teacher_comparison.json" in result["comparison_path"]
    with pytest.raises(ValueError, match="own run log"):
        v2.run_expanded_usage_v2(**args, registry_path=tmp_path / "output/results/runs.csv")
    with pytest.raises(ValueError, match="separate local"):
        v2.run_expanded_usage_v2(**args, registry_mirrors=(tmp_path / "sources.csv",))
    from fashion.train.registry import RunRegistry

    mixed_log = tmp_path / "unrelated/results/runs.csv"
    RunRegistry(mixed_log).start({"run_id": "gender", "experiment_id": "other-experiment"})
    before = mixed_log.read_bytes()
    with pytest.raises(ValueError, match="cannot overwrite"):
        v2.run_expanded_usage_v2(**args, registry_mirrors=(mixed_log,))
    assert mixed_log.read_bytes() == before


def test_reviewed_dataset_loads_all_sources_and_keeps_teacher_geometry_empty():
    if not (ROOT / v2.DATA_DIRECTORY / "splits.csv").is_file():
        pytest.skip("Reviewed dataset is not present in a source-only checkout")
    splits, contract = v2.validate_dataset(check_images=False)
    assert contract["usage_development_rows"] == 33459
    assert sum(row["new_added_validation_rows"] for row in contract["folds"]) == 567
    assert sum(row["previous_added_validation_rows"] for row in contract["folds"]) == 120
    assert splits.loc[splits.source_dataset.eq("teacher"), "content_left"].isna().all()
    assert splits.loc[splits.source_dataset.ne("teacher"), "content_left"].notna().all()
    assert (contract["folds"][0]["training_rows"], contract["folds"][0]["validation_rows"]) == (
        26751,
        6708,
    )


def test_trainer_refuses_the_older_split(monkeypatch):
    if not (ROOT / previous.DATA_DIRECTORY / "splits.csv").is_file():
        pytest.skip("Reviewed dataset is not present in a source-only checkout")
    monkeypatch.setattr(v2, "DATA_DIRECTORY", previous.DATA_DIRECTORY)
    with pytest.raises(ValueError, match="Reviewed Usage v2 data changed"):
        v2.validate_dataset(check_images=False)


def test_notebook_result_cells_accept_actual_training_outputs(trained_v2, tiny_v2, tmp_path):
    import nbformat
    from IPython.display import Image as DisplayImage

    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/usage_expanded_v2_e8.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert all(output.output_type != "error" for output in cell.outputs)
    comparison = v2.build_comparison(
        v2.read_predictions(trained_v2["prediction_path"]),
        splits=tiny_v2,
        folds=(0,),
        references=references_for(tiny_v2),
        contract={},
    )
    outputs = []
    namespace = {
        "pd": pd,
        "Path": Path,
        "display": outputs.append,
        "DisplayImage": DisplayImage,
        "DRIVE_TASK_DIR": tmp_path / "output",
        "save_learning_curves": v2.save_learning_curves,
        "result": {
            "fold_results": [trained_v2],
            "comparison": comparison,
            "comparison_path": str(tmp_path / "output/comparison.json"),
            "registry_path": str(v2.usage_registry_path(tmp_path / "output")),
        },
    }
    for cell in notebook.cells:
        if cell.id in {"usage-v2-results", "usage-v2-curves"}:
            exec(compile(cell.source, cell.id, "exec"), namespace)
    assert len(outputs) == 5
    assert namespace["curve_path"].stat().st_size > 1000
