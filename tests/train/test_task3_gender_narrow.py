"""Frozen width, score-loss, gap and inference-provenance contracts for G-N64."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.train.config import (
    NARROW_GEM3_FAMILY,
    Task3BaselineConfig,
    narrow_gem3_parameter_count,
)
from fashion.train.task3_clean_slate import _prediction_frame
from fashion.train.task3_dataset_v2 import dataset_v2_spec, run_task3_dataset_v2_screen
from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics
from fashion.train.task3_gender_ieee import POLICY, load_ieee_evaluation
from fashion.train.task3_gender_narrow import (
    NAME,
    RUNTIME,
    TRAINING_PRECISION,
    evaluate_gender_narrow_screen,
    narrow_config,
    read_precision_prerequisites,
)


def test_narrow_width_is_the_only_training_factor_change():
    ids = [f"g2-{f}" for f in range(5)]
    spec = dataset_v2_spec(NAME, ids)
    config = narrow_config(spec, fold=0, device_name="cuda")
    base = Task3BaselineConfig(target="gender")
    changes = {k for k in config.to_dict() if config.to_dict()[k] != base.to_dict()[k]}
    assert changes == {"channels", "model_family"}
    assert config.channels == (32, 64, 128, 64)
    assert config.weight_decay == 0.0001
    assert narrow_gem3_parameter_count() == 167653
    assert spec.training_augmentation == "translation_uniform_2px_p05"
    assert spec.to_dict()["screen_rule_version"] == "gn64_loss003_gap005_v1"
    assert "channels" not in dataset_v2_spec("gender_v2_translation", ids).to_dict()
    with pytest.raises(ValueError, match="predeclared factor"):
        replace(spec, training_augmentation="none")
    with pytest.raises(ValueError, match="channels"):
        replace(base, channels=(32, 64, 128, 64))
    with pytest.raises(ValueError, match="gender"):
        replace(config, target="usage")
    for fold, device in ((1, "cuda"), (0, "cpu")):
        with pytest.raises(ValueError, match="only folds"):
            narrow_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="run_gender_narrow_screen"):
        run_task3_dataset_v2_screen(NAME, parent_run_ids=ids, output_root="unused")


def _case(child_errors=300):
    classes = ["Boys", "Girls", "Men", "Unisex", "Women"]
    child, sources = {}, {"G2": {}, "E6": {}}
    for fold in (0, 4):
        frame = pd.DataFrame(
            {
                "id": fold * 10000 + np.arange(1000),
                "cv_fold": fold,
                "gender": [classes[i % 5] for i in range(1000)],
                "product_family_group": [f"{fold}-{i}" for i in range(1000)],
                "path": [f"{fold}-{i}.jpg" for i in range(1000)],
                "partition": "development",
            }
        )
        for name, runs in [("child", child), *sources.items()]:
            predicted = np.arange(1000) % 5
            count = child_errors if name == "child" else 300
            predicted[:count] = (predicted[:count] + 1) % 5
            probabilities = np.full((1000, 5), 0.025)
            probabilities[np.arange(1000), predicted] = 0.9
            p = _prediction_frame(
                frame,
                target="gender",
                classes=classes,
                probabilities=probabilities,
                run_id=f"{name}-{fold}",
            )
            metrics = oof_metrics(p, classes)
            training = 0.90 if name == "child" else 0.99
            metrics.update(
                final_train_eval_macro_f1=training,
                final_train_validation_macro_f1_gap=training - metrics["macro_f1"],
                parameter_count=167653 if name == "child" else 390181,
                peak_memory_bytes=400000000,
                train_seconds=999999,
                latency_ms_batch_1=9999,
                comparison_precision=POLICY,
            )
            delta = -0.1 if name == "E6" else -0.01
            robust = pd.DataFrame(
                [
                    dict(
                        run_id=f"{name}-{fold}",
                        validation_fold=fold,
                        corruption=c,
                        macro_f1=metrics["macro_f1"] + delta,
                        macro_f1_change=delta,
                    )
                    for c in CORE_CORRUPTIONS
                ]
            )
            runs[fold] = dict(
                run_id=f"{name}-{fold}", predictions=p, metrics=metrics, robustness=robust
            )
    return child, sources, classes


def _gates(result):
    return {c["gate"]: c["status"] for c in result["checks"]}


def test_same_validation_smaller_gap_passes_with_no_speed_cap():
    result = evaluate_gender_narrow_screen(*_case(), repetitions=100)
    assert result["status"] == "pass"
    assert result["required_validation_f1"] == pytest.approx(0.67)
    assert result["maximum_mean_clean_gap"] == pytest.approx(0.24)
    assert result["speed_cap"] is None


def test_three_point_loss_boundary_keeps_stricter_class_guard(monkeypatch):
    monkeypatch.setattr(
        "fashion.train.task3_gender_narrow.paired_family_bootstrap",
        lambda *args, **kwargs: {"lower_95": -0.03, "upper_95": 0.0},
    )
    result = evaluate_gender_narrow_screen(*_case(330))
    gates = _gates(result)
    assert gates["validation_delta"] == gates["validation_ci_lower"] == "pass"
    assert gates["fold_0.validation_delta"] == "pass"
    assert gates["class_f1"] == "fail"
    assert result["status"] == "fail"
    assert _gates(evaluate_gender_narrow_screen(*_case(331)))["validation_delta"] == "fail"


def test_mean_gap_cannot_hide_one_unimproved_fold():
    child, sources, classes = _case()
    child[0]["metrics"]["final_train_eval_macro_f1"] = 0.99
    child[0]["metrics"]["final_train_validation_macro_f1_gap"] = (
        0.99 - child[0]["metrics"]["macro_f1"]
    )
    child[4]["metrics"]["final_train_eval_macro_f1"] = 0.85
    child[4]["metrics"]["final_train_validation_macro_f1_gap"] = (
        0.85 - child[4]["metrics"]["macro_f1"]
    )
    result = evaluate_gender_narrow_screen(child, sources, classes, repetitions=30)
    assert _gates(result)["mean_gap_reduction"] == "pass"
    assert _gates(result)["fold_0.gap_reduction"] == "fail"


def test_confidence_interval_and_memory_guards(monkeypatch):
    monkeypatch.setattr(
        "fashion.train.task3_gender_narrow.paired_family_bootstrap",
        lambda *args, **kwargs: {"lower_95": -0.031, "upper_95": 0.0},
    )
    child, sources, classes = _case()
    child[0]["metrics"]["peak_memory_bytes"] = 3_000_000_000
    gates = _gates(evaluate_gender_narrow_screen(child, sources, classes))
    assert gates["validation_delta"] == "pass"
    assert gates["validation_ci_lower"] == gates["fold_0.gpu_memory_bytes"] == "fail"


def test_wrong_precision_or_gap_is_rejected():
    child, sources, classes = _case()
    sources["E6"][0]["metrics"]["comparison_precision"] = "historical"
    with pytest.raises(ValueError, match="matched IEEE"):
        evaluate_gender_narrow_screen(child, sources, classes)
    sources["E6"][0]["metrics"]["comparison_precision"] = POLICY
    child[0]["metrics"]["final_train_validation_macro_f1_gap"] = 0
    with pytest.raises(ValueError, match="gap arithmetic"):
        evaluate_gender_narrow_screen(child, sources, classes, repetitions=20)


def test_ieee_cache_rejects_changed_identity_or_file(tmp_path):
    from fashion.data.hashing import compute_sha256

    files = {"metrics.json", "robustness.csv", "clean_train_predictions.csv", "oof_predictions.csv"}
    files.update(f"{c}_predictions.csv" for c in CORE_CORRUPTIONS)
    for name in files:
        (tmp_path / name).write_text("original")
    manifest = {
        "identity": {"policy": POLICY},
        "weights_and_buffers_unchanged": True,
        "precision_settings": {key: "ieee" for key in TRAINING_PRECISION},
        "settings_restored": True,
        "files": {name: compute_sha256(tmp_path / name) for name in files},
    }
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        load_ieee_evaluation(tmp_path, run={}, splits=None, classes=[], identity={})
    (tmp_path / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_ieee_evaluation(
            tmp_path, run={}, splits=None, classes=[], identity=manifest["identity"]
        )


def test_ieee_float32_csv_round_trip_and_metric_validation(tmp_path, monkeypatch):
    from fashion.data.hashing import compute_sha256
    from fashion.train import task3_gender_ieee as ieee
    from fashion.train.task3_decisions import probability_columns

    child, _, classes = _case()
    run = {**child[0], "fold": 0}
    predictions = run["predictions"].copy()
    rng = np.random.default_rng(6)
    probabilities = rng.dirichlet(np.full(5, 0.2), size=len(predictions)).astype(np.float32)
    predicted = probabilities.argmax(axis=1)
    predictions[probability_columns(classes)] = probabilities
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["predicted_index"] = predicted
    predictions["predicted_label"] = [classes[i] for i in predicted]
    original = oof_metrics(predictions, classes)
    predictions.to_csv(tmp_path / "old.csv", index=False)
    old = oof_metrics(pd.read_csv(tmp_path / "old.csv"), classes)
    assert abs(original["nll"] - old["nll"]) > 1e-10

    metrics = ieee.save_ieee_predictions(predictions, tmp_path / "oof_predictions.csv", classes)
    saved = pd.read_csv(tmp_path / "oof_predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(saved[probability_columns(classes)].to_numpy(), probabilities)
    for name in ("macro_f1", "nll", "ece_15"):
        assert metrics[name] == original[name]
    metrics["comparison_precision"] = POLICY
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    robust = run["robustness"].copy()
    robust["macro_f1"] = metrics["macro_f1"]
    robust["macro_f1_change"] = 0.0
    robust.to_csv(tmp_path / "robustness.csv", index=False)
    extra = ["clean_train_predictions.csv", *(f"{c}_predictions.csv" for c in CORE_CORRUPTIONS)]
    for name in extra:
        ieee.save_ieee_predictions(predictions, tmp_path / name, classes)
    files = ["metrics.json", "robustness.csv", "oof_predictions.csv", *extra]
    manifest = {
        "identity": {},
        "weights_and_buffers_unchanged": True,
        "precision_settings": {key: "ieee" for key in TRAINING_PRECISION},
        "settings_restored": True,
        "files": {name: compute_sha256(tmp_path / name) for name in files},
    }
    expected = predictions.assign(gender=predictions["true_label"], partition="development")
    monkeypatch.setattr(ieee, "get_cv_split", lambda splits, fold: (None, expected))
    monkeypatch.setattr(ieee, "get_samples", lambda frame, target: frame)
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(manifest))
    loaded = load_ieee_evaluation(tmp_path, run=run, splits=None, classes=classes, identity={})
    assert loaded["metrics"]["nll"] == metrics["nll"]
    metrics["nll"] += 0.001
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    manifest["files"]["metrics.json"] = compute_sha256(tmp_path / "metrics.json")
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved nll differs"):
        load_ieee_evaluation(tmp_path, run=run, splits=None, classes=classes, identity={})


def test_precision_failure_blocks_before_torch_import(tmp_path):
    status = {
        "status": "error",
        "training_performed": False,
        "ieee_all_comparisons_pass": False,
        "earlier_summaries_reproduced": False,
        "settings_restored": True,
    }
    (tmp_path / "precision_status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="not passed"):
        read_precision_prerequisites(tmp_path)
    with pytest.raises(ValueError, match="required"):
        read_precision_prerequisites(None)


@pytest.fixture
def precision_evidence(tmp_path):
    from fashion.data.hashing import compute_sha256

    split = tmp_path / "data/processed/splits.csv"
    split.parent.mkdir(parents=True)
    split.write_text("fixed canonical split")
    directory = tmp_path / "precision"
    directory.mkdir()
    status = {
        "status": "complete_for_review",
        "training_performed": False,
        "ieee_all_comparisons_pass": True,
        "earlier_summaries_reproduced": True,
        "settings_restored": True,
        "runtime": RUNTIME,
        "runtime_default_settings": TRAINING_PRECISION,
        "ieee_settings": {k: "ieee" for k in TRAINING_PRECISION},
        "split_sha256": compute_sha256(split),
        "runs": [],
    }
    rows = []
    for model in ("G2", "CompactBlurCNN"):
        for fold in (0, 4):
            run_id = f"{model}-{fold}"
            status["runs"].append(
                {
                    "model": model,
                    "fold": fold,
                    "run_id": run_id,
                    "weights_and_buffers_unchanged": True,
                }
            )
            (directory / run_id).mkdir()
            for partition in ("train", "validation"):
                probabilities = np.full((160, 5), 0.025, dtype=np.float32)
                probabilities[:, 0] = 0.9
                arrays = {"ids": np.arange(160)}
                for mode in ("ieee", "runtime_default"):
                    arrays[f"{mode}_reference"] = probabilities
                    for batch in (1, 32, 128):
                        for order in ("forward", "reverse", "shuffle"):
                            arrays[f"{mode}_b{batch}_{order}"] = probabilities
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "partition": partition,
                                    "mode": mode,
                                    "batch_size": batch,
                                    "order": order,
                                    "max_abs_difference": 0.0,
                                    "prediction_flips": 0,
                                    "pass": True,
                                    "earlier_summary_reproduced": True,
                                    "earlier_max_abs_difference": 0.0,
                                }
                            )
                np.savez_compressed(directory / run_id / f"{partition}_probabilities.npz", **arrays)
    (directory / "precision_status.json").write_text(json.dumps(status))
    pd.DataFrame(rows).to_csv(directory / "precision_comparisons.csv", index=False)
    return tmp_path, directory


@pytest.mark.parametrize("change", ["arrays", "summary"])
def test_precision_evidence_is_checked_against_real_arrays(precision_evidence, change):
    root, directory = precision_evidence
    assert len(read_precision_prerequisites(directory, root=root)["artifact_sha256"]) == 10
    if change == "arrays":
        path = directory / "G2-0/train_probabilities.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in source.files}
        arrays["ieee_b1_forward"][0, 0] -= 0.0001
        arrays["ieee_b1_forward"][0, 1] += 0.0001
        np.savez_compressed(path, **arrays)
        message = "probability comparison fails"
    else:
        path = directory / "precision_comparisons.csv"
        data = pd.read_csv(path)
        data.loc[0, "max_abs_difference"] = 0.1
        data.to_csv(path, index=False)
        message = "summary differs"
    with pytest.raises(ValueError, match=message):
        read_precision_prerequisites(directory, root=root)


@pytest.mark.parametrize("failure", [None, "memory", "reference"])
@pytest.mark.parametrize("experiment_name", [NAME, "gender_dropout_030"])
def test_runner_evaluates_references_before_two_fits_and_stops_on_failure(
    tmp_path, monkeypatch, failure, experiment_name
):
    import sys
    from types import ModuleType

    import fashion.train.task3_gender_narrow as module

    child, sources, classes = _case()
    spec = dataset_v2_spec(experiment_name, [f"G2-{f}" for f in range(5)])
    expected = module._screen_config(spec, fold=0, device_name="cuda").to_dict()
    for fold, run in child.items():
        run["config"] = {
            **expected,
            "child_experiment": spec.to_dict(),
            "parent_run_id": f"G2-{fold}",
            "training_precision_settings": TRAINING_PRECISION,
            "precision_evidence_sha256": {},
        }
        run["sha256"] = {}
    for group in sources.values():
        for run in group.values():
            run.update(sha256={}, directory=str(tmp_path / run["run_id"]))
    registry = tmp_path / "runs.csv"
    registry.write_text("run_id\n")
    monkeypatch.setattr(
        module,
        "check_gender_narrow_sources",
        lambda **kwargs: (sources, classes, spec, {"artifact_sha256": {}}),
    )
    monkeypatch.setattr(module, "require_narrow_prerequisites", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_splits", lambda *args: None)
    monkeypatch.setattr(
        "fashion.train.task3_dataset_v2._reusable_fold", lambda *args, **kwargs: None
    )
    events = []

    def evaluate(run, **kwargs):
        events.append(("evaluate", run["run_id"]))
        if failure == "reference":
            raise ValueError("reference failed")
        return run

    monkeypatch.setattr(module, "evaluate_gender_ieee", evaluate)
    fake = ModuleType("fashion.train.task3_baseline")
    fake._json_dump = lambda payload, path: path.write_text(json.dumps(payload))

    def fit(target, fold, **kwargs):
        events.append(("fit", fold))
        assert kwargs["registry_path"] == registry
        assert kwargs["child_spec"] == spec
        if failure == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return {"run_dir": str(tmp_path / f"child-{fold}")}

    fake.run_task3_baseline_fold = fit
    monkeypatch.setitem(sys.modules, "fashion.train.task3_baseline", fake)
    monkeypatch.setattr(
        module, "inspect_gender_run", lambda path, **kwargs: child[int(path.name[-1])]
    )
    monkeypatch.setattr(
        module,
        "evaluate_gender_narrow_screen",
        lambda *args, **kwargs: {"status": "pass", "folds": []},
    )
    args = dict(
        g2_directory="unused",
        e6_directory="unused",
        source_registry_path=registry,
        precision_directory="unused",
        output_root=tmp_path,
        registry_path=registry,
        root=tmp_path,
        experiment_name=experiment_name,
    )
    if failure == "reference":
        with pytest.raises(ValueError, match="reference failed"):
            module.run_gender_narrow_screen(**args)
        assert not any(e[0] == "fit" for e in events)
    else:
        result = module.run_gender_narrow_screen(**args)
        assert events[:4] == [
            ("evaluate", f"{model}-{fold}") for model in ("G2", "E6") for fold in (0, 4)
        ]
        assert [e[1] for e in events if e[0] == "fit"] == ([0] if failure == "memory" else [0, 4])
        assert result["status"] == ("fail" if failure == "memory" else "pass")


def test_actual_narrow_model_has_only_the_final_block_width_change():
    torch = pytest.importorskip("torch")
    from fashion.train.model import Task3GeM3CNN
    from fashion.train.task3_baseline import _build_task3_model

    spec = dataset_v2_spec(NAME, [f"g2-{f}" for f in range(5)])
    config = narrow_config(spec, fold=0, device_name="cuda")
    model = _build_task3_model(config, spec)
    assert isinstance(model, Task3GeM3CNN)
    assert config.model_family == NARROW_GEM3_FAMILY
    assert sum(p.numel() for p in model.parameters()) == 167653
    assert [m.out_channels for m in model.features if isinstance(m, torch.nn.Conv2d)] == [
        32,
        64,
        128,
        64,
    ]
    assert model.pool.power == 3.0
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 80, 60)).shape == (2, 5)


def test_notebook_compiles_and_runs_only_the_frozen_screen():
    root = Path(__file__).resolve().parents[2]
    n = json.loads((root / "notebooks/04x_task3_gender_narrow64_screen.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code")
    assert code.count("run_gender_narrow_screen(") == 1
    assert "train_test_split" not in code and "confirmation" not in code
    for cell in n["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "04x", "exec")


def test_dropout_recipe_changes_only_dropout_and_keeps_full_width():
    from fashion.train.task3_gender_dropout import NAME as DROPOUT_NAME
    from fashion.train.task3_gender_dropout import dropout_config

    ids = [f"G2-{f}" for f in range(5)]
    spec = dataset_v2_spec(DROPOUT_NAME, ids)
    base = dataset_v2_spec("gender_v2_translation", ids)
    excluded = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "classifier_dropout",
    }
    for key, value in base.__dict__.items():
        if key not in excluded:
            assert spec.__dict__[key] == value
    assert spec.classifier_dropout == 0.30
    assert dropout_config(spec, fold=0, device_name="cuda") == Task3BaselineConfig(target="gender")
    with pytest.raises(ValueError, match="predeclared factor"):
        replace(spec, classifier_dropout=0.2)
    for fold, device in [(1, "cuda"), (0, "cpu")]:
        with pytest.raises(ValueError, match="only folds"):
            dropout_config(spec, fold=fold, device_name=device)
    with pytest.raises(ValueError, match="run_gender_dropout_screen"):
        run_task3_dataset_v2_screen(DROPOUT_NAME, parent_run_ids=ids, output_root="unused")


def test_dropout_gates_keep_score_gap_and_corruption_limits():
    from fashion.train.task3_gender_dropout import evaluate_gender_dropout_screen

    child, sources, classes = _case()
    for run in child.values():
        run["metrics"]["parameter_count"] = 390181
    result = evaluate_gender_dropout_screen(child, sources, classes, repetitions=20)
    assert result["status"] == "pass"
    assert result["rule_version"] == "gdrop030_loss003_gap005_v1"
    child[0]["metrics"]["parameter_count"] = 167653
    assert (
        _gates(evaluate_gender_dropout_screen(child, sources, classes, repetitions=20))[
            "fold_0.parameters"
        ]
        == "fail"
    )
    child[0]["metrics"]["parameter_count"] = 390181
    for run in child.values():
        run["metrics"]["final_train_eval_macro_f1"] += 0.08
        run["metrics"]["final_train_validation_macro_f1_gap"] += 0.08
    gates = _gates(evaluate_gender_dropout_screen(child, sources, classes, repetitions=20))
    assert gates["mean_gap_reduction"] == "fail"
    assert gates["validation_delta"] == "pass"


def test_gem_builder_passes_dropout_to_model_without_torch():
    import ast

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "src/fashion/train/task3_baseline.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_task3_model"
    )
    spec = dataset_v2_spec("gender_dropout_030", [f"G2-{f}" for f in range(5)])
    calls = []
    scope = {
        "_task3_model_contract": lambda *args: ("task3_small_cnn_gem_p3", "", 390181, None),
        "NARROW_GEM3_FAMILY": NARROW_GEM3_FAMILY,
        "Task3GeM3CNN": lambda config, **kwargs: calls.append(kwargs),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "builder", "exec"), scope)
    scope["_build_task3_model"](Task3BaselineConfig(target="gender"), spec)
    scope["_build_task3_model"](Task3BaselineConfig(target="gender"), None)
    assert calls == [{"classifier_dropout": 0.30}, {"classifier_dropout": 0.0}]


def test_actual_dropout_is_between_gem_and_classifier_and_off_in_eval():
    torch = pytest.importorskip("torch")
    from fashion.train.task3_baseline import _build_task3_model

    config = Task3BaselineConfig(target="gender")
    spec = dataset_v2_spec("gender_dropout_030", [f"G2-{f}" for f in range(5)])
    model = _build_task3_model(config, spec)
    assert sum(p.numel() for p in model.parameters()) == 390181
    assert model.classifier_dropout.p == 0.30
    assert model.pool.power == 3.0
    features = torch.ones(32, 256)
    model.train()
    assert (model.classifier_dropout(features) == 0).any()
    model.eval()
    assert torch.equal(model.classifier_dropout(features), features)
    order = []
    handles = [
        getattr(model, name).register_forward_hook(
            lambda m, args, out, name=name: order.append(name)
        )
        for name in ("pool", "classifier_dropout", "classifier")
    ]
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 80, 60)).shape == (2, 5)
    for handle in handles:
        handle.remove()
    assert order == ["pool", "classifier_dropout", "classifier"]


def test_dropout_notebook_has_only_the_frozen_screen():
    import nbformat

    root = Path(__file__).resolve().parents[2]
    n = nbformat.read(root / "notebooks/04y_task3_gender_dropout_screen.ipynb", as_version=4)
    nbformat.validate(n)
    code = "\n".join(c.source for c in n.cells if c.cell_type == "code")
    assert code.count("run_gender_dropout_screen(") == 1
    assert "run_gender_narrow_screen" not in code
    for c in n.cells:
        if c.cell_type == "code":
            assert not c.outputs
            compile(c.source, "04y", "exec")
