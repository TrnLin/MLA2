"""Two-stage scope, cache integrity, decision gates and disposable-worker checks."""

import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_two_stage as screen
from fashion.train.config import baseline_parameter_count
from fashion.train.registry import RunRegistry
from fashion.train.task3_usage_two_stage_runtime import FoldResourceError, run_fold_process


def rows(fold, offset=0):
    labels = [name for name in screen.CLASSES if name != "Home"] * 2
    ids = np.arange(len(labels)) + fold * 100 + offset
    return pd.DataFrame(
        {
            "id": ids,
            "cv_fold": fold,
            "usage": labels,
            "product_family_group": [f"family-{i}" for i in ids],
            "path": [f"images/{i}.jpg" for i in ids],
            "partition": "development",
        }
    )


def predictions(frame, run_id="run", confidence=1.0):
    result = frame[["id", "cv_fold", "product_family_group", "path"]].copy()
    indices = frame.usage.map(dict(zip(screen.CLASSES, range(9), strict=True))).to_numpy()
    probabilities = np.full((len(frame), 9), (1 - confidence) / 8)
    probabilities[np.arange(len(frame)), indices] = confidence
    result["run_id"] = run_id
    result["true_index"] = result["predicted_index"] = indices
    result["true_label"] = result["predicted_label"] = frame.usage.to_numpy()
    result["confidence"] = confidence
    result[screen.probability_columns(screen.CLASSES)] = probabilities
    return result


def run(fold, confidence):
    frame = predictions(rows(fold), f"run-{fold}", confidence)
    metrics = screen.oof_metrics(frame, screen.CLASSES)
    metrics.update(
        fold_wall_seconds=10,
        peak_host_memory_bytes=1000,
        peak_memory_bytes=1000,
        backbone_unchanged=True,
    )
    robust = pd.DataFrame(
        [
            {
                "validation_fold": fold,
                "run_id": f"run-{fold}",
                "corruption": name,
                "macro_f1": metrics["macro_f1"],
                "macro_f1_change": 0.0,
            }
            for name in screen.CORE_CORRUPTIONS
        ]
    )
    return {"run_id": f"run-{fold}", "predictions": frame, "metrics": metrics, "robustness": robust}


def test_fixed_recipe_and_zero_support_class():
    value = screen.recipe()
    assert value["stage_a_epochs"] == 30 and value["stage_b_epochs"] == 10
    assert value["baseline"]["channels"] == [32, 64, 128, 256]
    assert value["checkpoint_policy"] == "final_stage_b_epoch_10"
    counts, weights = screen.class_weights_for_training(rows(0))
    assert len(weights) == 9 and counts[3] == weights[3] == 0
    assert screen.CLASSES[4] == "NA"
    with pytest.raises(ValueError, match="only folds"):
        screen.training_scope(pd.DataFrame(), 1)


def test_float32_predictions_round_trip(tmp_path):
    frame = predictions(rows(0))
    probabilities = np.random.default_rng(6).dirichlet(np.ones(9), len(frame)).astype(np.float32)
    frame[screen.probability_columns(screen.CLASSES)] = probabilities
    frame["confidence"] = probabilities.max(axis=1)
    path = tmp_path / "predictions.csv"
    measured = screen.save_predictions(frame, path)
    loaded = screen.read_predictions(path)
    np.testing.assert_array_equal(loaded[screen.probability_columns(screen.CLASSES)], probabilities)
    assert measured == screen.oof_metrics(loaded, screen.CLASSES)


def test_probability_route_and_each_guard(monkeypatch):
    monkeypatch.setattr(
        screen, "training_scope", lambda splits, fold: (rows(fold, 1000), rows(fold))
    )
    parent = {fold: run(fold, 0.8) for fold in screen.FOLDS}
    child = {fold: run(fold, 0.99) for fold in screen.FOLDS}
    result = screen.evaluate_usage_two_stage(child, parent, None, repetitions=10)
    assert result["status"] == "pass"
    assert result["probability_relative_improvements"]["nll"] > 0.1
    assert result["clean_gap_route"].startswith("unavailable")
    for field, value in (
        ("backbone_unchanged", False),
        ("fold_wall_seconds", 5401),
        ("peak_memory_bytes", screen.MEMORY_LIMIT + 1),
    ):
        broken = copy.deepcopy(child)
        broken[0]["metrics"][field] = value
        assert (
            screen.evaluate_usage_two_stage(broken, parent, None, repetitions=10)["status"]
            == "fail"
        )
    broken = copy.deepcopy(child)
    broken[0]["robustness"].loc[0, ["macro_f1", "macro_f1_change"]] = [8 / 9 - 0.1, -0.1]
    assert screen.evaluate_usage_two_stage(broken, parent, None, repetitions=10)["status"] == "fail"
    broken = copy.deepcopy(child)
    broken[0]["predictions"].loc[0, "id"] = 9999
    with pytest.raises(ValueError, match="IDs"):
        screen.evaluate_usage_two_stage(broken, parent, None, repetitions=10)


@pytest.mark.parametrize(
    "program,seconds,budget,match",
    [
        ("import time; time.sleep(5)", 0.1, 10**9, "deadline"),
        ("raise RuntimeError('broken')", 5, 10**9, "failed"),
        ("import time; x=bytearray(20000000); time.sleep(5)", 5, 1, "RSS"),
    ],
)
def test_watchdog_stops_failed_or_over_budget_worker(tmp_path, program, seconds, budget, match):
    with pytest.raises(FoldResourceError, match=match):
        run_fold_process(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            log_path=tmp_path / "log",
            seconds=seconds,
            memory_bytes=budget,
        )


def test_watchdog_captures_success(tmp_path):
    result = run_fold_process(
        [sys.executable, "-c", "import time; print('done'); time.sleep(.25)"],
        cwd=tmp_path,
        log_path=tmp_path / "log",
        seconds=5,
        memory_bytes=10**9,
    )
    assert 0 < result["fold_wall_seconds"] < 5
    assert result["peak_host_memory_bytes"] > 0
    assert "done" in (tmp_path / "log").read_text()


def test_failed_fold_registered_before_worker_and_no_second_fold(tmp_path, monkeypatch):
    sources = {fold: {"run_id": f"e2-{fold}", "sha256": {}} for fold in screen.FOLDS}
    monkeypatch.setattr(screen, "check_usage_two_stage_sources", lambda **kwargs: (sources, None))
    monkeypatch.setattr(
        screen, "training_scope", lambda splits, fold: (rows(fold, 1000), rows(fold))
    )
    registry = tmp_path / "runs.csv"
    calls = []

    def fail(*args, **kwargs):
        rows = pd.read_csv(registry)
        assert len(rows) == 1 and rows.iloc[0].status == "running"
        calls.append(rows.iloc[0].validation_fold)
        raise FoldResourceError("test deadline")

    monkeypatch.setattr(screen, "run_fold_process", fail)
    with pytest.raises(FoldResourceError, match="deadline"):
        screen.run_usage_two_stage_screen(
            e2_directory=tmp_path,
            source_registry_path=registry,
            output_root=tmp_path,
            registry_path=registry,
            root=ROOT,
        )
    assert calls == [0]
    assert pd.read_csv(registry).iloc[0].status == "failed"


def test_cached_bundle_recomputes_stage_a_and_checks_training_scope(tmp_path, monkeypatch):
    train, validation = rows(4), rows(0)
    monkeypatch.setattr(screen, "training_scope", lambda splits, fold: (train, validation))
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "corruptions").mkdir()
    config = {"fold": 0, "recipe": screen.recipe()}
    screen.write_json(config, directory / "config.json")
    for name in ("normalization.json", "stage_a.pt", "final_epoch.pt"):
        (directory / name).write_text("test fixture")
    cm = screen.save_predictions(predictions(validation), directory / "oof_predictions.csv")
    tm = screen.save_predictions(predictions(train), directory / "clean_train_predictions.csv")
    screen.save_predictions(predictions(train), directory / "stage_a_train_predictions.csv")
    screen.save_predictions(predictions(validation), directory / "stage_a_oof_predictions.csv")
    stage_a = {"training": tm, "validation": cm}
    screen.write_json(stage_a, directory / "stage_a_metrics.json")
    metrics = {
        **cm,
        "clean_training": tm,
        "stage_a": stage_a,
        "backbone_unchanged": True,
        "backbone_sha256_after": "frozen",
        "selected_stage": "B",
        "selected_epoch": 10,
        "parameter_count": baseline_parameter_count("usage"),
        "stage_b_trainable_parameters": 2313,
    }
    metrics["class_counts"], metrics["class_weights"] = screen.class_weights_for_training(train)
    screen.write_json(metrics, directory / "metrics.json")
    cache = {
        "scope": "outer_training_only",
        "ids": train.id.tolist(),
        "product_family_groups": train.product_family_group.tolist(),
        "shape": [len(train), 256],
        "backbone_sha256_before": "frozen",
    }
    screen.write_json(cache, directory / "feature_cache_manifest.json")
    pd.DataFrame(
        [(stage, e) for stage, count in (("A", 30), ("B", 10)) for e in range(1, count + 1)],
        columns=["stage", "epoch"],
    ).to_csv(directory / "history.csv", index=False)
    robust = run(0, 1.0)["robustness"]
    robust["run_id"] = "run"
    robust.to_csv(directory / "robustness.csv", index=False)
    for name in screen.CORE_CORRUPTIONS:
        screen.save_predictions(predictions(validation), directory / "corruptions" / f"{name}.csv")

    def manifest():
        screen.write_json(
            {
                "run_id": "run",
                "config_hash": screen.configuration_hash(config),
                "files": {
                    str(p.relative_to(directory)): compute_sha256(p)
                    for p in directory.rglob("*")
                    if p.is_file() and p.name != "manifest.json"
                },
            },
            directory / "manifest.json",
        )

    registry = tmp_path / "runs.csv"
    RunRegistry(registry).start({"run_id": "run", "config_hash": screen.configuration_hash(config)})
    manifest()
    screen.load_two_stage_run(directory, config, registry, None, pending=True)
    cache["ids"][0] = 9999
    screen.write_json(cache, directory / "feature_cache_manifest.json")
    manifest()
    with pytest.raises(ValueError, match="cache scope"):
        screen.load_two_stage_run(directory, config, registry, None, pending=True)
    cache["ids"] = train.id.tolist()
    screen.write_json(cache, directory / "feature_cache_manifest.json")
    changed = predictions(train, confidence=0.8)
    screen.save_predictions(changed, directory / "stage_a_train_predictions.csv")
    manifest()
    with pytest.raises(ValueError, match="does not reproduce"):
        screen.load_two_stage_run(directory, config, registry, None, pending=True)


def test_frozen_backbone_and_cached_head_autograd():
    torch = pytest.importorskip("torch")
    from fashion.train.config import Task3BaselineConfig
    from fashion.train.model import Task3BaselineCNN
    from fashion.train.task3_usage_two_stage_fit import (
        backbone_digest,
        cache_training_features,
        freeze_and_reset_head,
        train_head_epoch,
    )

    model = Task3BaselineCNN(Task3BaselineConfig(target="usage"))
    old_head = model.classifier.weight.detach().clone()
    frozen = freeze_and_reset_head(model)
    assert not torch.equal(old_head, model.classifier.weight)
    frame = rows(0).iloc[:4]
    loader = [
        {
            "id": torch.tensor(frame.id.to_numpy()),
            "label": torch.tensor([0, 1, 2, 4]),
            "image": torch.randn(4, 3, 80, 60),
        }
    ]
    features, labels, ids = cache_training_features(model, loader, frame, torch.device("cpu"))
    assert ids == frame.id.tolist() and not features.is_inference()
    head_before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001)
    loss = train_head_epoch(
        model,
        features,
        labels,
        torch.nn.CrossEntropyLoss(weight=torch.ones(9)),
        optimizer,
        generator=torch.Generator().manual_seed(2753),
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert np.isfinite(loss) and backbone_digest(model) == frozen
    assert not model.features.training and not torch.equal(head_before, model.classifier.weight)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2313
    with pytest.raises(ValueError, match="training IDs"):
        cache_training_features(model, loader, frame.iloc[::-1], torch.device("cpu"))


def test_notebook_code_compiles_without_training():
    path = ROOT / "notebooks/04z_task3_usage_two_stage_screen.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), str(path), "exec")
