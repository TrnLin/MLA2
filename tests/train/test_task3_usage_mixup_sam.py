"""Usage-only mixing, weighted SAM math, fresh model fits and two-fold orchestration."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
from test_task3_usage_expanded_v2 import tiny_v2  # noqa: F401

from fashion.data.hashing import compute_sha256
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as screen
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import TrainingMixUp, training_contract


def test_recipe_preserves_e8_and_rejects_other_folds():
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
        k: v for k, v in asdict(v2.expanded_usage_spec()).items() if k not in identity
    }
    assert spec.to_dict()["initialization"] == "fresh_random_weights_for_each_fold"
    assert spec.to_dict()["mixup_policy"]["alpha"] == 0.2
    assert spec.to_dict()["sam_policy"]["rho"] == 0.05
    assert screen.screen_config(spec, fold=4, device_name="cuda").epochs == 30
    for fold in (1, 2, 3, False, True):
        with pytest.raises(ValueError, match="only folds 0 and 4"):
            screen.screen_config(spec, fold=fold, device_name="cuda")
    with pytest.raises(ValueError, match="CUDA GPU"):
        screen.screen_config(spec, fold=0, device_name="cpu")
    with pytest.raises(ValueError, match="frozen"):
        replace(spec, class_weight_cap=3)


def test_usage_mixup_keeps_na_external_rows_and_rejects_validation(tiny_v2):  # noqa: F811
    training, validation = v2.training_scope(tiny_v2, 0)
    training = training.drop(columns="gender")  # Usage must not depend on Gender labels.
    mapping = {label: index for index, label in enumerate(v2.CLASSES)}
    mix = TrainingMixUp(training, validation_fold=0, label_to_index=mapping, target="usage")
    assert len(mix.allowed) == 16
    assert mapping["NA"] in mix.allowed.values()
    assert mix.contract["target"] == "usage"
    mix.begin_epoch(1)
    lam, order = mix.plan(training.id.tolist(), training.usage.map(mapping).tolist())
    assert 0 <= lam <= 1 and sorted(order) == list(range(len(training)))
    mix.end_epoch()
    assert mix.receipt()["epochs"][0]["rows"] == len(training)
    contaminated = pd.concat([training, validation.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="fold-training"):
        training_contract(contaminated, validation_fold=0, target="usage")
    with pytest.raises(ValueError, match="frozen Usage"):
        training_contract(training, validation_fold=0, target="usage", alpha=0.4)


def test_weighted_mixup_sam_matches_explicit_soft_target_gradient_and_loss(tiny_v2):  # noqa: F811
    torch = pytest.importorskip("torch")
    from fashion.train.sam import SAMStep, usage_policy
    from fashion.train.task3_baseline import _pass

    torch.manual_seed(2753)
    training, _ = v2.training_scope(tiny_v2, 0)
    training = training.iloc[:7]
    mapping = {label: index for index, label in enumerate(v2.CLASSES)}
    actual_mix = TrainingMixUp(training, validation_fold=0, label_to_index=mapping, target="usage")
    ref_mix = TrainingMixUp(training, validation_fold=0, label_to_index=mapping, target="usage")
    actual = torch.nn.Linear(3, 9).double()
    reference = copy.deepcopy(actual)
    actual_opt = torch.optim.AdamW(actual.parameters(), lr=0.003, weight_decay=0.02)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0.02)
    weights = torch.tensor([0.1, 1, 1, 4, 2, 5, 1, 1, 1], dtype=torch.float64)
    sampler = torch.randn(len(training), 3, dtype=torch.float64)
    batches = []
    for start, stop in ((0, 4), (4, 7)):
        frame = training.iloc[start:stop]
        batches.append(
            dict(
                image=sampler[start:stop],
                label=torch.tensor(frame.usage.map(mapping).tolist()),
                id=torch.tensor(frame.id.tolist()),
            )
        )
    sam = SAMStep(actual, actual_opt, policy=usage_policy())
    actual_mix.begin_epoch(1)
    ref_mix.begin_epoch(1)
    sam.begin_epoch(1)
    numerator, denominator = 0.0, 0.0
    for batch in batches:
        x, partner, lam = ref_mix.apply(batch["image"], batch["label"], batch["id"])
        soft = (
            lam * torch.nn.functional.one_hot(batch["label"], 9).double()
            + (1 - lam) * torch.nn.functional.one_hot(partner, 9).double()
        )
        weighted = soft * weights
        mass = weighted.sum()

        def soft_loss():
            return -(weighted * reference(x).log_softmax(dim=1)).sum() / mass

        ref_opt.zero_grad(set_to_none=True)
        first = soft_loss()
        numerator += first.item() * mass.item()
        denominator += mass.item()
        first.backward()
        originals = [p.detach().clone() for p in reference.parameters()]
        norm = torch.cat([p.grad.flatten() for p in reference.parameters()]).norm()
        with torch.no_grad():
            for p in reference.parameters():
                p.add_(0.05 * p.grad / (norm + 1e-12))
        ref_opt.zero_grad(set_to_none=True)
        soft_loss().backward()
        with torch.no_grad():
            for p, original in zip(reference.parameters(), originals, strict=True):
                p.copy_(original)
        ref_opt.step()
    loss, labels, _, _ = _pass(
        actual,
        batches,
        torch.nn.CrossEntropyLoss(weight=weights),
        torch.device("cpu"),
        optimizer=actual_opt,
        mixup=actual_mix,
        sam=sam,
    )
    actual_mix.end_epoch()
    ref_mix.end_epoch()
    stats = sam.end_epoch(actual_mix.epochs[-1])
    for actual_p, reference_p in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_p, reference_p, atol=1e-12, rtol=1e-12)
    assert loss == pytest.approx(numerator / denominator, abs=1e-12)
    assert stats["first_loss"] == pytest.approx(loss, abs=1e-12)
    assert stats["loss_denominator_sum"] == pytest.approx(denominator)
    assert stats["optimizer_steps"] == 2 and stats["forward_backward_passes"] == 4
    assert labels.size == 0


@pytest.fixture
def trained_screen(tiny_v2, tmp_path, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    from fashion.train import sam as sam_module
    from fashion.train import task3_baseline as engine

    torch.set_num_threads(2)
    config = replace(Task3BaselineConfig(target="usage"), epochs=1, batch_size=4, num_workers=0)
    monkeypatch.setattr(engine, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(v2, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(screen, "Task3BaselineConfig", lambda **kwargs: config)
    monkeypatch.setattr(screen, "screen_config", lambda *args, **kwargs: config)
    policy = dict(sam_module.usage_policy(), diagnostic_epochs=[1])
    monkeypatch.setattr(sam_module, "usage_policy", lambda: dict(policy))
    monkeypatch.setattr(v2, "validate_dataset", lambda **kwargs: (tiny_v2, {"test_fixture": True}))
    monkeypatch.setattr(
        v2, "SPLIT_SHA256", compute_sha256(tmp_path / v2.DATA_DIRECTORY / "splits.csv")
    )
    monkeypatch.setattr(
        v2, "MAP_SHA256", compute_sha256(tmp_path / v2.DATA_DIRECTORY / "label_maps.json")
    )
    parent_root = tmp_path / "parents"
    output_root = tmp_path / "output"
    models = []
    original_builder = engine._build_task3_model

    def builder(*args):
        model = original_builder(*args)
        models.append(model)
        return model

    monkeypatch.setattr(engine, "_build_task3_model", builder)

    def reject_checkpoint_loading(*args, **kwargs):
        raise AssertionError("Fresh screen must never load reference checkpoint weights")

    monkeypatch.setattr(torch, "load", reject_checkpoint_loading)
    results = []
    for fold in screen.FOLDS:
        parent = parent_root / screen.BASELINE_RUN_IDS[fold]
        parent.mkdir(parents=True)
        v2.write_json({"run_id": parent.name, "validation_fold": fold}, parent / "metrics.json")
        for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
            (parent / name).write_text("Not a checkpoint: reference weights must not be loaded")
        result = engine.run_task3_baseline_fold(
            "usage",
            fold,
            root=tmp_path,
            output_root=output_root,
            registry_path=screen.usage_registry_path(output_root),
            device_name="cpu",
            child_spec=screen.screen_spec(),
            parent_run_directory=parent,
        )
        results.append(result)
    assert len(models) == 2 and models[0] is not models[1]
    return dict(results=results, output_root=output_root, splits=tiny_v2, root=tmp_path)


def test_fresh_usage_fits_register_both_passes_and_resume_without_loading_weights(trained_screen):
    fixture = trained_screen
    for fold, result in zip(screen.FOLDS, fixture["results"], strict=True):
        checked = screen.completed_fold(
            fold=fold,
            output_root=fixture["output_root"],
            registry_path=screen.usage_registry_path(fixture["output_root"]),
            splits=fixture["splits"],
            spec=screen.screen_spec(),
        )
        assert checked["run_id"] == result["run_id"]
        config = json.loads((Path(result["run_dir"]) / "config.json").read_text())
        assert config["scratch"] is True
        assert config["mixup_contract"]["target"] == "usage"
        assert "gender_label_variant" not in config
        assert config["class_weights"] is not None
        assert sum(config["class_counts"]) == 16
        assert set(result["metrics"]["source_metrics"]["validation"]) == {
            "combined",
            "teacher",
            "previous_added",
            "new_added",
            "added",
        }
    path = Path(fixture["results"][0]["run_dir"]) / "sam_training.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        screen.completed_fold(
            fold=0,
            output_root=fixture["output_root"],
            registry_path=screen.usage_registry_path(fixture["output_root"]),
            splits=fixture["splits"],
            spec=screen.screen_spec(),
        )


def test_runner_reuses_only_0_and_4_and_compares_matching_teacher_rows(trained_screen, monkeypatch):
    torch = pytest.importorskip("torch")
    from fashion.train import task3_baseline as engine

    fixture = trained_screen
    references = {}
    for fold, run in zip(screen.FOLDS, fixture["results"], strict=True):
        references[fold] = dict(run, predictions=v2.read_predictions(run["prediction_path"]))
    monkeypatch.setattr(screen, "check_reference", lambda **kwargs: references)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_training(*args, **kwargs):
        raise AssertionError("Completed two-fold screen should be reused")

    monkeypatch.setattr(engine, "run_task3_baseline_fold", reject_training)
    result = screen.run_usage_mixup_sam(
        root=fixture["root"],
        output_root=fixture["output_root"],
        baseline_directory=fixture["root"] / "parents",
        baseline_registry_path=fixture["root"] / "control/results/runs.csv",
    )
    assert [run["metrics"]["validation_fold"] for run in result["fold_results"]] == [0, 4]
    assert result["comparison"]["teacher_macro_f1_change"] == pytest.approx(0)
    assert result["comparison"]["sources"]["teacher"]["rows"] == 4
    assert len(result["comparison"]["teacher_per_class"]) == 9
    for folds in ((0,), (0, 1, 2, 3, 4), (4, 0), (False, 4)):
        with pytest.raises(ValueError, match="exactly folds 0 and 4"):
            screen.run_usage_mixup_sam(
                root=fixture["root"],
                output_root=fixture["output_root"],
                baseline_directory="unused",
                baseline_registry_path="unused",
                folds=folds,
            )
    bad = copy.deepcopy(references)
    bad[0]["predictions"] = bad[0]["predictions"].iloc[1:]
    with pytest.raises(ValueError, match="exactly cover"):
        screen.build_comparison(
            fixture["results"], splits=fixture["splits"], references=bad, contract={}
        )


def test_delivered_notebook_code_and_result_cells(trained_screen):
    import nbformat

    notebook = nbformat.read(
        Path(__file__).resolve().parents[2] / "notebooks/04am_task3_usage_mixup_sam_screen.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert code.count("run_usage_mixup_sam(") == 1
    assert "folds=(0, 4)" in code and "train_test_split" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04am", "exec")
    fixture = trained_screen
    references = {
        fold: dict(run, predictions=v2.read_predictions(run["prediction_path"]))
        for fold, run in zip(screen.FOLDS, fixture["results"], strict=True)
    }
    summary = screen.build_comparison(
        fixture["results"], splits=fixture["splits"], references=references, contract={}
    )
    namespace = dict(
        result=dict(
            fold_results=fixture["results"],
            comparison=summary,
            comparison_path="fixture/teacher_comparison.json",
            registry_path="fixture/runs.csv",
        ),
        pd=pd,
        Path=Path,
        display=lambda *args: None,
        DisplayImage=lambda **kwargs: kwargs,
        save_learning_curves=screen.save_learning_curves,
        DRIVE_TASK_DIR=fixture["root"],
    )
    exec(notebook.cells[10].source, namespace)
    exec(notebook.cells[12].source, namespace)
    assert namespace["curve_path"].stat().st_size > 1000
