"""Verify the Colab MixUp refit without touching real training or protected images."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from test_task3_gender_mixup_cv import cv_fixture  # noqa: F401

from fashion.config import ROOT
from fashion.train import task3_gender_mixup_refit as refit
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3GeM3CNN


@pytest.fixture
def refit_fixture(request, monkeypatch):
    root = request.getfixturevalue("cv_fixture")
    monkeypatch.setattr(refit, "_cuda_device", lambda: torch.device("cpu"))
    return root


def test_refit_uses_exact_cv_recipe_and_all_development_rows(refit_fixture):
    config, payload, training = refit.prepare_refit(root=refit_fixture)
    cv_config, cv_payload, _ = refit.cv.prepare_cv(root=refit_fixture)
    assert config == cv_config and config.epochs == payload["cosine_t_max"] == 30
    for key in (
        "effective_model_family",
        "classifier_dropout",
        "training_augmentation",
        "input_view",
        "mixup_policy",
        "gender_label_variant",
        "training_precision_settings",
        "source_config_sha256",
    ):
        assert payload[key] == cv_payload[key]
    assert payload["experiment_id"] != cv_payload["experiment_id"]
    assert payload["validation_fold"] is None and payload["validation_used"] is False
    assert payload["mixup_contract"]["scope"] == "development_refit"
    assert payload["mixup_contract"]["training_rows"] == len(training) == 25
    assert set(training.cv_fold) == set(range(5))
    assert training.partition.eq("development").all() and 999 not in set(training.id)
    assert not (refit_fixture / "sealed.png").exists()


def test_real_refit_30_epochs_one_model_no_validation_and_safe_reuse(refit_fixture, monkeypatch):
    from fashion.train.sam import SAMStep

    root = refit_fixture
    original_pass, original_load = refit._pass, torch.load
    model_creations, train_passes = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("The refit must not load weights, use SAM, or run evaluation")

    def checked_pass(model, loader, criterion, device, **kwargs):
        assert kwargs["optimizer"] is not None and kwargs.get("sam") is None
        assert kwargs["mixup"].contract["scope"] == "development_refit"
        assert set(loader.dataset.frame.cv_fold) == set(range(5))
        assert len(loader.dataset) == 25 and loader.dataset.augmentation != "none"
        train_passes.append(1)
        return original_pass(model, loader, criterion, device, **kwargs)

    def make_model(*args, **kwargs):
        model_creations.append(1)
        return Task3GeM3CNN(*args, **kwargs)

    monkeypatch.setattr(refit, "_pass", checked_pass)
    monkeypatch.setattr(refit, "Task3GeM3CNN", make_model)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(SAMStep, "step", forbidden)
    monkeypatch.setattr(refit.cv, "_evaluate", forbidden)
    mirror = root / "mirror/runs.csv"
    result = refit.run_gender_mixup_refit(
        root=root,
        output_root=root / "output",
        registry_mirrors=(mirror,),
    )
    assert len(train_passes) == 30 and len(model_creations) == 1
    assert result["selected_epoch"] == 30 and result["final_model_changed"] is False
    assert result["selection_status"] == "candidate_refit_for_later_review"
    rows = pd.read_csv(root / "results/runs.csv", keep_default_na=False)
    assert len(rows) == 1 and rows.iloc[0].status == "complete"
    assert rows.iloc[0].training_product_count == 25
    assert rows.iloc[0].validation_product_count == 0 and rows.iloc[0].validation_fold == ""
    pd.testing.assert_frame_equal(rows, pd.read_csv(mirror, keep_default_na=False))
    directory = Path(result["manifest_path"]).parent
    paths = {name: directory / record["path"] for name, record in result["files"].items()}
    checkpoint = original_load(paths["final_epoch.pt"], weights_only=True)
    reloaded = Task3GeM3CNN(Task3BaselineConfig(target="gender"), classifier_dropout=0.3)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
    assert checkpoint["class_names"] == refit.CLASSES
    assert checkpoint["normalization"]["total_pixels"] == 25 * 60 * 80
    assert "sam_policy" not in checkpoint["config"]
    history = pd.read_csv(paths["history.csv"])
    expected_lr = 1e-5 + 0.5 * (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(30) / 30))
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    assert history.epoch.tolist() == list(range(1, 31))
    assert history.selected_checkpoint.tolist() == [False] * 29 + [True]
    assert history.training_rows.eq(25).all() and history.optimizer_steps.eq(1).all()
    assert not any("validation" in column for column in history.columns)
    receipt = json.loads(paths["mixup_training.json"].read_text())
    assert receipt["contract"] == checkpoint["config"]["mixup_contract"]
    assert len(receipt["epochs"]) == 30 and all(e["rows"] == 25 for e in receipt["epochs"])
    assert not list(directory.rglob("*predictions*"))
    assert not list(directory.rglob("sam_training.json"))
    assert "macro_f1" not in json.loads(rows.iloc[0].metrics_json)
    assert not result["holdout_evaluated"] and not result["teacher_test_evaluated"]
    second = refit.run_gender_mixup_refit(root=root, output_root=root / "output")
    assert second["reused"] and second["run_id"] == result["run_id"]
    assert len(train_passes) == 30 and len(model_creations) == 1
    changed_payload = dict(checkpoint["config"], classifier_dropout=0.4)
    with pytest.raises(ValueError, match="different code/data/settings"):
        refit._verify_completed(directory, changed_payload, root / "results/runs.csv")
    paths["final_epoch.pt"].write_bytes(b"damaged")
    with pytest.raises(ValueError, match="artifact changed"):
        refit.run_gender_mixup_refit(root=root, output_root=root / "output")


def test_failed_refit_is_registered_and_restarts_fresh(refit_fixture, monkeypatch):
    root = refit_fixture
    attempts = []

    def interrupted(*args, **kwargs):
        attempts.append(kwargs["run_id"])
        registry = pd.read_csv(root / "results/runs.csv")
        assert registry.iloc[-1].status == "running"
        raise KeyboardInterrupt("simulated disconnect")

    monkeypatch.setattr(refit, "_fit", interrupted)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            refit.run_gender_mixup_refit(root=root, output_root=root / "output")
    assert len(set(attempts)) == 2
    registry = pd.read_csv(root / "results/runs.csv")
    assert len(registry) == 2 and registry.status.eq("failed").all()
    assert not list((root / "output").rglob("model_manifest.json"))


def test_refit_requires_colab_gpu_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Colab L4"):
        refit.run_gender_mixup_refit(output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_refit_notebook_uses_current_branch_drive_zip_and_separate_runner():
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/gender_mixup_refit.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'BRANCH = "task3-mixup-five-fold-training"' in code
    assert '"data/task3-data.zip"' in code
    assert "run_gender_mixup_refit" in code and "run_gender_mixup_cv(" not in code
    assert "registry_mirrors=(LOCAL_REGISTRY,)" in code
    assert "source_path.startswith" in code and "extractall" not in code
    assert "validate_verified_colab_runtime" in code
    assert "range(1, 31)" in code and "[False] * 29" in code
    assert "train_test_split" not in code and "torch.load" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "gender_mixup_refit.ipynb", "exec")
            assert all(output.output_type != "error" for output in cell.outputs)
            if cell.outputs:
                assert isinstance(cell.execution_count, int) and cell.execution_count > 0
