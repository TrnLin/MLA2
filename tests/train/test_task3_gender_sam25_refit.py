"""Refit covers all development folds, excludes protected images, and saves one model."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.train.mixup import TrainingMixUp, training_contract


def test_refit_mixup_requires_explicit_scope_and_keeps_cv_strict():
    frame = pd.DataFrame(
        {
            "id": range(5),
            "cv_fold": range(5),
            "partition": "development",
            "product_family_group": list("abcde"),
            "gender": "Men",
        }
    )
    with pytest.raises(ValueError, match="canonical fold"):
        training_contract(frame, validation_fold=None)
    with pytest.raises(ValueError, match="fold-training"):
        training_contract(frame, validation_fold=0)
    with pytest.raises(ValueError, match="no validation fold"):
        training_contract(frame, validation_fold=0, scope="development_refit")
    for bad in (frame.iloc[:4], frame.assign(partition="holdout"), frame.assign(cv_fold=-1)):
        with pytest.raises(ValueError, match="fold-training"):
            training_contract(bad, validation_fold=None, scope="development_refit")
    mix = TrainingMixUp(
        frame,
        validation_fold=None,
        scope="development_refit",
        label_to_index={"Men": 0},
    )
    mix.begin_epoch(1)
    mix.plan(list(range(5)), [0] * 5)
    with pytest.raises(ValueError, match="non-training"):
        mix.plan([999], [0])
    mix.end_epoch()
    assert mix.receipt()["epochs"][0]["rows"] == 5
    assert mix.contract["validation_fold"] is None


@pytest.fixture
def refit_fixture(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from test_task3_gender_sam import synthetic_gender

    from fashion.data.gender_name_truth import build_gender_name_truth_variant
    from fashion.data.hashing import compute_sha256
    from fashion.train import task3_gender_sam25_refit as refit

    # Reuse the real canonical-fixture builder, then add the names required by the label rule.
    synthetic_gender.__wrapped__(tmp_path)
    path = tmp_path / "data/processed/splits.csv"
    rows = pd.read_csv(path, keep_default_na=False)
    rows["productDisplayName"] = rows.gender + " item"
    rows.to_csv(path, index=False)
    summary = build_gender_name_truth_variant(tmp_path)
    source = json.loads((ROOT / refit.SOURCE_CONFIG).read_text())
    source["num_workers"] = 0
    source["gender_label_variant"].update(
        canonical_split_sha256=compute_sha256(path),
        labels_sha256=summary["files"]["labels.csv"],
    )
    target = tmp_path / refit.SOURCE_CONFIG
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(source))
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    monkeypatch.setattr(refit, "SOURCE_SHA256", compute_sha256(target))
    monkeypatch.setattr(refit, "EXPECTED_ROWS", 25)
    monkeypatch.setattr(refit, "_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(refit.subprocess, "check_output", lambda *a, **kw: "test-commit\n")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield refit, tmp_path
    torch.set_num_threads(previous_threads)


def test_refit_preflight_verifies_labels_images_and_all_development_rows(refit_fixture):
    refit, root = refit_fixture
    config, payload, training = refit.prepare_refit(root=root)
    assert len(training) == 25 and set(training.cv_fold) == set(range(5))
    assert training.partition.eq("development").all() and 999 not in set(training.id)
    assert config.epochs == 25 and payload["cosine_t_max"] == 30
    assert payload["classifier_dropout"] == 0.3
    assert payload["mixup_contract"]["policy"]["alpha"] == 0.2
    assert payload["sam_policy"]["rho"] == 0.05
    assert payload["validation_used"] is False
    assert not (root / "sealed.png").exists()
    (root / training.iloc[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input image"):
        refit.prepare_refit(root=root)


def test_complete_refit_real_25_epochs_checkpoint_and_reuse(refit_fixture):
    refit, root = refit_fixture
    import torch

    registry = root / "results/runs.csv"
    # The real model, images, normalization, augmentation, SAM, MixUp and 25 epochs run on CPU.
    result = refit.run_gender_sam25_refit(root=root, output_root=root / "output")
    rows = pd.read_csv(registry, keep_default_na=False)
    assert len(rows) == 1 and rows.iloc[0].status == "complete"
    assert rows.iloc[0].training_product_count == 25
    assert rows.iloc[0].validation_product_count == 0 and rows.iloc[0].validation_fold == ""
    directory = Path(result["manifest_path"]).parent
    checkpoint = torch.load(
        directory / result["files"]["final_epoch.pt"]["path"], weights_only=True
    )
    config = refit.Task3BaselineConfig(target="gender", epochs=25)
    reloaded = refit.Task3GeM3CNN(config, classifier_dropout=0.3)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 25
    assert checkpoint["class_names"] == refit.CLASSES
    assert checkpoint["normalization"]["total_pixels"] == 25 * 60 * 80
    history = pd.read_csv(directory / result["files"]["history.csv"]["path"])
    expected_lr = 1e-5 + 0.5 * (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(25) / 30))
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    assert history.selected_checkpoint.tolist() == [False] * 24 + [True]
    assert history.training_rows.eq(25).all()
    sam = json.loads((directory / result["files"]["sam_training.json"]["path"]).read_text())
    assert len(sam["epochs"]) == 25
    assert all(e["rows"] == 25 and e["forward_backward_passes"] == 2 for e in sam["epochs"])
    assert not any("validation" in c for c in history.columns)
    assert not list(directory.rglob("oof_predictions.csv"))
    assert "macro_f1" not in json.loads(rows.iloc[0].metrics_json)
    second = refit.run_gender_sam25_refit(root=root, output_root=root / "output")
    assert second["reused"] and second["run_id"] == result["run_id"]
    assert len(pd.read_csv(registry)) == 1
    (directory / result["files"]["final_epoch.pt"]["path"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="artifact changed"):
        refit.run_gender_sam25_refit(root=root, output_root=root / "output")


def test_failed_fit_stays_registered_without_completion_manifest(refit_fixture, monkeypatch):
    refit, root = refit_fixture

    def fail_fit(*args, **kwargs):
        rows = pd.read_csv(root / "results/runs.csv", keep_default_na=False)
        assert rows.iloc[0].status == "running"
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(refit, "_fit", fail_fit)
    with pytest.raises(RuntimeError, match="simulated training"):
        refit.run_gender_sam25_refit(root=root, output_root=root / "output")
    rows = pd.read_csv(root / "results/runs.csv", keep_default_na=False)
    assert len(rows) == 1 and rows.iloc[0].status == "failed"
    assert not list((root / "output").rglob("model_manifest.json"))


def test_refit_launcher_uses_existing_zip_and_separate_output():
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/gender_sam25_refit.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert '"data/task3-data.zip"' in code
    assert "run_gender_sam25_refit" in code
    assert "registry_mirrors=(LOCAL_REGISTRY,)" in code
    assert "source_path.startswith" in code
    assert "extractall" not in code
    assert "train_test_split" not in code and "torch.load" not in code
    assert "git" in code and "fetch" in code and "--ff-only" in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "gender_sam25_refit.ipynb", "exec")
            assert cell.execution_count is not None
            assert all(output.output_type != "error" for output in cell.outputs)
