"""Usage E1 boundaries, real scratch fitting and append-only rerun lifecycle."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from fashion.config import ROOT, TARGET_COLUMNS  # noqa: E402
from fashion.data.hashing import compute_sha256  # noqa: E402
from fashion.train import task3_usage_e1_refit as refit  # noqa: E402
from fashion.train.registry import RunRegistry  # noqa: E402


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(2753)
    rows = []
    images = tmp_path / "data/raw/teacher/train/images_train"
    images.mkdir(parents=True)
    for fold in range(5):
        for label in refit.CLASSES:
            image_id = len(rows) + 1
            image = images / f"{image_id}.png"
            # Non-square content checks normalization excludes letterbox padding.
            Image.fromarray(rng.integers(0, 220, (40, 60, 3), dtype=np.uint8)).save(image)
            row = dict(
                id=image_id,
                path=str(image.relative_to(tmp_path)),
                sha256=compute_sha256(image),
                product_family_group=f"f{image_id}",
                duplicate_group=f"d{image_id}",
                product_name_key=f"n{image_id}",
                partition="development",
                cv_fold=fold,
                is_cross_role_exact_duplicate=False,
                is_cross_role_near_duplicate=False,
                has_conflicting_target_labels=False,
                conflicting_targets="",
                quarantine_reason="",
            )
            for target in TARGET_COLUMNS:
                row[target] = label if target == "usage" else ""
                row[f"has_{target}_label"] = target == "usage"
            rows.append(row)
    protected = dict(
        rows[0],
        id=999,
        partition="holdout",
        cv_fold="",
        path="sealed.png",
        sha256="sealed",
        product_family_group="sealed",
        duplicate_group="sealed",
        product_name_key="sealed",
    )
    protected.update({t: "" for t in TARGET_COLUMNS})
    protected.update({f"has_{t}_label": False for t in TARGET_COLUMNS})
    rows.append(protected)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(processed / "splits.csv", index=False)
    (processed / "label_maps.json").write_text(
        json.dumps(
            {
                "usage": {
                    "source_scope": "development",
                    "classes": refit.CLASSES,
                    "label_to_index": {n: i for i, n in enumerate(refit.CLASSES)},
                }
            }
        )
    )
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    manifest = json.loads((ROOT / refit.SOURCE).read_text())
    for name in ("data/processed/splits.csv", "data/processed/label_maps.json"):
        manifest["source_contracts"][name]["sha256"] = compute_sha256(tmp_path / name)
    path = tmp_path / refit.SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(refit, "SOURCE_SHA256", compute_sha256(path))
    monkeypatch.setattr(refit, "EXPECTED_ROWS", 45)
    monkeypatch.setattr(refit, "_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(refit.subprocess, "check_output", lambda *a, **kw: "fixture\n")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield tmp_path
    torch.set_num_threads(previous)


def test_population_recipe_and_changed_inputs(fixture):
    config, payload, training = refit.prepare_refit(root=fixture)
    assert len(training) == 45 and set(training.cv_fold) == set(range(5))
    assert training.partition.eq("development").all() and 999 not in set(training.id)
    assert config.to_dict() == refit.Task3BaselineConfig(target="usage").to_dict()
    assert payload["parameter_count"] == 391209 and payload["validation_used"] is False
    assert payload["training_class_counts"]["NA"] == 5
    assert not (fixture / "sealed.png").exists()
    (fixture / training.iloc[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input image"):
        refit.prepare_refit(root=fixture)


@pytest.mark.parametrize("changed", ["manifest", "split", "label_map"])
def test_frozen_contract_tampering_fails(fixture, changed):
    path = {
        "manifest": refit.SOURCE,
        "split": Path("data/processed/splits.csv"),
        "label_map": Path("data/processed/label_maps.json"),
    }[changed]
    with (fixture / path).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="changed"):
        refit.prepare_refit(root=fixture)


def test_real_30_epoch_fit_checkpoint_mirrors_and_idempotency(fixture, monkeypatch):
    registry = fixture / "results/runs.csv"
    mirror = fixture / "mirror/runs.csv"
    other = RunRegistry(registry, mirrors=[mirror])
    other.start({"run_id": "other_gender", "target": "gender", "experiment_id": "other"})
    other.complete("other_gender", {})
    before = pd.read_csv(registry, keep_default_na=False).iloc[0].to_dict()
    result = refit.run_usage_e1_refit(
        root=fixture, output_root=fixture / "out", registry_mirrors=[mirror]
    )
    directory = Path(result["manifest_path"]).parent
    assert result["training_completed"] and not result["evaluation_completed"]
    assert not result["new_blind_evaluation"]
    checkpoint = torch.load(directory / "final_epoch.pt", weights_only=True)
    model = refit.Task3BaselineCNN(refit.Task3BaselineConfig(target="usage"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["selected_epoch"] == checkpoint["epochs_completed"] == 30
    assert checkpoint["class_names"] == refit.CLASSES
    assert checkpoint["normalization"]["total_pixels"] == 45 * 40 * 60
    history = pd.read_csv(directory / "history.csv")
    expected_lr = 1e-5 + 0.5 * (0.001 - 1e-5) * (1 + np.cos(np.pi * np.arange(30) / 30))
    np.testing.assert_allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
    assert history.selected_checkpoint.tolist() == [False] * 29 + [True]
    assert history.training_rows.eq(45).all()
    assert not any("validation" in c for c in history.columns)
    assert not list(directory.glob("*predictions*"))
    for path in (registry, mirror):
        rows = pd.read_csv(path, keep_default_na=False)
        assert len(rows) == 2 and rows.iloc[0].to_dict() == before
        row = rows.iloc[1]
        assert row.status == "complete" and row.validation_fold == ""
        assert row.training_product_count == "45" or row.training_product_count == 45
        assert "macro_f1" not in json.loads(row.metrics_json)
    monkeypatch.setattr(refit, "_fit", lambda *a, **kw: pytest.fail("Second physical fit"))
    assert refit.run_usage_e1_refit(
        root=fixture, output_root=fixture / "out", registry_mirrors=[mirror]
    )["reused"]
    with pytest.raises(ValueError, match="already registered"):
        refit.run_usage_e1_refit(root=fixture, output_root=fixture / "other")
    (directory / "final_epoch.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="artifact changed"):
        refit.run_usage_e1_refit(root=fixture, output_root=fixture / "out")


def test_interrupted_fit_is_registered_and_cannot_silently_restart(fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise KeyboardInterrupt("synthetic stop")

    monkeypatch.setattr(refit, "_fit", fail)
    with pytest.raises(KeyboardInterrupt):
        refit.run_usage_e1_refit(root=fixture, output_root=fixture / "out")
    rows = pd.read_csv(fixture / "results/runs.csv", keep_default_na=False)
    assert len(rows) == 1 and rows.iloc[0].status == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        refit.run_usage_e1_refit(root=fixture, output_root=fixture / "out")


def test_notebook_is_unexecuted_and_compiles():
    nb = nbformat.read(ROOT / "notebooks/task3_training/usage_e1_refit.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "usage_e1_refit.ipynb", "exec")
