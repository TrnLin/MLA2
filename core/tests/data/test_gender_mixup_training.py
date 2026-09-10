"""Exercise the complete shared trainer on tiny local data, without a GPU fit."""

import json
from pathlib import Path

import pandas as pd
import pytest
from test_gender_name_truth import project as project

import fashion.train.task3_gender_mixup as screen
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import build_gender_name_truth_variant
from fashion.train.config import Task3BaselineConfig

CLASSES = ["Boys", "Girls", "Men", "Unisex", "Women"]


def test_mixup_keeps_canonical_rows_and_requires_source_evidence(project):
    from fashion.train.task3_gender_name_truth import name_truth_spec
    from fashion.train.task3_gender_name_truth import training_splits as parent_splits

    build_gender_name_truth_variant(project)
    spec = screen.mixup_spec(project)
    splits = screen.training_splits(spec, root=project)
    pd.testing.assert_frame_equal(splits, parent_splits(name_truth_spec(project), root=project))
    with pytest.raises(ValueError, match="audit is required"):
        screen.require_mixup_prerequisites(None, spec=spec, fold=0, root=project)
    with pytest.raises(ValueError, match="frozen label"):
        screen.training_splits(
            screen.MixUpSpec(json.dumps({"labels_sha256": "wrong"})), root=project
        )


def test_complete_training_loop_saves_verifiable_unmixed_gap_and_mixup_receipt(
    project, monkeypatch
):
    torch = pytest.importorskip("torch")
    from PIL import Image

    import fashion.train.task3_baseline as trainer
    from fashion.train.task3_g2_audit import inspect_gender_run

    build_gender_name_truth_variant(project)
    spec = screen.mixup_spec(project)
    splits = screen.training_splits(spec, root=project)
    mapping = dict(zip(CLASSES, range(5)))
    (project / "data/processed/label_maps.json").write_text(
        json.dumps(
            {
                "gender": {
                    "classes": CLASSES,
                    "label_to_index": mapping,
                    "source_scope": "development",
                }
            }
        )
    )
    for i, relative in enumerate(splits.path):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 80), (40 + i * 5, 110, 190 - i * 3)).save(path)
    parent = project / "parent"
    parent.mkdir()
    (parent / "metrics.json").write_text(
        json.dumps({"run_id": spec.parent_run_id_for_fold(0), "validation_fold": 0})
    )
    for name in ("final_epoch.pt", "oof_predictions.csv", "robustness.csv"):
        (parent / name).write_text("test parent evidence; no weights are loaded")
    registry = project / "results/runs.csv"
    # Only the GPU/evidence boundary is stubbed. Use the real model, images,
    # optimizer, scheduler, 30-epoch loop, checkpoint, registry and saved-run audit.
    monkeypatch.setattr(
        screen, "mixup_config", lambda *a, **kw: Task3BaselineConfig(target="gender")
    )
    monkeypatch.setattr(
        screen,
        "require_mixup_prerequisites",
        lambda *a, **kw: {
            "precision": {"status": {"runtime_default_settings": {}}, "artifact_sha256": {}},
            "parent_directory": parent,
            "prerequisite_sha256": "test-boundary",
        },
    )
    monkeypatch.setattr(
        trainer,
        "_loader",
        lambda dataset, *, config, shuffle, device: torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0
        ),
    )
    monkeypatch.setattr(trainer, "_measure_latency", lambda *a, **kw: 1.0)
    actual_pass = trainer._pass
    seen = []

    def checked_pass(model, loader, criterion, device, **kwargs):
        mixed = kwargs.get("mixup") is not None
        training = kwargs.get("optimizer") is not None
        assert mixed == training
        assert type(criterion) is torch.nn.CrossEntropyLoss and criterion.weight is None
        if training:
            rows = pd.read_csv(registry)
            assert rows.status.tolist() == ["running"]
        seen.append((mixed, len(loader.dataset)))
        return actual_pass(model, loader, criterion, device, **kwargs)

    monkeypatch.setattr(trainer, "_pass", checked_pass)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = trainer.run_task3_baseline_fold(
            "gender",
            0,
            output_root=project,
            root=project,
            child_spec=spec,
            registry_path=registry,
            registry_mirrors=(project / "mirror.csv",),
            device_name="cpu",
        )
    finally:
        torch.set_num_threads(previous_threads)
    run_dir = Path(result["run_dir"])
    run = inspect_gender_run(
        run_dir,
        registry=pd.read_csv(registry, keep_default_na=False),
        splits=splits,
        classes=CLASSES,
        root=project,
    )
    screen.verify_mixup_evidence(run, fold=0, splits=splits, directory=run_dir)
    history = pd.read_csv(run_dir / "history.csv")
    assert history.epoch.tolist() == list(range(1, 31)) and history.train_macro_f1.isna().all()
    assert history.train_loss.notna().all() and history.validation_macro_f1.notna().all()
    training = get_samples(get_cv_split(splits, 0)[0], target="gender")
    assert sum(mixed for mixed, _ in seen) == 30
    assert all(rows == len(training) for mixed, rows in seen if mixed)
    assert any(not mixed and rows == len(training) for mixed, rows in seen)
    metrics = run["metrics"]
    assert metrics["parameter_count"] == 390181
    assert metrics["selected_epoch"] == 30 and not metrics["early_stopped"]
    assert metrics["final_train_validation_macro_f1_gap"] == pytest.approx(
        metrics["final_train_eval_macro_f1"] - metrics["macro_f1"]
    )
    assert pd.read_csv(registry).status.tolist() == ["complete"]
    assert registry.read_bytes() == (project / "mirror.csv").read_bytes()
    assert not pd.read_csv(registry).submission_eligible.iloc[0]


def test_prerequisites_bind_mixup_code_rows_labels_and_direct_parent(project, monkeypatch):
    build_gender_name_truth_variant(project)
    spec = screen.mixup_spec(project)
    for name in ("mixup.py", "task3_gender_mixup.py", "task3_baseline.py"):
        implementation = project / "src/fashion/train" / name
        implementation.parent.mkdir(parents=True, exist_ok=True)
        implementation.write_text("frozen implementation")
    sources = {
        "NameTruth": {
            f: {
                "run_id": spec.parent_run_id_for_fold(f),
                "sha256": {"config.json": f"verified-{f}"},
                "directory": str(project / f"parent-{f}"),
            }
            for f in (0, 4)
        }
    }
    evidence = {"artifact_sha256": {"precision": "verified"}}
    paths = {"precision_directory": str(project / "precision")}
    identity = screen._source_identity(sources, spec, evidence, paths, root=project)
    audit = project / "source_audit.json"
    audit.write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(
        screen, "check_gender_mixup_sources", lambda **kw: (sources, [], spec, evidence)
    )
    monkeypatch.setattr(screen, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    assert (
        screen.require_mixup_prerequisites(audit, spec=spec, fold=0, root=project)[
            "parent_directory"
        ]
        == project / "parent-0"
    )
    with pytest.raises(ValueError, match="verified name-truth parent"):
        screen.require_mixup_prerequisites(
            audit, spec=spec, fold=0, root=project, parent_run_directory=project / "parent-4"
        )
    changed = json.loads(audit.read_text())
    changed["identity"]["mixup_contracts"]["0"]["training_rows"] += 1
    audit.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="evidence changed"):
        screen.require_mixup_prerequisites(audit, spec=spec, fold=0, root=project)
    audit.write_text(json.dumps({"identity": identity}))
    implementation.write_text("changed implementation")
    with pytest.raises(ValueError, match="evidence changed"):
        screen.require_mixup_prerequisites(audit, spec=spec, fold=0, root=project)
