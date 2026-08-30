from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import fashion.task2.refit as refit_module
from fashion.config import ROOT
from fashion.data.dataset import load_label_maps
from fashion.data.hashing import compute_sha256
from fashion.data.torch import FoldImageStats
from fashion.task2.multitask import I2_IMPLEMENTATION_PATHS
from fashion.task2.refit import (
    REFIT_IMPLEMENTATION_PATHS,
    load_verified_development_refit_manifest,
    run_or_load_development_refit,
)
from fashion.train.artifacts import canonical_sha256
from fashion.train.multitask import RefitResult
from fashion.train.registry import RunRegistry


def _fake_loader() -> SimpleNamespace:
    mappings = load_label_maps()
    labels = tuple(str(value) for value in mappings["season"]["classes"])
    auxiliary_labels = tuple(str(value) for value in mappings["articleType"]["classes"])
    training_ids = tuple(range(32_753))
    stats = FoldImageStats(
        validation_fold=None,
        image_size=(80, 60),
        image_count=32_753,
        content_pixel_count=32_753 * 80 * 60,
        mean=(0.4, 0.45, 0.5),
        std=(0.2, 0.21, 0.22),
        training_id_sha256=canonical_sha256(sorted(training_ids)),
    )
    audit = {
        "partition": "development",
        "training_products": 32_753,
        "validation_products": 0,
        "protected_products": 0,
        "labels": list(labels),
        "auxiliary_target": "articleType",
        "auxiliary_labels": list(auxiliary_labels),
        "auxiliary_training_products": 32_753,
        "auxiliary_training_id_sha256": canonical_sha256(sorted(training_ids)),
        "training_id_sha256": canonical_sha256(sorted(training_ids)),
        "train_transform_id": "a0-7dba8a62f166046d",
        "normalisation_scope": "all_valid_development_content_pixels_only",
        "stats": stats.to_dict(),
    }
    return SimpleNamespace(
        train=object(),
        stats=stats,
        labels=labels,
        label_to_index={label: index for index, label in enumerate(labels)},
        auxiliary_labels=auxiliary_labels,
        auxiliary_label_to_index={label: index for index, label in enumerate(auxiliary_labels)},
        training_ids=training_ids,
        audit=lambda: audit,
    )


def test_refit_implementation_hash_covers_selected_i2_dependencies() -> None:
    assert set(I2_IMPLEMENTATION_PATHS).issubset(REFIT_IMPLEMENTATION_PATHS)


def _fake_train(model, train_loader, *, config, auxiliary_weight) -> RefitResult:
    del train_loader, auxiliary_weight
    history = [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "train_season_loss": 0.8 / epoch,
            "train_auxiliary_loss": 0.2 / epoch,
            "train_accuracy": min(0.5 + epoch / 100, 1.0),
            "train_samples": 32_753,
            "train_auxiliary_labeled_samples": 32_753,
            "learning_rate": config.learning_rate,
        }
        for epoch in range(1, config.epochs + 1)
    ]
    return RefitResult(
        seed=config.seed,
        final_epoch=config.epochs,
        epochs_completed=config.epochs,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=12.5,
        peak_vram_mb=256.0,
        device="cuda",
        metadata={
            "amp_enabled": True,
            "accumulation_steps": 4,
            "updates_completed": 6_144,
            "selection_metric": None,
            "validation_used": False,
            "early_stopping_used": False,
            "checkpoint_rule": "save_the_declared_final_epoch_state",
            "auxiliary_weight": 0.3,
        },
    )


def _patch_fast_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        refit_module,
        "capture_git_state",
        lambda root: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        refit_module,
        "verify_implementation_at_head",
        lambda *paths, root: tuple(REFIT_IMPLEMENTATION_PATHS),
    )
    monkeypatch.setattr(
        refit_module,
        "capture_runtime",
        lambda: {"python": "test", "cuda_available": True},
    )
    monkeypatch.setattr(
        refit_module,
        "build_development_multitask_loader",
        lambda **kwargs: _fake_loader(),
    )
    monkeypatch.setattr(refit_module, "train_masked_multitask_refit", _fake_train)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "registry_path": root / "runs.csv",
        "bundle_path": root / "task2_season.pt",
        "manifest_path": root / "task2_season.manifest.json",
        "history_path": root / "training_history.csv",
        "runtime_path": root / "runtime.json",
    }


def test_refit_writes_fixed_epoch_bundle_registry_and_verifiable_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))

        outcome = run_or_load_development_refit(
            mode="run",
            project_root=ROOT,
            **paths,
        )
        manifest, manifest_path, bundle = load_verified_development_refit_manifest(
            paths["manifest_path"],
            project_root=ROOT,
            registry_path=paths["registry_path"],
        )
        registry = RunRegistry(paths["registry_path"]).read()
        history = pd.read_csv(paths["history_path"])

        assert outcome.source == "run"
        assert manifest_path == paths["manifest_path"]
        assert manifest["selected_candidate"] == "I2"
        assert manifest["final_epoch"] == 24
        assert manifest["valid_development_rows"] == 32_753
        assert manifest["holdout_opened"] is False
        assert manifest["validation_used"] is False
        assert manifest["primary_metric_name"] is None
        assert bundle["training"]["final_epoch"] == 24
        assert bundle["inference"]["inputs"] == ["image"]
        assert bundle["auxiliary"]["used_at_inference"] is False
        assert history["epoch"].tolist() == list(range(1, 25))
        assert not any("validation" in column for column in history.columns)
        assert len(registry) == 1
        assert registry.loc[0, "fold"] == ""
        assert registry.loc[0, "status"] == "completed"
        assert registry.loc[0, "primary_metric_name"] == ""
        assert registry.loc[0, "primary_metric_value"] == ""
        assert registry.loc[0, "best_epoch"] == ""
        assert registry.loc[0, "checkpoint_sha256"] == outcome.bundle_sha256

        loaded = run_or_load_development_refit(
            mode="run_or_load",
            project_root=ROOT,
            **paths,
        )
        assert loaded.source == "load"
        assert loaded.bundle_sha256 == outcome.bundle_sha256
        assert len(RunRegistry(paths["registry_path"]).read()) == 1

        tampered_bundle = torch.load(paths["bundle_path"], map_location="cpu", weights_only=True)
        tampered_bundle["labels"] = list(reversed(tampered_bundle["labels"]))
        torch.save(tampered_bundle, paths["bundle_path"])
        tampered_manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        tampered_manifest["bundle"]["sha256"] = compute_sha256(paths["bundle_path"])
        paths["manifest_path"].write_text(json.dumps(tampered_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="bundle metadata disagrees"):
            load_verified_development_refit_manifest(
                paths["manifest_path"],
                project_root=ROOT,
                registry_path=paths["registry_path"],
            )


def test_refit_manifest_rejects_a_claim_that_holdout_was_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        run_or_load_development_refit(mode="run", project_root=ROOT, **paths)
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        manifest["holdout_opened"] = True
        paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="boundary changed"):
            load_verified_development_refit_manifest(
                paths["manifest_path"],
                project_root=ROOT,
                registry_path=paths["registry_path"],
            )


@pytest.mark.parametrize(
    ("field", "integer_value"),
    [
        ("scratch", 1),
        ("holdout_opened", 0),
        ("model_change_after_holdout_allowed", 0),
    ],
)
def test_refit_manifest_rejects_integer_substitutes_for_booleans(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    integer_value: int,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        run_or_load_development_refit(mode="run", project_root=ROOT, **paths)
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        manifest[field] = integer_value
        paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="boolean"):
            load_verified_development_refit_manifest(
                paths["manifest_path"],
                project_root=ROOT,
                registry_path=paths["registry_path"],
            )


def test_refit_refuses_to_replace_an_unmanifested_model_bundle() -> None:
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        paths["bundle_path"].write_bytes(b"unverified")

        with pytest.raises(FileExistsError, match="unmanifested"):
            run_or_load_development_refit(
                mode="run_or_load",
                project_root=ROOT,
                **paths,
            )


def test_refit_load_mode_requires_an_existing_manifest() -> None:
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        with pytest.raises(FileNotFoundError, match="manifest does not exist"):
            run_or_load_development_refit(
                mode="load",
                project_root=ROOT,
                **_paths(Path(directory)),
            )


def test_refit_registry_stays_failed_when_manifest_publish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    original_write_json = refit_module.atomic_write_json
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))

        def fail_manifest_publish(path: str | Path, value: object) -> Path:
            if Path(path) == paths["manifest_path"]:
                raise OSError("simulated manifest publish failure")
            return original_write_json(path, value)

        monkeypatch.setattr(refit_module, "atomic_write_json", fail_manifest_publish)

        with pytest.raises(OSError, match="manifest publish failure"):
            run_or_load_development_refit(
                mode="run",
                project_root=ROOT,
                **paths,
            )

        registry = RunRegistry(paths["registry_path"]).read()
        assert len(registry) == 1
        assert registry.loc[0, "status"] == "failed"
        assert registry.loc[0, "error_type"] == "OSError"
        assert not paths["bundle_path"].exists()
        assert not paths["manifest_path"].exists()
        assert not paths["history_path"].exists()
        assert not paths["runtime_path"].exists()


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing", "registry"),
        ("failed", "completed"),
        ("checkpoint_hash", "checkpoint"),
    ],
)
def test_refit_load_rejects_missing_or_tampered_registry_row(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_message: str,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        run_or_load_development_refit(mode="run", project_root=ROOT, **paths)

        if mutation == "missing":
            paths["registry_path"].unlink()
        else:
            registry = pd.read_csv(paths["registry_path"], dtype=str, keep_default_na=False)
            if mutation == "failed":
                registry.loc[0, "status"] = "failed"
            else:
                registry.loc[0, "checkpoint_sha256"] = "0" * 64
            registry.to_csv(paths["registry_path"], index=False)

        with pytest.raises(ValueError, match=expected_message):
            run_or_load_development_refit(
                mode="load",
                project_root=ROOT,
                **paths,
            )


def test_refit_rejects_a_parallel_live_process_before_writing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        lock_path = paths["manifest_path"].parent / ".task2-season-refit.lock"
        lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

        with pytest.raises(RuntimeError, match="already running"):
            run_or_load_development_refit(
                mode="run",
                project_root=ROOT,
                **paths,
            )

        assert not paths["registry_path"].exists()
        assert not paths["bundle_path"].exists()


def test_refit_load_rejects_non_finite_model_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        run_or_load_development_refit(mode="run", project_root=ROOT, **paths)

        bundle = torch.load(paths["bundle_path"], map_location="cpu", weights_only=True)
        first_tensor = next(iter(bundle["model_state_dict"].values()))
        first_tensor.view(-1)[0] = float("nan")
        torch.save(bundle, paths["bundle_path"])
        bundle_sha256 = compute_sha256(paths["bundle_path"])
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        manifest["bundle"]["sha256"] = bundle_sha256
        paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
        registry = pd.read_csv(paths["registry_path"], dtype=str, keep_default_na=False)
        registry.loc[0, "checkpoint_sha256"] = bundle_sha256
        registry.to_csv(paths["registry_path"], index=False)

        with pytest.raises(ValueError, match="non-finite"):
            load_verified_development_refit_manifest(
                paths["manifest_path"],
                project_root=ROOT,
                registry_path=paths["registry_path"],
            )


def test_refit_load_rejects_non_finite_training_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_refit(monkeypatch)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as directory:
        paths = _paths(Path(directory))
        run_or_load_development_refit(mode="run", project_root=ROOT, **paths)

        history = pd.read_csv(paths["history_path"])
        history.loc[0, "train_loss"] = float("inf")
        history.to_csv(paths["history_path"], index=False)
        history_sha256 = compute_sha256(paths["history_path"])
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        manifest["artifacts"]["history"]["sha256"] = history_sha256
        paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
        registry = pd.read_csv(paths["registry_path"], dtype=str, keep_default_na=False)
        registry.loc[0, "history_sha256"] = history_sha256
        registry.to_csv(paths["registry_path"], index=False)

        with pytest.raises(ValueError, match="non-finite"):
            load_verified_development_refit_manifest(
                paths["manifest_path"],
                project_root=ROOT,
                registry_path=paths["registry_path"],
            )
