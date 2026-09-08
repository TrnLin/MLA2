from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fashion.config import ROOT
from fashion.task2 import refit
from fashion.task2.inference import load_season_bundle, predict_season


def test_frozen_bundle_loads_without_historical_git_objects(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise ValueError("source-only submission has no Git history")

    monkeypatch.setattr(refit, "implementation_sha256_at_commit", unavailable)
    manifest, _, payload = refit.load_verified_development_refit_manifest(project_root=ROOT)

    assert manifest["bundle"]["sha256"] == (
        "5927eff73130acedc8015199e1df5a6c6edf64c0b45023ebd91c48d7ed40f93c"
    )
    assert payload["training"]["final_epoch"] == 24


def test_archive_verification_still_rejects_changed_runtime(monkeypatch) -> None:
    original = refit.implementation_sha256

    def digest(*paths, root):
        if paths == refit.REFIT_LOAD_COMPATIBILITY_PATHS:
            return "0" * 64
        return original(*paths, root=root)

    monkeypatch.setattr(refit, "implementation_sha256", digest)
    with pytest.raises(ValueError, match="runtime compatibility bytes changed"):
        refit.load_verified_development_refit_manifest(project_root=ROOT)


def _archive_copy(tmp_path: Path) -> Path:
    """Copy only the files consumed by bundle verification, without Git or raw data."""
    for relative in ("src", "configs/task2"):
        shutil.copytree(ROOT / relative, tmp_path / relative, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc"
        ))
    paths = {
        "models/task2_season.manifest.json",
        "results/evidence/task2/selection_freeze.json",
        "results/evidence/task2/development_refit/source_provenance.json",
        "results/evidence/task2/final_handoff/registry_snapshot.csv",
    }

    def declarations(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= set(value):
                paths.add(value["path"])
            for child in value.values():
                declarations(child)
        elif isinstance(value, list):
            for child in value:
                declarations(child)

    for relative in ("models/task2_season.manifest.json",
                     "results/evidence/task2/selection_freeze.json"):
        declarations(json.loads((ROOT / relative).read_text(encoding="utf-8")))
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_real_source_archive_loads_and_predicts(tmp_path) -> None:
    from PIL import Image

    root = _archive_copy(tmp_path)
    assert not (root / ".git").exists()
    bundle = load_season_bundle(
        "models/task2_season.manifest.json", project_root=root, device="cpu",
        registry_path="results/evidence/task2/final_handoff/registry_snapshot.csv",
    )
    image = root / "example.png"
    Image.new("RGB", (60, 80), (220, 220, 220)).save(image)
    prediction = predict_season(bundle, image)
    assert prediction.predicted_label in ("Fall", "Spring", "Summer", "Winter")
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_archive_provenance_tampering_is_rejected(tmp_path) -> None:
    root = _archive_copy(tmp_path)
    path = root / "results/evidence/task2/development_refit/source_provenance.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["runtime_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="trust anchor changed"):
        refit._load_verified_development_refit_package(
            "models/task2_season.manifest.json", project_root=root
        )
