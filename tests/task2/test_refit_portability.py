from __future__ import annotations

import pytest

from fashion.config import ROOT
from fashion.task2 import refit


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
