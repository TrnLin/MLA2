"""Evaluated source identity must survive edits to the current runnable scripts."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_usage", ROOT / "scripts/verify_task3_usage_manifest.py"
)
verify_usage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_usage)


def test_frozen_sources_resolve_to_exact_archives():
    pack = ROOT / verify_usage.PACK
    mapping = json.loads((pack / "source-archive-map.json").read_text())
    manifest = json.loads((pack / "model_manifest.json").read_text())
    assert len(list(verify_usage.references(manifest))) == 38
    assert len(mapping["evaluated_sources"]) == 2
    for original, entry in mapping["evaluated_sources"].items():
        archived = verify_usage.resolve_frozen(original, mapping)
        current = ROOT / entry["current_path"]
        assert archived != current
        assert archived.is_relative_to(pack / "evaluated_sources")
        verify_usage.check_hash(archived, manifest["source_contracts"][original]["sha256"])
        verify_usage.check_hash(current, entry["current_sha256"])
        with pytest.raises(ValueError, match="Hash mismatch"):
            verify_usage.check_hash(current, entry["evaluated_sha256"])


def test_changed_archive_is_rejected(tmp_path):
    path = tmp_path / "evaluated.py"
    path.write_text("changed source")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_usage.check_hash(path, "0" * 64)


def test_other_frozen_paths_use_the_reorganized_checkout(tmp_path):
    mapping = {"evaluated_sources": {}}
    assert verify_usage.resolve_frozen("reports/task3_trial/result.csv", mapping, tmp_path) == (
        tmp_path / "reports/task3/trial/result.csv"
    )
    assert verify_usage.resolve_frozen("results/run/model.pt", mapping, tmp_path) == (
        tmp_path / "results/run/model.pt"
    )
