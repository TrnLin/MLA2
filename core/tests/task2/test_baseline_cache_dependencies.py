from __future__ import annotations

import shutil

import pytest

from fashion.config import ROOT
from fashion.task2.experiments import _implementation_paths
from fashion.train.cache import implementation_sha256


@pytest.mark.parametrize("method", ["majority", "hog_hsv_svm"])
def test_dataset_logic_changes_invalidate_baseline_cache(tmp_path, method) -> None:
    paths = _implementation_paths(method)
    for relative in {*paths, "src/fashion/data/dataset.py"}:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    before = implementation_sha256(*paths, root=tmp_path)
    dataset = tmp_path / "src/fashion/data/dataset.py"
    dataset.write_bytes(dataset.read_bytes() + b"\n# Changed fold-selection implementation.\n")
    after = implementation_sha256(*paths, root=tmp_path)
    assert before != after
