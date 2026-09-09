from pathlib import Path

import pytest

from fashion.train.task3_bundle_inputs import install_bundle_inputs


def test_bundle_inputs_keep_current_code_and_reuse_identical_files(tmp_path):
    bundle, root = tmp_path / "bundle", tmp_path / "checkout"
    for folder in ("data", "reference", "results", "src"):
        (bundle / folder).mkdir(parents=True)
        (bundle / folder / "value.csv").write_text("saved input")
    (root / "src").mkdir(parents=True)
    (root / "src/value.csv").write_text("current code")
    assert install_bundle_inputs(bundle, root) == 3
    assert install_bundle_inputs(bundle, root) == 3
    assert (root / "src/value.csv").read_text() == "current code"
    (root / "data/value.csv").write_text("local change")
    with pytest.raises(ValueError, match="Local input differs"):
        install_bundle_inputs(bundle, root)


def test_bundle_inputs_reject_code_in_data(tmp_path: Path):
    (tmp_path / "bundle/data").mkdir(parents=True)
    (tmp_path / "bundle/data/hidden.py").write_text("print('archive code')")
    with pytest.raises(ValueError, match="not a data file"):
        install_bundle_inputs(tmp_path / "bundle", tmp_path / "checkout")
