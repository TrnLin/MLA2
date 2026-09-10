"""Check final Usage artifact identity and the two notebooks' evidence boundaries."""

import hashlib
import json
import shutil

import nbformat
import pytest

from fashion.config import ROOT
from fashion.task3_final import (
    CHECKPOINT_SHA256,
    CLASS_NAMES,
    PACK,
    _check_training_reference,
    check_hash,
    verify_usage_final,
    verify_usage_holdout_sources,
)

TASK1_PATH_ADDITIONS = (
    'TASK1_RESULT_DIR = RESULTS_DIR / "task1"\n'
    'TASK1_FIGURE_DIR = FIGURE_DIR / "task1"\n'
    'TASK1_EVIDENCE_DIR = EVIDENCE_DIR / "task1"\n'
    'TASK1_HOG_CACHE_DIR = PROCESSED_DATA_DIR / "task1_hog_cache"\n'
)


def test_task1_path_additions_do_not_invalidate_usage_model(tmp_path):
    # Copy only the shared file being changed; keep the accepted artifacts read-only.
    manifest = json.loads((ROOT / PACK / "model_manifest.json").read_text())
    paths = set(manifest["files"])
    paths.update(
        manifest[key]["path"]
        for key in ("checkpoint", "config", "normalization", "training_manifest")
    )
    paths.update(str(PACK / name) for name in ("model_manifest.json", "model_manifest.sha256"))
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "src/fashion/config.py":
            shutil.copyfile(ROOT / relative, target)
        else:
            target.symlink_to(ROOT / relative)
    config = tmp_path / "src/fashion/config.py"
    source = config.read_text()
    anchor = 'TASK2_EVIDENCE_DIR = EVIDENCE_DIR / "task2"\n'
    assert anchor in source
    config.write_text(source.replace(anchor, anchor + TASK1_PATH_ADDITIONS))
    assert verify_usage_final(tmp_path)["checkpoint"]["sha256"] == CHECKPOINT_SHA256


@pytest.mark.parametrize(
    "replacement",
    [
        ("RANDOM_SEED = 2753", "RANDOM_SEED = 42"),
        (
            'SPLITS_CSV = PROCESSED_DATA_DIR / "splits.csv"',
            'SPLITS_CSV = PROCESSED_DATA_DIR / "other.csv"',
        ),
        ("ROOT = _resolve_project_root()", 'ROOT = Path("/other")'),
    ],
)
def test_task1_additions_do_not_hide_changes_to_existing_settings(tmp_path, replacement):
    path = tmp_path / "src/fashion/config.py"
    path.parent.mkdir(parents=True)
    original = (ROOT / "src/fashion/config.py").read_bytes()
    source = original.decode()
    assert replacement[0] in source
    path.write_text(source.replace(*replacement) + TASK1_PATH_ADDITIONS)
    with pytest.raises(ValueError, match="Hash mismatch"):
        _check_training_reference(
            tmp_path, "src/fashion/config.py", hashlib.sha256(original).hexdigest()
        )


def test_unreviewed_shared_config_code_is_rejected(tmp_path):
    path = tmp_path / "src/fashion/config.py"
    path.parent.mkdir(parents=True)
    original = (ROOT / "src/fashion/config.py").read_bytes()
    path.write_bytes(original + b'\nTASK1_RESULT_DIR = Path("/different")\n')
    with pytest.raises(ValueError, match="Hash mismatch"):
        _check_training_reference(
            tmp_path, "src/fashion/config.py", hashlib.sha256(original).hexdigest()
        )


def test_accepted_e8_artifact_and_holdout_sources_are_intact():
    manifest = verify_usage_final()
    assert manifest["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert manifest["class_names"] == CLASS_NAMES
    assert manifest["inference"]["models"] == 1
    assert manifest["report_scope"]["teacher_test"] == "prediction_only"
    # Acceptance does not rewrite the original training receipt's scope.
    training = json.loads((ROOT / manifest["training_manifest"]["path"]).read_text())
    assert training["holdout_evaluated"] is False
    assert verify_usage_holdout_sources()["files"]


def test_modified_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "final_epoch.pt"
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="Hash mismatch"):
        check_hash(checkpoint, CHECKPOINT_SHA256)


def test_modified_acceptance_manifest_is_rejected(tmp_path):
    pack = tmp_path / PACK
    pack.mkdir(parents=True)
    (pack / "model_manifest.sha256").write_bytes(
        (ROOT / PACK / "model_manifest.sha256").read_bytes()
    )
    changed = json.loads((ROOT / PACK / "model_manifest.json").read_text())
    changed["selected_epoch"] = 29
    (pack / "model_manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_usage_final(tmp_path)


def test_holdout_analysis_stays_in_task3_evaluation_and_test_scores_stay_private():
    main = nbformat.read(ROOT / "notebooks/06_task3_part1_gender_usage.ipynb", as_version=4)
    final = nbformat.read(ROOT / "notebooks/07_task3_part2_final_evaluation.ipynb", as_version=4)
    main_code = "\n".join(c.source for c in main.cells if c.cell_type == "code")
    final_source = "\n".join(c.source for c in final.cells)
    assert "e1_e8_comparison.csv" not in main_code
    assert "verify_usage_holdout_sources" not in main_code
    assert "e1_e8_comparison.csv" in final_source
    assert "42.26%" in final_source and "6.17 F1 points" in final_source
    assert "not a new blind evaluation" in final_source
    for notebook in (main, final):
        content = json.dumps(notebook)
        for private_result in (
            "usage_refits_test_20260907",
            "refit_test_probabilities.csv",
            "30.35%",
            "83.36%",
            "85.81%",
            "25.74%",
        ):
            assert private_result not in content
        assert not any(
            output.output_type == "error"
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
