"""Check final Usage artifact identity and the two notebooks' evidence boundaries."""

import json

import nbformat
import pytest

from fashion.config import ROOT
from fashion.task3_final import (
    CHECKPOINT_SHA256,
    CLASS_NAMES,
    PACK,
    check_hash,
    verify_usage_final,
    verify_usage_holdout_sources,
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
    main = nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4)
    final = nbformat.read(ROOT / "notebooks/04_task3_final_evaluation.ipynb", as_version=4)
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
