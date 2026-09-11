"""Protect the accepted Gender model identity and the saved evaluation boundary."""

import json

import nbformat
import numpy as np
import pytest

from fashion.config import ROOT
from fashion.task3_gender_final import (
    CHECKPOINT_SHA256,
    PACK,
    verify_gender_final,
    verify_gender_holdout_sources,
)


def test_accepted_gender_refit_preserves_original_training_receipt():
    final = verify_gender_final()
    assert final["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert final["selected_epoch"] == 30
    assert final["metrics"]["holdout_evaluated"] is False
    assert final["validation_used"] is False


def test_changed_final_manifest_is_rejected(tmp_path):
    pack = tmp_path / PACK
    pack.mkdir(parents=True)
    manifest = json.loads((ROOT / PACK / "model_manifest.json").read_text())
    manifest["files"]["final_epoch.pt"]["sha256"] = "different checkpoint"
    (pack / "model_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        verify_gender_final(tmp_path)


def test_holdout_tables_match_saved_confusion_counts():
    verify_gender_holdout_sources()
    pack = ROOT / "reports/task3/gender_mixup_refit_holdout_20260911"
    evaluation = json.loads((pack / "evaluation.json").read_text())
    metrics = evaluation["metrics"]
    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.sum() == 5778 and np.trace(cm) == 5227
    assert np.trace(cm) / cm.sum() == pytest.approx(metrics["accuracy"])
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(len(cm)), where=denominator > 0)
    assert f1.mean() == pytest.approx(metrics["macro_f1"])


def test_gender_holdout_results_stay_in_final_notebook():
    main = nbformat.read(ROOT / "notebooks/06_task3_part1_gender_usage.ipynb", as_version=4)
    final = nbformat.read(ROOT / "notebooks/07_task3_part2_final_evaluation.ipynb", as_version=4)
    main_code = "\n".join(c.source for c in main.cells if c.cell_type == "code")
    final_code = "\n".join(c.source for c in final.cells if c.cell_type == "code")
    assert "verify_mixup_selection" in main_code
    assert "gender_sam25_refit_holdout_20260907" not in main_code
    assert "verify_gender_holdout_sources" not in main_code
    assert "load_selected_evaluation" in final_code
    assert "gender_sam25" not in final_code
    assert "load_splits_for_final_evaluation" not in final_code
    assert "torch.load" not in final_code
