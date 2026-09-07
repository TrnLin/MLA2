"""Protect the accepted Gender model identity and the saved evaluation boundary."""

import json

import nbformat
import numpy as np
import pandas as pd
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
    assert final["inference"]["models"] == 1
    training = json.loads((ROOT / final["training_manifest"]["path"]).read_text())
    assert training["holdout_evaluated"] is False
    assert training["validation_used"] is False


def test_changed_final_manifest_is_rejected(tmp_path):
    pack = tmp_path / PACK
    pack.mkdir(parents=True)
    (pack / "model_manifest.sha256").write_bytes(
        (ROOT / PACK / "model_manifest.sha256").read_bytes()
    )
    manifest = json.loads((ROOT / PACK / "model_manifest.json").read_text())
    manifest["selected_epoch"] = 24
    (pack / "model_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_gender_final(tmp_path)


def test_holdout_tables_match_saved_confusion_counts():
    verify_gender_holdout_sources()
    pack = ROOT / "reports/task3/gender_sam25_refit_holdout_20260907"
    evaluation = json.loads((pack / "evaluation.json").read_text())
    summary = pd.read_csv(pack / "holdout_comparison.csv")
    for row in summary.itertuples():
        cm = np.asarray(evaluation["metrics"][f"{row.labels}: {row.model}"]["confusion_matrix"])
        assert cm.sum() == 5778
        assert np.trace(cm) / cm.sum() == pytest.approx(row.accuracy)
        denominator = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(len(cm)), where=denominator > 0)
        assert f1.mean() == pytest.approx(row.macro_f1)


def test_gender_holdout_results_stay_in_final_notebook():
    main = nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4)
    final = nbformat.read(ROOT / "notebooks/04_task3_final_evaluation.ipynb", as_version=4)
    main_code = "\n".join(c.source for c in main.cells if c.cell_type == "code")
    final_code = "\n".join(c.source for c in final.cells if c.cell_type == "code")
    assert "verify_gender_final" in main_code
    assert "gender_sam25_refit_holdout_20260907" not in main_code
    assert "verify_gender_holdout_sources" not in main_code
    assert "verify_gender_holdout_sources" in final_code
    assert "load_splits_for_final_evaluation" not in final_code
    assert "torch.load" not in final_code
