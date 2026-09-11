"""Protect the development boundary and selected-refit metric provenance."""

import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task3_development import (
    development_path,
    gender_development_comparison,
    selection_tables,
    verify_development_files,
)
from fashion.task3_refit_evaluation import family_score_interval, load_selected_evaluation
from fashion.train.metrics import classification_metrics


def test_development_lock_does_not_open_final_evaluation_assets(tmp_path):
    good = tmp_path / "reports/task3/development.csv"
    good.parent.mkdir(parents=True)
    good.write_text("score\n0.8\n")
    # Forbidden files deliberately do not exist: reading them would fail.
    files = {
        "reports/task3/development.csv": hashlib.sha256(good.read_bytes()).hexdigest(),
        "reports/task3/model_holdout/summary.csv": "not-readable",
        "reports/task3/model_test/test_summary.csv": "not-readable",
    }
    assert verify_development_files(tmp_path, files) == 1
    with pytest.raises(ValueError, match="outside development scope"):
        development_path(tmp_path, "reports/task3/model_holdout/summary.csv")
    good.write_text("changed")
    with pytest.raises(AssertionError):
        verify_development_files(tmp_path, files)


def test_full_gender_comparison_preserves_stages_and_separates_scoring_labels():
    selection, _, _ = selection_tables(ROOT)
    comparison = gender_development_comparison(ROOT, selection)
    registered = pd.read_csv(
        ROOT / "reports/task3/gender_mixup_selection_20260911/all_gender_development_stages.csv"
    )
    originals = comparison.loc[comparison["Score source"].eq("Registry")]
    assert originals.stage.tolist() == registered.stage.tolist()
    np.testing.assert_allclose(originals["F1 (%)"], registered.macro_f1 * 100)
    audited = comparison.loc[comparison["Score source"].eq("IEEE audit")]
    assert len(audited) == 4 and audited.rows.eq(32773).all()
    mixup = audited.loc[audited.experiment.eq("MixUp · Five-fold audit")].set_index("label_basis")
    assert mixup.loc["Original", "F1 (%)"] == pytest.approx(75.73372366824067)
    assert mixup.loc["Name-corrected", "F1 (%)"] == pytest.approx(
        selection["metrics"]["MixUp"]["macro_f1"] * 100
    )


def test_final_replay_uses_the_selected_refits_and_recomputes_their_scores():
    results = load_selected_evaluation(ROOT)
    assert set(results) == {"Gender MixUp", "Usage E8"}
    gender = results["Gender MixUp"]
    assert gender["manifest"]["selected_epoch"] == 30
    assert gender["manifest"]["checkpoint"]["sha256"] == (
        "860f688162cccfcd903e8e4874b368697c0637b6e6a15baae3b4d4a3008f4ef9"
    )
    assert sum(gender["rows"].actual.eq(gender["rows"].predicted)) == 5227
    assert int(np.trace(results["Usage E8"]["metrics"]["confusion_matrix"])) == 5125


def test_family_interval_matches_explicit_whole_family_sampling():
    rows = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "product_family_group": ["a", "a", "b", "c", "c"],
            "true_index": [0, 1, 1, 0, 1],
            "predicted_index": [0, 0, 1, 1, 1],
        }
    )
    classes = ["A", "B", "Absent"]
    result = family_score_interval(rows, classes, draws=80, seed=7)
    rng = np.random.default_rng(7)
    scores = []
    families = [rows.loc[rows.product_family_group.eq(name)] for name in ["a", "b", "c"]]
    for _ in range(80):
        sampled = pd.concat([families[i] for i in rng.integers(0, 3, 3)])
        probabilities = np.eye(3)[sampled.predicted_index]
        scores.append(
            classification_metrics(sampled.true_index.to_numpy(), probabilities, classes)[
                "macro_f1"
            ]
        )
    np.testing.assert_allclose(
        [result["lower_95"], result["upper_95"]], np.quantile(scores, [0.025, 0.975])
    )
    assert result["families"] == 3
    with pytest.raises(ValueError):
        family_score_interval(pd.concat([rows, rows.iloc[:1]]), classes)


def test_retained_usage_evaluation_contains_only_the_selected_refit():
    folder = ROOT / "reports/task3/usage_e8_refit_holdout_20260907"
    evaluation = json.loads((folder / "evaluation.json").read_text())
    assert set(evaluation["metrics"]) == {"Single refit"}
    assert "paired_changes" not in evaluation
    per_class = pd.read_csv(folder / "holdout_per_class.csv", keep_default_na=False)
    assert set(per_class.model) == {"Single refit"}
    assert len(per_class) == 9
    labels = pd.read_csv(folder / "holdout_predictions_and_labels.csv", keep_default_na=False)
    assert labels.columns.tolist() == [
        "id",
        "product_family_group",
        "actual_usage",
        "refit_prediction",
    ]
    provenance = json.loads((folder / "evaluation_provenance.json").read_text())
    for name, expected in provenance["outputs"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected


def test_current_evaluation_metadata_has_no_retrospective_selection_flags():
    paths = [
        "gender_mixup_refit_holdout_20260911/evaluation.json",
        "gender_mixup_refit_holdout_20260911/evaluation_plan.json",
        "usage_e8_refit_holdout_20260907/evaluation.json",
        "usage_e8_refit_holdout_20260907/prediction_freeze.json",
        "usage_final_e8_refit_20260907/model_manifest.json",
        "usage_final_e1_20260907/model_manifest.json",
        "gender_final_sam25_refit_20260907/model_manifest.json",
    ]
    for relative in paths:
        text = (ROOT / "reports/task3" / relative).read_text()
        for obsolete in [
            "new_blind_evaluation",
            "new_blind_selection",
            "after_holdout_and_test_review",
            "acceptance_timing",
            "already opened in prior project work",
        ]:
            assert obsolete not in text, relative


def test_main_has_no_holdout_mentions_and_final_has_no_other_gender_model():
    main = nbformat.read(ROOT / "notebooks/06_task3_part1_gender_usage.ipynb", as_version=4)
    final = nbformat.read(ROOT / "notebooks/07_task3_part2_final_evaluation.ipynb", as_version=4)
    # Check text outputs too, excluding base64 images where arbitrary substrings can occur.
    main_text = "\n".join(c.source for c in main.cells)
    for cell in main.cells:
        for output in cell.get("outputs", []):
            main_text += str(output.get("text", ""))
            main_text += str(output.get("data", {}).get("text/plain", ""))
            main_text += str(output.get("data", {}).get("text/html", ""))
    assert "holdout" not in main_text.lower()
    final_source = "\n".join(c.source for c in final.cells)
    assert "SAM25" not in final_source and "sam25" not in final_source
    assert "load_selected_evaluation" in final_source
    assert "78.74%" in final_source and "42.26%" in final_source
    assert "The development study fixes the model and prediction rule" in final_source
    assert [
        int(c.source.split(".")[0][3:]) for c in final.cells if c.source.startswith("## ")
    ] == list(range(1, 17))
    for notebook in [main, final]:
        assert not any(
            o.output_type == "error" for c in notebook.cells for o in c.get("outputs", [])
        )
