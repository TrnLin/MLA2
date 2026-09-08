"""Academic-source contracts for the Task 2 training and evaluation notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING_NOTEBOOK = ROOT / "notebooks/03_task2_season.ipynb"
EVALUATION_NOTEBOOK = ROOT / "notebooks/06_task2_season_evaluation.ipynb"
EVALUATION_HTML = ROOT / "results/notebooks/06_task2_season_evaluation.html"

TRAINING_FOUNDATIONS = (
    "Dalal-cvpr05.pdf",
    "10.1007/BF00994018",
    "10.1109/5.726791",
    "He_Deep_Residual_Learning_CVPR_2016_paper.html",
    "Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html",
    "10.1186/s40537-019-0197-0",
    "openreview.net/forum?id=Bkg6RiCqY7",
    "Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples",
    "10.1023/A:1007379606734",
    "10.1109/CVPR.2009.5206848",
)

EVALUATION_FOUNDATIONS = (
    "10.1016/j.ipm.2009.03.002",
    "grandvalet04a.html",
    "guo17a.html",
    "10.1111/j.1467-9868.2007.00593.x",
    "4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html",
    "1903.12261",
    "Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html",
    "294a8ed24b1ad22ec2e7efea049b8737-Abstract.html",
    "8558cb408c1d76621371888657d2eb1d-Abstract.html",
    "koh21a.html",
)


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_task2_training_notebook_maps_training_and_evaluation_sources() -> None:
    source = _notebook_source(TRAINING_NOTEBOOK)

    for token in (*TRAINING_FOUNDATIONS, *EVALUATION_FOUNDATIONS):
        assert token in source
    assert '"training",' in source
    assert '"evaluation",' in source
    assert "Training and evaluation method source map" in source


def test_task2_evaluation_repeats_only_relevant_training_sources() -> None:
    source = _notebook_source(EVALUATION_NOTEBOOK)

    for token in (
        *TRAINING_FOUNDATIONS[:4],
        TRAINING_FOUNDATIONS[5],
        TRAINING_FOUNDATIONS[7],
        TRAINING_FOUNDATIONS[8],
        TRAINING_FOUNDATIONS[9],
        *EVALUATION_FOUNDATIONS,
    ):
        assert token in source
    assert '"Training (repeated)"' in source
    assert '"Evaluation"' in source
    assert TRAINING_FOUNDATIONS[4] not in source
    assert TRAINING_FOUNDATIONS[6] not in source
    assert "2007.00602" not in source


def test_task2_evaluation_html_exports_the_correct_source_map() -> None:
    source = EVALUATION_HTML.read_text(encoding="utf-8")

    assert "Training (repeated)" in source
    assert "Evaluation" in source
    assert "2007.00593" in source
    assert "2007.00602" not in source


def test_task2_evaluation_explains_outputs_and_bootstrap_uncertainty() -> None:
    source = _notebook_source(EVALUATION_NOTEBOOK)

    assert "def show_table" in source
    assert "def show_figure" in source
    assert "How to read this table" in source
    assert "How to read this figure" in source
    assert "paired_group_bootstrap" in source
    assert "np.testing.assert_allclose" in source
    assert "actual 10,000 resampled values" in source
    assert "No p-value is reported" in source
