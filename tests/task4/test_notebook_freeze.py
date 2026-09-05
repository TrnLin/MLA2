from __future__ import annotations

import nbformat

from fashion.config import ROOT

NOTEBOOK = ROOT / "notebooks/task-4/05_task4_visual_search.ipynb"
ADR = ROOT / "docs/decisions/0023-task4-learned-model-comparison.md"


def test_task4_notebook_loads_the_strict_repository_freeze() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    freeze_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "# task4-final-freeze" in cell.source
    ]
    assert len(freeze_cells) == 1

    namespace: dict[str, object] = {"TASK4_FREEZE_ROOT": ROOT}
    exec(freeze_cells[0].source, namespace)
    frozen = namespace["task4_freeze"]

    assert frozen["decision"]["method"] == "R5"
    assert frozen["decision"]["gallery_policy"] == "teacher"
    assert len(frozen["methods"]) == 6
    assert sum(len(item["folds"]) for item in frozen["stability"]) == 10
    assert frozen["safety"] == {
        "development_only": True,
        "holdout_opened": False,
        "quarantine_opened": False,
        "official_teacher_test_opened": False,
    }


def test_task4_notebook_has_no_split_or_protected_data_escape() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ).lower()

    assert "train_test_split" not in code_source
    assert "load_splits_for_final_evaluation" not in code_source
    assert "data/raw/teacher/test" not in code_source
    assert "/quarantine" not in code_source
    assert "quarantine/" not in code_source


def test_task4_notebook_describes_r4_and_independent_r5_correctly() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "R4 added batch-hard product-family triplet loss to R3." in markdown
    assert (
        "R5 was an independent scratch convolutional autoencoder with a 128-value "
        "bottleneck"
    ) in markdown


def test_task4_adr_describes_r1_through_r4_as_incremental_candidates() -> None:
    adr = ADR.read_text()

    assert "R1–R4: incremental CNN/VICReg candidates" in adr
    assert "R1–R4: incremental CNN/VICReg and pooling choices" not in adr


def test_task4_notebook_states_final_bundle_validation_boundary() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "The notebook validates the compact final bundle." in markdown
    assert (
        "The bundle producer validated and hashed the full source artifacts." in markdown
    )
    assert "reads those files directly and hashes every source it uses" not in markdown
