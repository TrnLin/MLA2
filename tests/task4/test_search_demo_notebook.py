from __future__ import annotations

import nbformat
import pytest

from fashion.config import ROOT

NOTEBOOK = ROOT / "notebooks/task-4/07_task4_search_demo.ipynb"


@pytest.mark.parametrize("outside", [False, True])
def test_search_demo_passes_selected_crop_to_both_query_modes(outside: bool) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    dispatch = next(
        cell.source for cell in notebook.cells
        if cell.cell_type == "code" and "bundle = load_search_bundle(" in cell.source
    )
    crop = (1, 2, 10, 12)
    calls = []
    namespace = {
        "ROOT": ROOT,
        "load_search_bundle": lambda **kwargs: object(),
        "run_search": lambda bundle, **kwargs: calls.append(kwargs),
        "KNOWN_QUERY_ID": None if outside else 1529,
        "OUTSIDE_IMAGE": ROOT / "outside.png" if outside else None,
        "CROP": crop,
        "TOP_K": 5,
        "RATING": None,
        "NOTE": None,
    }
    exec(dispatch, namespace)
    assert len(calls) == 1
    assert calls[0]["crop"] == crop
    key = "image_path" if outside else "query_id"
    assert calls[0][key] == namespace["OUTSIDE_IMAGE" if outside else "KNOWN_QUERY_ID"]


def test_search_demo_uses_public_output_writer() -> None:
    from fashion import task4

    assert callable(task4.write_search_outputs)


def test_search_demo_notebook_is_thin_safe_and_runnable() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert [cell.cell_type for cell in notebook.cells] == [
        "markdown",
        "code",
        "markdown",
        "code",
        "code",
        "code",
        "code",
        "code",
    ]

    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "Task 4 — Search Test Platform" in markdown
    assert "holdout remains sealed" in markdown.lower()
    assert "KNOWN_QUERY_ID = 1529" in code
    assert "OUTSIDE_IMAGE: Path | None = None" in code
    assert "load_search_bundle" in code
    assert "run_search" in code
    assert "write_search_outputs" in code
    assert "load_splits_for_final_evaluation" not in code
    assert "train_test_split" not in code
    assert "results/cache/task4" not in code
    assert ".read_csv(" not in code
    assert ".load_state_dict(" not in code
    assert "import torch" not in code
    assert "import numpy" not in code
    assert "scripts.task4" not in code


def test_search_demo_notebook_is_saved_clean() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert len(notebook.cells) == 8
    assert len({cell.id for cell in notebook.cells}) == 8
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
