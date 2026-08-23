from __future__ import annotations

import re
from pathlib import Path

import nbformat

from fashion.config import ROOT


TASK_SPECS = {
    "02_task1_article_type.ipynb": {
        "title": "Task 1 — Article Type Classification",
        "tokens": ("articleType", "long tail", "rare-class"),
    },
    "03_task2_season.ipynb": {
        "title": "Task 2 — Season Classification",
        "tokens": ("season", "article-type shortcuts", "calibration"),
    },
    "04_task3_gender_usage.ipynb": {
        "title": "Task 3 — Gender and Usage Classification",
        "tokens": ("gender", "usage", "negative transfer", "label masks"),
    },
    "05_task4_visual_search.ipynb": {
        "title": "Task 4 — Fashion Visual Search",
        "tokens": ("query", "gallery", "variant fusion", "metadata proxy"),
    },
    "06_final_evaluation.ipynb": {
        "title": "Final Evaluation and Ultimate Judgement",
        "tokens": ("holdout", "unlock exactly once", "official predictions", "ultimate judgement"),
    },
}


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def test_only_planned_notebook_names_are_present() -> None:
    allowed = {"00_problem_definition.ipynb", "01_data_preparation.ipynb", *TASK_SPECS}
    present = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert {"00_problem_definition.ipynb", "01_data_preparation.ipynb"} <= present
    assert present <= allowed


def test_existing_task_scaffolds_keep_decisions_with_teammates() -> None:
    for filename, spec in TASK_SPECS.items():
        path = ROOT / "notebooks" / filename
        if not path.exists():
            continue

        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        source = _source(notebook)
        lowered = source.lower()
        headings = [
            line
            for cell in notebook.cells
            for line in cell.source.splitlines()
            if line.startswith("#")
        ]

        assert notebook.metadata["title"] == spec["title"]
        assert notebook.cells
        assert all(cell.cell_type == "markdown" for cell in notebook.cells)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        assert headings[0] == f"# {spec['title']}"
        assert "## Scaffold status and owner handoff" in headings
        assert [
            int(match.group(1))
            for heading in headings
            if (match := re.fullmatch(r"## (\d+)\. .+", heading))
        ] == list(range(1, 10))

        for required in (
            "Planning scaffold only. No completed model claim.",
            "TODO(owner)",
            "data/processed/splits.csv",
            "results/runs.csv",
            "trained from scratch",
        ):
            assert required in source
        assert all(token.lower() in lowered for token in spec["tokens"])
        assert "train_test_split" not in source
        assert "nDCG@" not in source
        assert "Recall@" not in source
        assert "macro-F1" not in source
        assert "Final metric selected: yes" not in source


def test_task_metric_placeholders_are_explicit_when_scaffolds_exist() -> None:
    for filename in TASK_SPECS:
        path = ROOT / "notebooks" / filename
        if not path.exists():
            continue
        source = _source(nbformat.read(path, as_version=4))
        if filename == "04_task3_gender_usage.ipynb":
            assert source.count("Primary development metric | `TODO(owner)`") == 1
            assert "| `TODO(owner)` | `TODO(owner)` |" in source
        elif filename == "05_task4_visual_search.ipynb":
            assert "Primary ranking-quality metric: `TODO(owner)`" in source
            assert "Cutoff, averaging, zero-positive rule, and tie-break: `TODO(owner)`" in source
        elif filename != "06_final_evaluation.ipynb":
            assert "Primary development metric: `TODO(owner)`" in source
            assert "No exact metric is selected by this scaffold." in source
