from __future__ import annotations

import json
import re

import nbformat
import pandas as pd

from fashion.config import ROOT

TASK_SPECS = {
    "02_task1_article_type.ipynb": {
        "title": "Task 1 — Article Type Classification",
        "tokens": ("articleType", "long-tail taxonomy", "rare-class error"),
        "sections": 15,
    },
    "03_task2_season.ipynb": {
        "title": "Task 2 — Season Classification",
        "tokens": ("weak-visual-signal", "article-type shortcut", "calibration"),
        "sections": 15,
    },
    "04_task3_gender_usage.ipynb": {
        "title": "Task 3 — Gender and Usage Classification",
        "tokens": ("gender", "usage", "negative transfer", "label-mask"),
        "sections": 15,
    },
    "05_task4_visual_search.ipynb": {
        "title": "Task 4 — Fashion Visual Search",
        "tokens": ("arbitrary query size", "optional additional image", "embedding", "Top-K"),
        "sections": 15,
    },
    "06_final_evaluation.ipynb": {
        "title": "Final Evaluation and Ultimate Judgement",
        "tokens": ("holdout", "opened once", "official predictions", "ultimate judgement"),
        "sections": 13,
    },
}
TASK_PATHS = {
    "05_task4_visual_search.ipynb": ROOT / "notebooks/task-4/05_task4_visual_search.ipynb"
}


def _task_path(filename: str):
    return TASK_PATHS.get(filename, ROOT / "notebooks" / filename)


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def test_only_planned_notebook_names_are_present() -> None:
    allowed = {
        "00_problem_definition.ipynb",
        "01_data_preparation.ipynb",
        *(name for name in TASK_SPECS if name != "05_task4_visual_search.ipynb"),
    }
    present = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert present == allowed
    task4_present = {path.name for path in (ROOT / "notebooks/task-4").glob("*.ipynb")}
    assert task4_present == {"01_v1_eda.ipynb", "05_task4_visual_search.ipynb"}


def test_task_notebooks_preserve_common_structure_and_safety() -> None:
    for filename, spec in TASK_SPECS.items():
        notebook = nbformat.read(_task_path(filename), as_version=4)
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
        assert headings[0] == f"# {spec['title']}"
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        assert [
            int(match.group(1))
            for heading in headings
            if (match := re.fullmatch(r"## (\d+)\. .+", heading))
        ] == list(range(1, spec["sections"] + 1))

        for required in ("data/processed/splits.csv", "results/runs.csv"):
            assert required in source
        assert all(token.lower() in lowered for token in spec["tokens"])
        assert "train_test_split" not in source
        assert "pretrained=True" not in source


def test_task_1_to_3_scaffolds_leave_owner_decisions_open() -> None:
    for filename in (
        "02_task1_article_type.ipynb",
        "03_task2_season.ipynb",
        "04_task3_gender_usage.ipynb",
    ):
        notebook = nbformat.read(_task_path(filename), as_version=4)
        source = _source(notebook)

        assert all(
            cell.cell_type == "markdown" or not cell.source.strip() for cell in notebook.cells
        )
        assert "TODO(owner)" in source
        assert "Final metric selected: yes" not in source
        for unselected in ("macro-F1", "nDCG@", "Recall@", "Adam", "cross-entropy"):
            assert unselected not in source


def test_task_metric_placeholders_are_explicit() -> None:
    for filename in ("02_task1_article_type.ipynb", "03_task2_season.ipynb"):
        source = _source(nbformat.read(ROOT / "notebooks" / filename, as_version=4))
        assert "Primary development metric: TODO(owner)" in source

    task3 = _source(nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4))
    assert "Primary development metric for `gender`: TODO(owner)" in task3
    assert "Primary development metric for `usage`: TODO(owner)" in task3

    task4 = _source(nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4))
    assert "Primary ranking-quality metric: mean per-query linear nDCG@10" in task4


def test_task4_evaluation_protocol_is_frozen_and_executed() -> None:
    notebook = nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    source = _source(notebook)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()
    ]

    for required in (
        "fold `1`",
        "nDCG@10",
        "Recall@10",
        "same `articleType` and `baseColour`",
        "results/evidence/task4/retrieval_protocol_coverage.csv",
        "### Evaluation protocol overview",
        "coverage_summary",
        "retrieval_protocol_overview.png",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
    ):
        assert required in source
    assert "Primary ranking-quality metric: TODO(owner)" not in source
    assert len(code_cells) >= 2
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(
        not cell.get("outputs")
        or not any(
            output.get("output_type") == "error" for output in cell["outputs"]
        )
        for cell in code_cells
    )
    saved_stderr = "\n".join(
        output.get("text", "")
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stderr"
    )
    assert "Fontconfig error" not in saved_stderr
    assert "Matplotlib created a temporary cache" not in saved_stderr
    assert "load_splits_for_final_evaluation" not in source
    assert (ROOT / "results/figures/task4/retrieval_protocol_overview.png").exists()


def test_task4_preprocessing_milestone_is_frozen_and_executed() -> None:
    notebook = nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    source = _source(notebook)
    preprocessing = source.split(
        "## 5. Task-specific preprocessing and leakage rules", maxsplit=1
    )[1].split("## 7. Hypotheses and baseline", maxsplit=1)[0]

    for required in (
        "Frozen input contract: `240×320`",
        "60×80",
        "96×128",
        "240×320",
        "Teacher → teacher",
        "V1 → V1",
        "Teacher → V1",
        "V1 → teacher",
        "LANCZOS",
        "4×4",
        "training folds only",
        "never refit",
        "preprocessing_size_selection.csv",
        "preprocessing_stability.csv",
        "preprocessing_robustness.csv",
        "preprocessing_comparison.png",
        "does not prove learned-model quality",
        "probe winner remained `96×128`",
    ):
        assert required in preprocessing
    assert "TODO(owner)" not in preprocessing
    assert "load_splits_for_final_evaluation" not in source
    for settled in (
        "Image-size strategy and any comparison of multiple sizes: TODO(owner)",
        "Behaviour for an arbitrary query size/aspect ratio: TODO(owner)",
        "frozen preprocessing configuration: TODO(owner)",
        "complete fixed-fold and top-two five-fold stability evidence: TODO(owner after runs)",
    ):
        assert settled not in source

    for path in (
        ROOT / "results/evidence/task4/preprocessing_comparison.csv",
        ROOT / "results/evidence/task4/preprocessing_size_selection.csv",
        ROOT / "results/evidence/task4/preprocessing_stability.csv",
        ROOT / "results/evidence/task4/preprocessing_robustness.csv",
        ROOT / "results/evidence/task4/preprocessing_contract.json",
        ROOT / "results/evidence/task4/preprocessing_normalization_fold1.json",
        ROOT / "results/figures/task4/preprocessing_comparison.png",
    ):
        assert path.exists()


def test_task4_baseline_milestone_is_frozen_and_executed() -> None:
    notebook = nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    source = _source(notebook)
    start = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.source.startswith("## 7. Hypotheses and baseline")
    )
    end = next(
        index
        for index, cell in enumerate(notebook.cells[start + 1 :], start=start + 1)
        if cell.source.startswith("## 8. Candidate model comparisons")
    )
    baseline_cells = notebook.cells[start:end]
    baseline = "\n".join(cell.source for cell in baseline_cells)
    baseline_code = [
        cell for cell in baseline_cells if cell.cell_type == "code" and cell.source.strip()
    ]

    for required in (
        "spatial-hsv-edge-4x4-v2",
        "240×320",
        "Hypothesis 1",
        "Hypothesis 2",
        "PASS",
        "FAIL",
        "reject",
        "Teacher → teacher",
        "V1 → V1",
        "Teacher → V1",
        "V1 → teacher",
        "baseline_summary.csv",
        "baseline_examples.png",
    ):
        assert required in baseline
    assert "TODO(owner)" not in baseline
    assert len(baseline_code) == 2
    assert all(cell.execution_count is not None for cell in baseline_code)
    assert all(
        not any(output.get("output_type") == "error" for output in cell.get("outputs", []))
        for cell in baseline_code
    )

    for required in (
        "baseline_failure_slices.csv",
        "baseline_cost.json",
        "p50",
        "p95",
        "p95 end-to-end latency < 1 second: PASS",
        "index size < 1 GiB: PASS",
    ):
        assert required in source


def test_task4_preprocessing_artifacts_are_development_only_and_complete() -> None:
    evidence = ROOT / "results/evidence/task4"
    comparison = pd.read_csv(evidence / "preprocessing_comparison.csv")
    selection = pd.read_csv(evidence / "preprocessing_size_selection.csv")
    stability = pd.read_csv(evidence / "preprocessing_stability.csv")
    robustness = pd.read_csv(evidence / "preprocessing_robustness.csv")

    assert set(comparison["scope"]) == {"development"}
    assert {
        "query_transform_seconds",
        "uint8_tensor_bytes_per_image",
        "float32_tensor_bytes_per_image",
    }.issubset(comparison.columns)
    assert set(comparison.loc[comparison["fold"].eq(1), "size"]) == {
        "60x80",
        "96x128",
        "240x320",
    }
    assert set(
        zip(
            comparison.loc[comparison["fold"].eq(1), "query_source"],
            comparison.loc[comparison["fold"].eq(1), "gallery_source"],
            strict=False,
        )
    ) == {
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    }
    assert set(selection["scope"]) == {"development"}
    primary_selection_rows = comparison.loc[
        comparison["fold"].eq(1)
        & comparison["protocol"].eq("primary")
        & comparison["metric"].eq("ndcg")
        & comparison["k"].eq(10)
        & comparison["aggregation"].eq("query_mean")
        & comparison["query_source"].eq(comparison["gallery_source"])
    ]
    recomputed = (
        primary_selection_rows.groupby("size")["value"].mean().sort_values(ascending=False)
    )
    recorded = selection.set_index("size")["selection_ndcg_at_10"]
    pd.testing.assert_series_equal(
        recomputed.sort_index(),
        recorded.sort_index(),
        check_names=False,
    )
    assert selection.sort_values("selection_rank").iloc[0]["size"] == "96x128"
    assert set(stability["scope"]) == {"development"}
    assert stability.groupby("size")["fold"].nunique().to_dict() == {
        "60x80": 5,
        "96x128": 5,
    }
    assert set(robustness["scope"]) == {"development"}
    assert set(robustness["query_variant"]) == {"clean", "wide", "tall"}
    contract = json.loads(
        (evidence / "preprocessing_contract.json").read_text(encoding="utf-8")
    )
    assert contract["selected_size"] == "240x320"
    assert contract["probe_winner_size"] == "96x128"
    assert contract["selected_probe_rank"] == 3

    notebook_source = _source(
        nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    )
    assert "images_per_second" in notebook_source
    assert "float32_rgb_kib_per_image" in notebook_source


def test_task4_v1_eda_is_separate_safe_and_executed() -> None:
    notebook = nbformat.read(ROOT / "notebooks/task-4/01_v1_eda.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = _source(notebook)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()
    ]

    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    for required in (
        "V1 High-Resolution Image EDA",
        "data/processed/splits.csv",
        "never a new split",
        "development rows only",
        "results/evidence/task4/",
    ):
        assert required in source
    for forbidden in ("train_test_split", "load_splits_for_final_evaluation", "styles.csv"):
        assert forbidden not in source
