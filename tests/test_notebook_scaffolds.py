from __future__ import annotations

import re

import nbformat

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


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def test_only_planned_notebook_names_are_present() -> None:
    allowed = {"00_problem_definition.ipynb", "01_data_preparation.ipynb", *TASK_SPECS}
    present = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert present == allowed


def test_task_scaffolds_leave_owner_decisions_open() -> None:
    for filename, spec in TASK_SPECS.items():
        if filename == "03_task2_season.ipynb":
            continue
        notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
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
        assert all(cell.cell_type == "markdown" for cell in notebook.cells)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        assert [
            int(match.group(1))
            for heading in headings
            if (match := re.fullmatch(r"## (\d+)\. .+", heading))
        ] == list(range(1, spec["sections"] + 1))

        for required in ("TODO(owner)", "data/processed/splits.csv", "results/runs.csv"):
            assert required in source
        assert all(token.lower() in lowered for token in spec["tokens"])
        assert "train_test_split" not in source
        assert "pretrained=True" not in source
        assert "Final metric selected: yes" not in source
        for unselected in ("macro-F1", "nDCG@", "Recall@", "Adam", "cross-entropy"):
            assert unselected not in source


def _heading_level(cell: nbformat.NotebookNode) -> int | None:
    if cell.cell_type != "markdown":
        return None
    lines = cell.source.splitlines()
    first_line = lines[0] if lines else ""
    match = re.match(r"^(#{1,4}) ", first_line)
    return len(match.group(1)) if match else None


def test_task2_execution_scaffold_has_one_code_cell_per_leaf() -> None:
    path = ROOT / "notebooks/03_task2_season.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = _source(notebook)
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]

    assert notebook.metadata["title"] == "Task 2 — Season Classification"
    assert headings[0] == "# Task 2 — Season Classification"
    assert len(notebook.cells) == 193
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert [
        int(match.group(1))
        for heading in headings
        if (match := re.fullmatch(r"## (\d+)\. .+", heading))
    ] == list(range(1, 16))

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.source.strip() for cell in code_cells)
    placeholder_cells = [cell for cell in code_cells if cell.source.startswith("# TODO:")]
    assert all("# Expected output:" in cell.source for cell in placeholder_cells)
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in placeholder_cells
    )

    h3_indices = [
        index
        for index, cell in enumerate(notebook.cells)
        if _heading_level(cell) == 3
    ]
    h4_indices = [
        index
        for index, cell in enumerate(notebook.cells)
        if _heading_level(cell) == 4
    ]
    assert len(h3_indices) == 40
    assert len(h4_indices) == 27
    assert len(code_cells) == 55

    grouped_h3_count = 0
    for index in h3_indices:
        next_level = _heading_level(notebook.cells[index + 1])
        if next_level == 4:
            grouped_h3_count += 1
            continue
        assert notebook.cells[index + 1].cell_type == "code"
        assert notebook.cells[index + 2].cell_type == "markdown"
        assert notebook.cells[index + 2].source.lstrip().startswith("> **Interpretation")

    for index in h4_indices:
        assert notebook.cells[index + 1].cell_type == "code"
        assert notebook.cells[index + 2].cell_type == "markdown"
        assert notebook.cells[index + 2].source.lstrip().startswith("> **Interpretation")

    leaf_h3_count = len(h3_indices) - grouped_h3_count
    assert len(code_cells) == leaf_h3_count + len(h4_indices)

    for required in (
        "data/processed/splits.csv",
        "results/runs.csv",
        "weak-visual-signal",
        "article-type shortcut",
        "calibration",
        "pooled out-of-fold macro-F1",
        "weights=None",
        "How Task 2 files affect one another",
        "build_file_impact_edges",
        "file_impact_flow.png",
    ):
        assert required.lower() in source.lower()
    assert "TODO(owner)" not in source
    assert "train_test_split" not in source
    assert "pretrained=True" not in source


def test_task2_data_protocol_cells_are_executable_orchestration() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    protocol_cell_ids = (
        "s01-01-code",
        "s01-02-code",
        "s02-01-code",
        "s02-02-01-code",
        "s02-02-02-code",
        "s03-01-code",
        "s03-02-code",
        "s03-03-code",
        "s04-01-code",
        "s04-02-code",
        "s04-03-code",
    )

    for cell_id in protocol_cell_ids:
        cell = cells[cell_id]
        assert cell.cell_type == "code"
        assert not cell.source.startswith("# TODO:")
        compile(cell.source, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in protocol_cell_ids)
    for required in (
        "load_splits()",
        "has_season_label",
        "iter_cv_folds(splits)",
        "build_task_loaders(",
        "validate_oof",
        "protected labels remain sealed",
        "atomic_write_json",
        "atomic_write_csv",
    ):
        assert required in combined


def test_task2_environment_table_uses_dataframe_compatible_reset_index() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    environment_cell = next(cell for cell in notebook.cells if cell.id == "s01-02-code")

    assert ".reset_index(name=" not in environment_cell.source


def test_task2_preprocessing_mask_uses_an_informative_padding_example() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    preprocessing_cell = next(cell for cell in notebook.cells if cell.id == "s04-01-code")

    assert "nonstandard_aspect" in preprocessing_cell.source
    assert "first_content_mask.any() and (~first_content_mask).any()" in preprocessing_cell.source
    assert 'cmap="gray", vmin=0, vmax=1' in preprocessing_cell.source


def test_task2_model_and_registry_cells_use_shared_interfaces() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    wired_cell_ids = (
        "s06-01-01-code",
        "s06-01-02-code",
        "s06-01-03-code",
        "s06-02-code",
        "s06-03-code",
        "s07-02-code",
    )

    for cell_id in wired_cell_ids:
        cell = cells[cell_id]
        assert cell.cell_type == "code"
        assert not cell.source.startswith("# TODO:")
        compile(cell.source, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in wired_cell_ids)
    for required in (
        "build_season_model(",
        "assert_final_model(",
        "weights",
        "benchmark_only",
        "state_dict_sha256",
        "RunRegistry(RUNS_CSV)",
        "RUN_COLUMNS",
        "scratch_model_audit.csv",
        "benchmark_boundary.csv",
    ):
        assert required in combined


def test_task2_g0_cell_records_a_non_comparison_pass() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s07-01-code"].source
    finding = cells["s07-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s07-01-code", "exec")
    for required in (
        "g0_pipeline_smoke.json",
        "run_or_load_g0_smoke",
        "build_g0_evidence",
        "integration_macro_f1_non_comparison",
    ):
        assert required in code
    assert "passed" in finding
    assert "excluded from every leaderboard" in finding
    assert "g0-pipeline-smoke-f0-s2753-a5a6902f6fd0" in finding


def test_task2_b0_cell_records_complete_five_fold_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s05-01-code"].source
    finding = cells["s05-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s05-01-code", "exec")
    for required in (
        "b0_majority.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "protected_ids",
        'b0_manifest["run_ids"]',
    ):
        assert required in code
    assert "32,753" in finding
    assert "0.165704" in finding
    assert "Spring" in finding
    assert "b0-majority-f0-s2753-8200c51534e3" in finding
    assert "results/evidence/task2/b0_majority/manifest.json" in finding


def test_task2_b1_cell_records_uncalibrated_five_fold_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s05-02-code"].source
    finding = cells["s05-02-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s05-02-code", "exec")
    for required in (
        "b1_hog_hsv_svm.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "calibration_claim_allowed=False",
        "macro_f1_gain_over_B0",
    ):
        assert required in code
    assert "32,753" in finding
    assert "0.609561" in finding
    assert "0.486901" in finding
    assert "not calibrated probabilities" in finding
    assert "b1-hog-hsv-svm-f0-s2753-b23d1eafe35f" in finding
    assert "results/evidence/task2/b1_hog_hsv_svm/manifest.json" in finding


def test_task_metric_placeholders_are_explicit() -> None:
    task1 = _source(nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4))
    assert "Primary development metric: TODO(owner)" in task1

    task2 = _source(nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4))
    assert "Primary development metric:** pooled out-of-fold macro-F1" in task2

    task3 = _source(nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4))
    assert "Primary development metric for `gender`: TODO(owner)" in task3
    assert "Primary development metric for `usage`: TODO(owner)" in task3

    task4 = _source(nbformat.read(ROOT / "notebooks/05_task4_visual_search.ipynb", as_version=4))
    assert "Primary ranking-quality metric: TODO(owner)" in task4
    assert "Cutoff, averaging, zero-positive rule, and tie-break: TODO(owner)" in task4
