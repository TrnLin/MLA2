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
        "sections": 16,
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
    allowed = {
        "00_problem_definition.ipynb",
        "01_data_preparation.ipynb",
        "04a_task3_smallcnn_baseline_training.ipynb",
        "04b_task3_smallcnn_child_experiments.ipynb",
        "04c_task3_smallcnn_e3_experiments.ipynb",
        *TASK_SPECS,
    }
    present = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert present == allowed


def test_task3_baseline_training_runner_is_foreground_and_complete() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/04a_task3_smallcnn_baseline_training.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — SmallCNN Baseline Training"
    assert source.count("folds=range(5)") == 2
    assert 'run_task3_baseline_cv(\n    "gender"' in source
    assert 'run_task3_baseline_cv(\n    "usage"' in source
    assert "registry_path=DRIVE_REGISTRY" in source
    assert "output_root=DRIVE_TASK_DIR" in source
    assert "nohup" not in code
    assert "START_BASELINE_TRAINING" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")


def test_task3_child_runner_never_retrains_the_baseline() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/04b_task3_smallcnn_child_experiments.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — SmallCNN Child Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert '"gender_brightness"' in source
    assert '"usage_class_balanced"' in source
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "run_task3_baseline_cv" not in source
    assert "START_BASELINE_TRAINING" not in code
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e3_runner_never_retrains_e1_or_e2() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/04c_task3_smallcnn_e3_experiments.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — SmallCNN E3 Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert '"gender_class_balanced"' in source
    assert '"usage_classifier_dropout"' in source
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert "run_task3_baseline_cv" not in source
    assert 'run_task3_child_cv(\n    "gender_brightness"' not in source
    assert 'run_task3_child_cv(\n    "usage_class_balanced"' not in source
    assert "START_BASELINE_TRAINING" not in code
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_task_scaffolds_leave_owner_decisions_open() -> None:
    for filename, spec in TASK_SPECS.items():
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
        if filename == "04_task3_gender_usage.ipynb":
            code_ids = {cell.id for cell in notebook.cells if cell.cell_type == "code"}
            assert code_ids == {
                "t3-colab-config",
                "t3-colab-repository",
                "t3-colab-data",
                "t3-colab-load",
                "t3-baseline-model",
                "t3-baseline-results-load",
                "t3-baseline-tables",
                "t3-baseline-curves",
                "t3-baseline-diagnostics",
                "t3-baseline-failure-gallery",
                "t3-child-results-load",
                "t3-child-comparison",
                "t3-child-curves",
                "t3-child-diagnostics",
                "t3-child-failure-gallery",
            }
            assert all(
                cell.cell_type == "markdown" or cell.id in code_ids for cell in notebook.cells
            )
        else:
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
        if filename != "04_task3_gender_usage.ipynb":
            for unselected in ("macro-F1", "nDCG@", "Recall@", "Adam", "cross-entropy"):
                assert unselected not in source


def test_task_metric_contracts_are_explicit() -> None:
    for filename in ("02_task1_article_type.ipynb", "03_task2_season.ipynb"):
        source = _source(nbformat.read(ROOT / "notebooks" / filename, as_version=4))
        assert "Primary development metric: TODO(owner)" in source

    task3 = _source(nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4))
    assert "Primary development metric for `gender`: pooled five-fold OOF macro-F1" in task3
    assert "Primary development metric for `usage`: pooled five-fold OOF macro-F1" in task3
    assert task3.count("| Written before training | Yes |") == 2
    assert "Gender E2 — reject brightness augmentation" in task3
    assert "Usage E2 — accept class-balanced cross-entropy" in task3
    assert "Effective-number class-balanced cross-entropy only" in task3
    assert "Dropout `p=0.20` after global pooling" in task3
    assert "fixed 10,000-sample product-family bootstrap" in task3
    assert "mean Boys/Girls/Unisex F1 ≥ 0.6048" in task3
    assert "macro-F1 without Home ≥ 0.4692" in task3
    assert "Fold macro-F1 SD ≤ 0.0303" in task3
    assert "scratch low-resolution ResNet-18" in task3
    assert "This table grows one approved row at a time" in task3
    assert "The EDA justifies a small native-resolution CNN" in task3
    assert "does **not** prove that `32,64,128,256` is optimal" in task3
    assert "Task3BaselineCNN" in task3
    assert "load_target_evidence" in task3
    assert task3.count("keep_default_na=False") == 6
    assert "Loaded completed E2 evidence for gender and usage from Drive" in task3
    assert "e2_learning_curves.png" in task3
    assert "e2_per_class_f1_comparison.png" in task3
    assert "e2_normalised_confusion.png" in task3
    assert "e2_robustness_comparison.png" in task3
    assert "e2_{target}_high_confidence_errors.png" in task3
    assert "0.7118" in task3
    assert "0.3738" in task3
    assert "0.6989 pooled macro-F1" in task3
    assert "0.4082 pooled macro-F1" in task3
    assert "04a_task3_smallcnn_baseline_training.ipynb` reproduces only" in task3
    assert "04b_task3_smallcnn_child_experiments.ipynb` loads those saved parents" in task3
    assert "04c_task3_smallcnn_e3_experiments.ipynb` loads the accepted E1/E2 parents" in task3
    assert "run_task3_baseline_cv" not in task3

    notebook = nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4)
    child_cells = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.id.startswith("t3-child-")
    )
    assert "DRIVE_TASK_DIR" in child_cells
    assert "google.colab" not in child_cells
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )

    task4 = _source(nbformat.read(ROOT / "notebooks/05_task4_visual_search.ipynb", as_version=4))
    assert "Primary ranking-quality metric: TODO(owner)" in task4
    assert "Cutoff, averaging, zero-positive rule, and tie-break: TODO(owner)" in task4
