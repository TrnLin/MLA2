from __future__ import annotations

import ast
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
    assert "g0-pipeline-smoke-f0-s2753-5ad5ee9d433c" in finding
    assert "git_dirty=false" in finding


def test_task2_g0_tracked_evidence_uses_one_clean_run() -> None:
    evidence = ROOT / "results/evidence/task2/g0"
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    snapshot = pd.read_csv(evidence / "registry_snapshot.csv", dtype=str)
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    finding = {cell.id: cell for cell in notebook.cells}["s07-01-finding"].source

    assert len(snapshot) == 1
    assert snapshot.loc[0, "run_id"] == manifest["run_id"]
    assert snapshot.loc[0, "status"] == "completed"
    assert snapshot.loc[0, "git_dirty"] == "false"
    assert manifest["run_id"] in finding
    assert "git_dirty=true" not in finding


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


def test_task2_g1_cell_records_equal_budget_family_screen() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-01-01-code"].source
    finding = cells["s08-01-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-01-01-code", "exec")
    for required in (
        "g1_c1_smallcnn.json",
        "g1_c2_resnet18.json",
        "g1_c3_mobilenetv3.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "build_g1_family_screen_evidence",
        "calibration_claim_allowed=False",
        "protected_ids",
        "len(g1_trace_rows) == 15",
        "g1_family_screen",
    ):
        assert required in code
    for required in (
        "32,753",
        "0.707099",
        "0.699902",
        "0.638495",
        "shortlist C2",
        "Reject C3",
        "Run P0/P1 and A0/A1 on C2",
        "not yet calibrated",
        "g1-c1-smallcnn-f0-s2753-59e9743d3899",
        "g1-c2-resnet18-f0-s2753-b91662d47026",
        "g1-c3-mobilenetv3-f0-s2753-9e07fe2a3158",
        "results/evidence/task2/g1_family_screen/manifest.json",
    ):
        assert required in finding


def test_task2_g3_cell_records_audited_full_budget_comparison() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-01-02-code"].source
    finding = cells["s08-01-02-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-01-02-code", "exec")
    for required in (
        "g3_c1_t1_smallcnn.json",
        "g3_c2_t0_resnet18.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "build_g3_full_budget_evidence",
        "protected_ids",
        'status\"].eq(\"interrupted\")',
        'near_tie\"] is True',
        'ultimate_winner_frozen\"] is False',
        'artifacts\"][\"c1_learning_curves\"]',
        'artifacts\"][\"c2_learning_curves\"]',
    ):
        assert required in code
    for required in (
        "0.737661",
        "0.735036",
        "0.002626",
        "near-tie threshold",
        "teacher-style",
        "validation accuracy",
        "validation macro-F1",
        "Training accuracy is absent",
        "9.51×",
        "1.52×",
        "provisional reference",
        "not the ultimate winner",
        "C3 remains rejected",
        "interrupted outside Python",
        "g3-c1-t1-smallcnn-f2-s2753-2283b6495a44",
        "g3-c1-t1-smallcnn-f2-s2753-e46d771b6d56",
        "g3-c1-t1-smallcnn-f0-s2753-1ce4f9978b12",
        "g3-c2-t0-resnet18-f0-s2753-66ee7a85d5c6",
        "47442a2",
        "results/evidence/task2/g3_full_budget/manifest.json",
    ):
        assert required in finding


def test_task2_g3_cell_scopes_validation_to_the_current_run_ids() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    code = {cell.id: cell for cell in notebook.cells}["s08-01-02-code"].source

    assert "current_run_ids = {output.run_id for output in outputs}" in code
    assert 'attempts["run_id"].isin(current_run_ids)' in code
    assert 'len(current_attempts) == 5' in code
    assert 'g3_attempts["status"].eq("completed").sum() == 10' not in code
    assert 'g3_attempts["status"].eq("interrupted").sum() == 1' not in code


def test_task2_g3_notebook_preserves_registered_probability_note() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    code = {cell.id: cell for cell in notebook.cells}["s08-01-02-code"].source
    string_literals = {
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert (
        "Uncalibrated softmax probabilities from the best validation macro-F1 "
        "checkpoint in each fold; calibration metrics are diagnostic only."
        in string_literals
    )


def test_task2_i1_cell_records_audited_class_balance_decision() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-03-01-code"].source
    finding = cells["s08-03-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-03-01-code", "exec")
    for required in (
        "g4_i1_effective_number_c1.json",
        "run_or_load_i1_experiment",
        "build_experiment_evidence",
        "build_i1_class_balance_evidence",
        "protected_ids",
        "calibration_claim_allowed=False",
        "current_run_ids",
        'status"].eq("running")',
        'i1_manifest["gate"] == "G4-I1"',
        'i1_manifest["keep_i1"] is False',
        'i1_manifest["selected_experiment_id"] == "g3-c1-t1-smallcnn"',
        'artifacts"]["learning_curves"]',
        'artifacts"]["per_class_f1_delta"]',
    ):
        assert required in code
    for required in (
        "0.701471",
        "0.737661",
        "-0.036191",
        "0.702439",
        "0.744975",
        "-0.042536",
        "+0.022573",
        "-0.152558",
        "-0.060738",
        "reject I1",
        "loss values are not comparable",
        "g4-i1-effective-number-c1-f0-s2753-9288633e212b",
        "g4-i1-effective-number-c1-f4-s2753-a70502c769db",
        "results/evidence/task2/i1_class_balance/manifest.json",
    ):
        assert required in finding


def test_task2_i2_cell_records_audited_multitask_decision() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell = cells["s08-03-02-code"]
    code = cell.source
    finding = cells["s08-03-02-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-03-02-code", "exec")
    for required in (
        "g4_i2_article_type_lambda_0_1_c1.json",
        "g4_i2_article_type_lambda_0_3_c1.json",
        "run_i2_matrix",
        "build_experiment_evidence",
        "build_i2_transfer_evidence",
        "protected_ids",
        "calibration_claim_allowed=False",
        "current_run_ids",
        'status"].eq("running")',
        'i2_manifest["gate"] == "G4-I2"',
        'i2_manifest["keep_i2"] is True',
        '== "g4-i2-article-type-lambda-0-3-c1"',
        'artifacts"]["learning_curves"]',
        'artifacts"]["transfer_deltas"]',
    ):
        assert required in code
    assert cell.execution_count == 29
    assert len(cell.outputs) == 8
    assert all(output.output_type != "error" for output in cell.outputs)
    for required in (
        "B0",
        "B1",
        "C1/G3",
        "I1",
        "I2",
        "0.750758",
        "0.752687",
        "0.737661",
        "+0.015026",
        "0.764784",
        "+0.019809",
        "+0.013792",
        "+0.027598",
        "+0.031917",
        "training-only ArticleType head",
        "Notebook 01/data preparation is unchanged",
        "association, not proof",
        "keep I2 lambda `0.3`",
        "g4-i2-article-type-lambda-0-3-c1-f0-s2753-902fcc852d5f",
        "g4-i2-article-type-lambda-0-3-c1-f4-s2753-8d210d54f01e",
        "results/evidence/task2/i2_multitask/manifest.json",
    ):
        assert required in finding


def test_task2_pstar_cell_records_matched_pretraining_boundary() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell = cells["s08-03-03-code"]
    code = cell.source
    finding = cells["s08-03-03-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-03-03-code", "exec")
    for required in (
        "g4_p0s_resnet18_standard_scratch.json",
        "g4_pstar_resnet18_standard_pretrained.json",
        "load_experiment_config",
        "run_pretraining_matrix",
        'mode="run_or_load"',
        "build_experiment_evidence",
        "build_pretraining_benchmark_evidence",
        "protected_ids",
        "calibration_claim_allowed=False",
        "current_run_ids",
        'status"].eq("running")',
        'pretraining_manifest["gate"] == "G4-PSTAR"',
        'pretraining_manifest["candidate_selection_affected"] is False',
        'pretraining_decision["pstar_final_eligible"] is False',
        'artifacts"]["learning_curves"]',
        'artifacts"]["effect_figure"]',
    ):
        assert required in code
    assert cell.execution_count == 30
    assert len(cell.outputs) == 7
    assert all(output.output_type != "error" for output in cell.outputs)
    for required in (
        "0.731172",
        "0.754196",
        "+0.023024",
        "+0.038829",
        "all five paired folds",
        "22` to `11",
        "32.47` to `22.69",
        "teacher-style chart",
        "early stopping",
        "EDA reflection",
        "benchmark ceiling",
        "I2 lambda `0.3`",
        "Notebook 01/data preparation is unchanged",
        "ResNet18_Weights.IMAGENET1K_V1",
        "g4-p0s-resnet18-standard-scratch-f0-s2753-7db32dfd4d00",
        "g4-pstar-resnet18-standard-pretrained-f4-s2753-a15e79285ae2",
        "results/evidence/task2/pretraining_benchmark/manifest.json",
    ):
        assert required in finding


def test_task2_g5_cell_records_second_seed_stability() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell = cells["s08-04-code"]
    code = cell.source
    finding = cells["s08-04-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-04-code", "exec")
    for required in (
        'stability_root = TASK2_EVIDENCE_DIR / "seed_stability"',
        'stability_manifest_path = stability_root / "manifest.json"',
        "verify_artifact",
        'artifact_group in ("artifacts", "input_configs", "input_manifests")',
        'manifest["coverage"]["row_count"] == len(season_development)',
        'manifest["coverage"]["protected_id_count"] == 0',
        'stability_manifest["gate"] == "G5-SEED"',
        'stability_manifest["ordering_stable"] is True',
        'stability_manifest["ultimate_winner_frozen"] is False',
        'stability_decision["current_candidate"] == "I2"',
        "len(stability_registry) == 20",
        'stability_registry["status"].eq("completed").all()',
        'scores["I2"] > scores["C2"]',
        'artifact_paths["learning_curves"]',
        'artifact_paths["comparison_figure"]',
    ):
        assert required in code
    assert cell.execution_count == 31
    assert len(cell.outputs) == 7
    assert all(output.output_type != "error" for output in cell.outputs)
    for required in (
        "0.752687",
        "0.735036",
        "+0.017651",
        "0.744743",
        "0.733137",
        "+0.011607",
        "-0.007944",
        "three of five paired folds",
        "teacher-style figure",
        "EDA reflection",
        "directionally accurate",
        "uniform was inaccurate",
        "I2 lambda `0.3`",
        "do **not** freeze the ultimate winner",
        "04ef69d",
        "results/evidence/task2/seed_stability/manifest.json",
    ):
        assert required in finding


def test_task2_g2_size_cell_records_audited_selection() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-02-01-code"].source
    finding = cells["s08-02-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-02-01-code", "exec")
    for required in (
        "g2_p1_c2_resnet18.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "build_g2_input_size_evidence",
        "g2_input_size_ablation",
        "protected_ids",
        "calibration_claim_allowed=False",
        'g2_size_decision["selected_variant"] == "P0"',
        "RunRegistry(RUNS_CSV)",
        '["status"].eq("running")',
    ):
        assert required in code
    for required in (
        "0.707099",
        "0.705312",
        "-0.001787",
        "58.18",
        "29.21",
        "1.992",
        "2.180",
        "retain P0",
        "four of five",
        "interrupted",
        "failed",
        "g2-p1-c2-resnet18-f0-s2753-67217738d381",
        "g2-p1-c2-resnet18-f4-s2753-9294db7bbaf4",
        "results/evidence/task2/g2_input_size_ablation/manifest.json",
    ):
        assert required in finding


def test_task2_g2_tuning_cell_records_audited_incremental_selection() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-02-03-code"].source
    finding = cells["s08-02-03-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-02-03-code", "exec")
    for required in (
        "g1_c1_smallcnn.json",
        "g1_c2_resnet18.json",
        "g2_t1_c1_smallcnn.json",
        "g2_t2_c1_smallcnn.json",
        "g2_t1_c2_resnet18.json",
        "g2_t2_c2_resnet18.json",
        "run_or_load_experiment",
        "build_experiment_evidence",
        "build_g2_tuning_evidence",
        "protected_ids",
        'selected_tuning_id"] == "T1"',
        'selected_tuning_id"] == "T0"',
        'status"].eq("running")',
        "incremental_model_selection.csv",
        "eda_reflection.csv",
        'artifacts"]["c1_learning_curves"]',
        'artifacts"]["c2_learning_curves"]',
    ):
        assert required in code
    for required in (
        "0.708075",
        "+0.008173",
        "0.708246",
        "+0.001146",
        "select C1-T1",
        "retain C2-T0",
        "train and validation loss",
        "validation accuracy and macro-F1",
        "supported",
        "contradicted",
        "still untested",
        "results/evidence/task2/g2_compact_tuning/manifest.json",
        "g2-t1-c1-smallcnn-f0-s2753-b5a391a13f44",
        "g2-t2-c2-resnet18-f2-s2753-c0de62f1594e",
    ):
        assert required in finding


def test_task2_teacher_feedback_is_explicitly_connected_to_eda() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}

    eda_handoff = "\n".join(
        cells[cell_id].source
        for cell_id in (
            "s02-01-finding",
            "s02-02-01-finding",
            "s02-02-02-finding",
            "s08-02-03-finding",
        )
    )
    for required in (
        "EDA hypothesis",
        "accurate",
        "supported",
        "contradicted",
        "still untested",
        "ArticleType",
        "file size",
        "acquisition year",
    ):
        assert required in eda_handoff

    b0_story = cells["s05-01-heading"].source + cells["s05-01-finding"].source
    for required in (
        "selected first because",
        "Strength",
        "Limitation",
        "B1",
    ):
        assert required in b0_story

    b1_story = cells["s05-02-heading"].source + cells["s05-02-finding"].source
    for required in (
        "selected next because",
        "Strength",
        "Limitation",
        "C1",
    ):
        assert required in b1_story

    architecture_story = "\n".join(
        cells[cell_id].source
        for cell_id in (
            "s06-01-01-finding",
            "s06-01-02-finding",
            "s06-01-03-finding",
        )
    )
    assert "Interpretation to write after running" not in architecture_story
    for required in ("B1", "C1", "C2", "C3", "weights=None", "alternative"):
        assert required in architecture_story


def test_task2_results_cells_load_only_verified_measured_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = (
        "s09-01-code",
        "s09-02-code",
        "s09-03-01-code",
        "s09-03-02-code",
    )

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "load_verified_notebook_manifest",
        "verify_artifact",
        "results/evidence/task2/seed_stability/manifest.json",
        "len(stability_registry) == stability_registry[\"run_id\"].nunique() == 20",
        "results/evidence/task2/g3_c2_t0_resnet18/manifest.json",
        "g4_i2_article_type_lambda_0_3_c1/manifest.json",
        "results/evidence/task2/calibration/manifest.json",
        "cross_fitted_evaluation_claim_allowed",
        "learning_curve_summary",
        "median_best_epoch",
    ):
        assert required in combined
    assert "atomic_write" not in combined
    assert "run_or_load_experiment" not in combined
    assert "train_fold" not in combined

    findings = "\n".join(
        cells[cell_id].source
        for cell_id in (
            "s09-01-finding",
            "s09-02-finding",
            "s09-03-01-finding",
            "s09-03-02-finding",
        )
    )
    for required in (
        "0.752687",
        "0.733137",
        "3,041",
        "0.647103",
        "0.011701",
        "20% review budget",
        "median best epoch is `24`",
        "never holdout",
    ):
        assert required in findings


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
