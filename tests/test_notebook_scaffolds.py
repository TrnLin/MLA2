from __future__ import annotations

import ast
import json
import re

import nbformat
import pandas as pd

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256

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
        "sections": 45,
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


def _code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_only_planned_notebook_names_are_present() -> None:
    allowed = {
        "00_problem_definition.ipynb",
        "01_data_preparation.ipynb",
        "task3_training/smallcnn_baseline_training.ipynb",
        "task3_training/smallcnn_child_experiments.ipynb",
        "task3_training/smallcnn_e3_experiments.ipynb",
        "task3_training/tinyresnet18_pm_e4_experiments.ipynb",
        "task3_training/compactblurcnn_label_smoothing_e5_experiments.ipynb",
        "task3_training/gem_focal_e6_experiments.ipynb",
        "task3_training/tinyconvnext_tinyhrnet_e7_experiments.ipynb",
        "task3_training/early_stopping_translation_e8_experiments.ipynb",
        "task3_training/semantic_filter_exception_balance_e9_experiments.ipynb",
        "task3_training/audience_aux_e10_experiment.ipynb",
        "task3_training/clean_slate_eda.ipynb",
        "task3_training/clean_slate_screen_1.ipynb",
        "task3_training/micro_swin_clean_slate_screen_2.ipynb",
        "task3_training/gem_gender_v2_g1_foreground_mask.ipynb",
        "task3_training/gem_gender_v2_g2_translation.ipynb",
        "task3_training/gem_gender_v2_g3_component_weight.ipynb",
        "task3_training/smallcnn_usage_v2_u1_component_weight.ipynb",
        "task3_training/gem_gender_v2_g2_confirmation.ipynb",
        "task3_training/usage_v2_u2_full_rgb_hog_svm.ipynb",
        "task3_training/gender_gd1_mild_darkening.ipynb",
        "task3_training/gender_weight_decay_screen.ipynb",
        "task3_training/gender_saved_model_diagnostic.ipynb",
        "task3_training/gender_precision_check.ipynb",
        "task3_training/gender_narrow64_screen.ipynb",
        "task3_training/gender_dropout_screen.ipynb",
        "task3_training/gender_dropout_darkening_screen.ipynb",
        "task3_training/gender_stronger_dropout_screen.ipynb",
        "task3_training/gender_grayscale_screen.ipynb",
        "task3_training/gender_name_truth_screen.ipynb",
        "task3_training/gender_group_weight_screen.ipynb",
        "task3_training/gender_mixup_screen.ipynb",
        "task3_training/usage_expanded_e8.ipynb",
        "task3_training/gender_stronger_mixup_screen.ipynb",
        "task3_training/gender_sam_screen.ipynb",
        "task3_training/gender_sam25_screen.ipynb",
        "task3_training/gender_sam25_five_fold.ipynb",
        "task3_training/usage_expanded_v2_e8.ipynb",
        "task3_training/usage_mixup_sam_screen.ipynb",
        "task3_training/usage_replaced_v3_mixup_sam.ipynb",
        "task3_training/usage_two_stage_screen.ipynb",
        *(name for name in TASK_SPECS if name != "05_task4_visual_search.ipynb"),
        "task-4/01_v1_eda.ipynb",
        "task-4/05_task4_visual_search.ipynb",
    }
    present = {
        path.relative_to(ROOT / "notebooks").as_posix()
        for path in (ROOT / "notebooks").rglob("*.ipynb")
    }
    assert present == allowed


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


def test_task_1_scaffold_leaves_owner_decisions_open() -> None:
    for filename in ("02_task1_article_type.ipynb",):
        notebook = nbformat.read(_task_path(filename), as_version=4)
        source = _source(notebook)

        assert all(
            cell.cell_type == "markdown" or not cell.source.strip() for cell in notebook.cells
        )
        assert "TODO(owner)" in source
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
    # The notebook is intentionally split into small output/interpretation
    # trios.  Keep the contract structural rather than freezing a brittle
    # total-cell count: adding a focused diagnostic must remain safe.
    assert len(notebook.cells) >= 193
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
    assert all(cell.execution_count is None and cell.outputs == [] for cell in placeholder_cells)

    h3_indices = [index for index, cell in enumerate(notebook.cells) if _heading_level(cell) == 3]
    h4_indices = [index for index, cell in enumerate(notebook.cells) if _heading_level(cell) == 4]
    assert len(h3_indices) >= 40
    assert len(h4_indices) >= 27
    assert len(code_cells) >= 55

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
        "artifact_replay",
        "load_verified_notebook_manifest",
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


def _display_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "display"
    ]


def test_task2_notebook_has_one_readable_output_per_code_cell() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    deep_analyses: list[tuple[str, str]] = []

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        calls = _display_calls(cell.source)
        assert len(calls) <= 1, f"cell {cell.id} has multiple display calls"
        assert all(len(call.args) == 1 and not call.keywords for call in calls), (
            f"cell {cell.id} uses a multi-value display call"
        )
        assert len(cell.outputs) <= 1, f"cell {cell.id} stores multiple outputs"
        assert all(output.output_type != "error" for output in cell.outputs)

        # A setup/computation cell that has no explicit display must not carry
        # a stale rich output from an earlier execution.
        if not calls and "savefig" not in cell.source:
            assert cell.outputs == [], f"setup cell {cell.id} has stale output"

        # A final bare expression is another implicit visible output.  Output
        # cells must use an explicit display call, while setup/computation
        # cells should end in an assignment/assertion.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            value = tree.body[-1].value
            is_explicit_display = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "display"
            )
            is_file_save = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "savefig"
            )
            assert is_explicit_display or is_file_save, (
                f"cell {cell.id} has an accidental implicit output"
            )

        if calls:
            assert index + 1 < len(notebook.cells)
            following = notebook.cells[index + 1]
            assert following.cell_type == "markdown"
            assert following.source.lstrip().startswith("> **Interpretation")
            assert (
                "**Column guide.**" in following.source or "**Chart guide.**" in following.source
            ), f"cell {cell.id} has no output-specific reading guide"
            assert "**Deep analysis.**" in following.source, (
                f"cell {cell.id} has no output-specific deep analysis"
            )
            deep_analysis = (
                following.source.split("**Deep analysis.**", maxsplit=1)[1]
                .split("> **Decision and limitation.**", maxsplit=1)[0]
                .strip()
            )
            assert len(deep_analysis) >= 120, f"cell {cell.id} has a shallow output analysis"
            deep_analyses.append((cell.id, " ".join(deep_analysis.lower().split())))

            for generic in (
                "A separate compact table exposes",
                "A separate figure keeps",
                "Rows represent checks, configurations, candidates, or folds",
                "The plotted trend is loaded from the verified evidence artifact",
                "This output isolates",
                "read without a wide, scroll-heavy result block",
                "Interpretation to write after running",
                "It supports the frozen development decision in the parent subsection",
                "**Senior analysis.**",
                "**Why this output exists.**",
                "**What it shows.**",
            ):
                assert generic not in following.source, (
                    f"cell {cell.id} still uses a generic interpretation template"
                )

            preceding_heading = notebook.cells[index - 1].source.splitlines()[0].lower()
            if "decision audit" in preceding_heading:
                assert "pd.json_normalize" not in cell.source, (
                    f"cell {cell.id} exposes a tall raw decision JSON view"
                )

    normalized_analyses = [analysis for _, analysis in deep_analyses]
    assert len(normalized_analyses) == len(set(normalized_analyses)), (
        "Task 2 output cells reuse a deep-analysis template"
    )


def test_task2_notebook_has_no_accidental_matplotlib_output() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code" or "plt.subplots" not in cell.source:
            continue
        assert "plt.close(" in cell.source, f"cell {cell.id} leaves a figure open"


def test_task2_notebook_maps_training_code_without_executing_it() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    title = next(cell.source for cell in notebook.cells if cell.id == "task2-title")

    for required in (
        "Training code map (read-only)",
        "[B0/B1 and G0-G5 declarations](../configs/task2/)",
        "fit_training_fold_majority",
        "fit_hog_hsv_svm",
        "SeasonArticleTypeMultiTaskModel",
        "build_multitask_season_model",
        "[general B0/B1 and G1-G3](../scripts/run_task2_experiment.py)",
        "[I1](../scripts/run_task2_i1_experiment.py)",
        "[I2](../scripts/run_task2_i2_experiments.py)",
        "[P0S/P* benchmark](../scripts/run_task2_pretraining_benchmark.py)",
        "[stability](../scripts/run_task2_stability.py)",
        "run_or_load_experiment",
        "_execute_baseline",
        "run_or_load_g0_smoke",
        "run_or_load_i1_experiment",
        "run_or_load_i2_experiment",
        "run_i2_matrix",
        "run_pretraining_matrix",
        "run_stability_matrix",
        "train_masked_multitask_fold",
        "[refit_task2_season.py](../scripts/refit_task2_season.py)",
        "run_or_load_development_refit",
        "train_masked_multitask_refit",
        "[refit history](../results/evidence/task2/development_refit/training_history.csv)",
        "[final weights](../models/task2_season.pt)",
        "benchmark-only",
        "Markdown never imports or executes Python",
        "Restart Kernel and Run All Cells",
        "`load` | Verify and load",
        "`run_or_load` | Verify and reuse",
        "`run` | Deliberately start",
        "Never use it for G0, I1, I2, P0S/P*, or G5",
    ):
        assert required in title

    assert "without being imported or executed here" in title
    assert "use the declared script outside Notebook 03" in title
    assert title.count("--mode load") == 6
    assert title.count("--mode run") == 6


def test_task2_notebook_is_fully_executed_artifact_replay() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    combined = "\n".join(cell.source for cell in code_cells)

    assert len(code_cells) == 147
    assert [cell.execution_count for cell in code_cells] == list(range(1, 148))
    assert all(len(cell.outputs) == 1 for cell in code_cells)
    assert all(output.output_type != "error" for cell in code_cells for output in cell.outputs)
    assert 'TASK2_NOTEBOOK_MODE = "artifact_replay"' in combined
    assert "load_verified_notebook_manifest" in combined
    assert "verify_artifact" in combined

    for forbidden in (
        'mode="run_or_load"',
        "mode='run_or_load'",
        "run_or_load_experiment",
        "run_or_load_g0_smoke",
        "run_or_load_i1_experiment",
        "run_or_load_i2_experiment",
        "run_i2_matrix",
        "run_pretraining_matrix",
        "run_stability_matrix",
        "run_or_load_development_refit",
        "train_masked_multitask_fold",
        "train_masked_multitask_refit",
        "build_experiment_evidence(",
        "build_g0_evidence(",
        "build_g1_family_screen_evidence(",
        "build_g3_full_budget_evidence(",
        "build_g2_input_size_evidence(",
        "build_g2_augmentation_evidence(",
        "build_g2_compact_tuning_evidence(",
        "build_i1_class_balance_evidence(",
        "build_i2_multitask_evidence(",
        "build_pretraining_evidence(",
        "build_task2_handoff_evidence(",
        "load_or_fit_fold_stats",
        "atomic_write",
        ".savefig(",
        ".to_csv(",
        ".to_json(",
        ".write_text(",
        ".write_bytes(",
        "torch.save(",
        ".mkdir(",
        "train_fold(",
        ".backward(",
    ):
        assert forbidden not in combined


def test_task2_replay_locks_match_every_declared_root() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    setup = next(cell.source for cell in notebook.cells if cell.id == "s01-01-code")
    assignment = next(
        node
        for node in ast.parse(setup).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TASK2_REPLAY_LOCKS"
            for target in node.targets
        )
    )
    replay_locks = ast.literal_eval(assignment.value)

    for required in (
        "data/processed/splits.csv",
        "results/evidence/task2/environment.json",
        "results/evidence/task2/registry_health.json",
        "results/evidence/task2/registry_recovery.json",
        "results/evidence/task2/selection_freeze.json",
        "results/evidence/task2/ultimate_judgement/manifest.json",
        "results/evidence/task2/final_handoff/manifest.json",
        "results/figures/task2/fold0_preprocessing_audit.png",
        "results/figures/task2/development_refit_training_curve.png",
        "models/task2_season.manifest.json",
    ):
        assert required in replay_locks
    for relative_path, expected_sha256 in replay_locks.items():
        assert compute_sha256(ROOT / relative_path) == expected_sha256


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
        "FoldImageStats(",
        "stats=fold0_frozen_stats",
        "validate_oof",
        "protected labels remain sealed",
        'TASK2_NOTEBOOK_MODE = "artifact_replay"',
        "load_verified_notebook_manifest",
        "verify_artifact",
        "environment.json",
        "eda_handoff.csv",
        "fold_handoff.csv",
        "oof_contract.json",
        "metric_contract.json",
        "transform_matrix.csv",
        "leakage_audit.json",
    ):
        assert required in combined
    assert "atomic_write" not in combined


def test_task2_environment_table_uses_dataframe_compatible_reset_index() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    environment_cell = next(cell for cell in notebook.cells if cell.id == "s01-02-code")

    assert ".reset_index(name=" not in environment_cell.source


def test_task2_preprocessing_mask_uses_an_informative_padding_example() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    preprocessing_cell = next(cell for cell in notebook.cells if cell.id == "s04-01-code")

    assert "nonstandard_aspect" in preprocessing_cell.source
    assert "first_content_mask.any() and (~first_content_mask).any()" in preprocessing_cell.source
    assert "fold0_preprocessing_audit.png" in preprocessing_cell.source
    assert "stats=fold0_frozen_stats" in preprocessing_cell.source
    assert "verify_artifact(" in preprocessing_cell.source
    assert "load_or_fit_fold_stats" not in preprocessing_cell.source
    assert ".savefig(" not in preprocessing_cell.source


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
        "registry_health.json",
        "registry_status_counts",
        "scratch_model_audit.csv",
        "benchmark_boundary.csv",
    ):
        assert required in combined
    assert "RunRegistry" not in combined


def test_task2_g0_cell_records_a_non_comparison_pass() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s07-01-code"].source
    finding = cells["s07-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s07-01-code", "exec")
    for required in (
        'TASK2_EVIDENCE_DIR / "g0/manifest.json"',
        "verify_artifact",
        'g0_manifest["passed"] is True',
        'g0_manifest["comparison_eligible"] is False',
        "integration_macro_f1_non_comparison",
    ):
        assert required in code
    assert "run_or_load_g0_smoke" not in code
    assert "build_g0_evidence" not in code
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
        "results/evidence/task2/b0_majority/manifest.json",
        "load_verified_notebook_manifest",
        'b0_artifacts["pooled_metrics"]',
        'b0_artifacts["fold_summary"]',
        'b0_manifest["coverage"]["protected_id_count"] == 0',
        'b0_manifest["run_ids"]',
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "build_experiment_evidence" not in code
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
        "results/evidence/task2/b1_hog_hsv_svm/manifest.json",
        "load_verified_notebook_manifest",
        'b1_artifacts["pooled_metrics"]',
        'b1_artifacts["registry_snapshot"]',
        'not b1_manifest["calibration_claim_allowed"]',
        "macro_f1_gain_over_B0",
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "threading" not in code
    assert "32,753" in finding
    assert "0.609561" in finding
    assert "0.486901" in finding
    assert "not calibrated probabilities" in finding
    assert "operational checkpoints" in finding
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
        "results/evidence/task2/g1_family_screen/manifest.json",
        "load_verified_notebook_manifest",
        "load_verified_input_registry_rows",
        'g1_screen_artifacts["leaderboard"]',
        'g1_screen_artifacts["shortlist"]',
        "len(g1_registry_rows) == 15",
        "g1_trace_rows = g1_trace.to_dict",
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "build_g1_family_screen_evidence" not in code
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
        "results/evidence/task2/g3_full_budget/manifest.json",
        "load_verified_notebook_manifest",
        "load_verified_input_registry_rows",
        'g3_artifacts["leaderboard"]',
        'g3_artifacts["decision"]',
        'g3_attempts["status"].eq("completed").all()',
        'near_tie"] is True',
        'ultimate_winner_frozen"] is False',
        'len(g3_attempts) == g3_attempts["run_id"].nunique() == 10',
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "build_g3_full_budget_evidence" not in code
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

    assert "g3_attempts = load_verified_input_registry_rows(g3_manifest)" in code
    assert 'len(g3_attempts) == g3_attempts["run_id"].nunique() == 10' in code
    assert 'g3_attempts["status"].eq("completed").all()' in code
    assert "RunRegistry" not in code


def test_task2_g3_notebook_preserves_registered_probability_note() -> None:
    note = (
        "Uncalibrated softmax probabilities from the best validation macro-F1 "
        "checkpoint in each fold; calibration metrics are diagnostic only."
    )
    for directory in ("g3_c1_t1_smallcnn", "g3_c2_t0_resnet18"):
        manifest = json.loads(
            (ROOT / "results/evidence/task2" / directory / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["probability_note"] == note


def test_task2_i1_cell_records_audited_class_balance_decision() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s08-03-01-code"].source
    finding = cells["s08-03-01-finding"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s08-03-01-code", "exec")
    for required in (
        "results/evidence/task2/i1_class_balance/manifest.json",
        "load_verified_notebook_manifest",
        'i1_artifacts["comparison"]',
        'i1_artifacts["registry_snapshot"]',
        'i1_artifacts["class_weights_by_fold"]',
        'i1_manifest["gate"] == "G4-I1"',
        'i1_manifest["keep_i1"] is False',
        'i1_manifest["selected_experiment_id"] == "g3-c1-t1-smallcnn"',
        'i1_decision["loss_values_comparable_to_reference"] is False',
        'i1_registry_snapshot["experiment_id"]',
    ):
        assert required in code
    assert "run_or_load_i1_experiment" not in code
    assert "build_i1_class_balance_evidence" not in code
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
        "results/evidence/task2/i2_multitask/manifest.json",
        "load_verified_notebook_manifest",
        'i2_artifacts["comparison"]',
        'i2_artifacts["registry_snapshot"]',
        'i2_artifacts["slice_metrics"]',
        'i2_manifest["gate"] == "G4-I2"',
        'i2_manifest["keep_i2"] is True',
        'i2_manifest["selected_experiment_id"] == "g4-i2-article-type-lambda-0-3-c1"',
        'i2_decision["loss_values_comparable_to_reference"] is False',
        'i2_current_attempts["status"].eq("completed").all()',
    ):
        assert required in code
    assert "run_i2_matrix" not in code
    assert "build_i2_transfer_evidence" not in code
    assert cell.execution_count is not None
    assert len(cell.outputs) == 1
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
        "results/evidence/task2/pretraining_benchmark/manifest.json",
        "load_verified_notebook_manifest",
        'pretraining_artifacts["comparison"]',
        'pretraining_artifacts["registry_snapshot"]',
        'pretraining_manifest["gate"] == "G4-PSTAR"',
        'pretraining_manifest["candidate_selection_affected"] is False',
        'pretraining_decision["pstar_final_eligible"] is False',
        'pretraining_current_attempts["status"].eq("completed").all()',
    ):
        assert required in code
    assert "run_pretraining_matrix" not in code
    assert "build_pretraining_benchmark_evidence" not in code
    assert cell.execution_count is not None
    assert len(cell.outputs) == 1
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
    assert cell.execution_count is not None
    assert len(cell.outputs) == 1
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
        "results/evidence/task2/g2_input_size_ablation/manifest.json",
        "load_verified_notebook_manifest",
        "load_verified_input_registry_rows",
        'g2_size_artifacts["comparison"]',
        'g2_size_artifacts["decision"]',
        'g2_size_decision["selected_variant"] == "P0"',
        'g2_p1_attempts["status"].eq("completed").all()',
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "build_g2_input_size_evidence" not in code
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
        "results/evidence/task2/g2_compact_tuning/manifest.json",
        "results/evidence/task2/selection_story/manifest.json",
        "load_verified_notebook_manifest",
        "load_verified_input_registry_rows",
        'selected_tuning_id"] == "T1"',
        'selected_tuning_id"] == "T0"',
        'selection_story_artifacts["incremental_model_selection"]',
        'selection_story_artifacts["eda_reflection"]',
        'g2_tuning_attempts["status"].eq("completed").all()',
    ):
        assert required in code
    assert "run_or_load_experiment" not in code
    assert "build_g2_tuning_evidence" not in code
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

    combined = (
        cells["s01-01-code"].source
        + "\n"
        + "\n".join(cells[cell_id].source for cell_id in cell_ids)
    )
    for required in (
        "load_verified_notebook_manifest",
        "verify_artifact",
        "results/evidence/task2/seed_stability/manifest.json",
        'len(stability_registry) == stability_registry["run_id"].nunique() == 20',
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


def test_task2_slice_cells_load_verified_post_inference_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = (
        "s10-01-01-code",
        "s10-01-02-code",
        "s10-02-01-code",
        "s10-02-02-code",
        "s10-03-01-code",
        "s10-03-02-code",
    )

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "results/evidence/task2/shortcut_error_slices/manifest.json",
        'slice_manifest["analysis_role"] == "development_oof_diagnosis_only"',
        'slice_manifest["candidate_selection_affected"] is False',
        'slice_manifest["ultimate_winner_frozen"] is False',
        "results/evidence/task2/b0_majority/manifest.json",
        "results/evidence/task2/b1_hog_hsv_svm/manifest.json",
        "results/evidence/task2/i1_class_balance/manifest.json",
        "article_type_fold_audit",
        "missing_article_type",
        "training_id_sha256",
        "file_size_boundaries",
        "product_family_size",
        "greyscale_deltas",
        "other_mode",
    ):
        assert required in combined
    assert "train_test_split" not in combined
    assert "train_fold" not in combined
    assert "run_or_load_experiment" not in combined

    findings = "\n".join(
        cells[cell_id].source
        for cell_id in (
            "s10-01-01-finding",
            "s10-01-02-finding",
            "s10-02-01-finding",
            "s10-02-02-finding",
            "s10-03-01-finding",
            "s10-03-02-finding",
        )
    )
    for required in (
        "B0 ignores",
        "I1 raises recall",
        "+0.017238",
        "11,619",
        "+0.026993",
        "higher accuracy yet much lower macro-F1",
        "0.038842",
        "26,201-26,203",
        "-0.009415",
        "only `294` greyscale rows",
        "reverses across seeds",
        "never an inference input",
    ):
        assert required in findings


def test_task2_robustness_cells_load_verified_stress_and_cost_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = ("s11-01-code", "s11-02-01-code", "s11-02-02-code")

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "results/evidence/task2/robustness_cost/manifest.json",
        'robustness_manifest["git_dirty"] is False',
        'robustness_manifest["candidate_selection_affected"] is False',
        'robustness_manifest["ultimate_winner_frozen"] is False',
        "expected_conditions",
        "clean_reconciliation",
        'len(probe_registry) == probe_registry["probe_id"].nunique() == 50',
        "model_only_warmups",
        "end_to_end_repeats",
        "parameter_and_buffer_bytes",
        "process_rss_delta_bytes",
        "peak_cuda_allocated_bytes",
        "deployment_cost_figure",
    ):
        assert required in combined
    assert "train_fold" not in combined
    assert "run_or_load_experiment" not in combined

    findings = "\n".join(
        cells[cell_id].source
        for cell_id in (
            "s11-01-finding",
            "s11-02-01-finding",
            "s11-02-02-finding",
        )
    )
    for required in (
        "brightness `0.85` is the worst",
        "0.363961",
        "0.003010",
        "6.493/7.069 ms",
        "47.357/51.721 ms",
        "1.312",
        "1,206,112",
        "0.108x",
        "28.56",
        "Process RSS",
        "both better and cheaper",
    ):
        assert required in findings


def test_task2_gradcam_cells_load_verified_noncausal_failure_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = ("s12-01-code", "s12-02-code", "s12-03-code")

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "results/evidence/task2/gradcam_failure_review/manifest.json",
        'gradcam_manifest["holdout_opened"] is False',
        'gradcam_manifest["causal_failure_claim_allowed"] is False',
        'gradcam_manifest["candidate_selection_affected"] is False',
        'gradcam_manifest["ultimate_winner_frozen"] is False',
        "len(selected_examples) == len(heatmap_index) == 48",
        'selected_examples["id"].nunique() == 44',
        'gradcam_checkpoints["run_id"].nunique() == 8',
        'attention_metrics["probability_max_absolute_delta"].max() <= 1e-4',
        "c2_contact_sheet",
        "i2_contact_sheet",
        "len(failure_taxonomy) == 24",
        "human_label_ambiguity_review_required",
        "diagnostic_tags",
    ):
        assert required in combined
    assert "generate_gradcam" not in combined
    assert "run_or_load_experiment" not in combined

    findings = "\n".join(
        cells[cell_id].source for cell_id in ("s12-01-finding", "s12-02-finding", "s12-03-finding")
    )
    for required in (
        "`48` rows, `44` distinct images",
        "not a random sample",
        "0.633160/1.490200",
        "0.642716/1.719056",
        "cannot establish why",
        "`6` ArticleType-shortcut conflicts",
        "`9` conflicts",
        "Every selected error",
        "not causal labels or frequency estimates",
    ):
        assert required in findings


def test_task2_statistical_and_literature_cells_keep_claim_boundaries() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = ("s13-01-code", "s13-02-code")

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "results/evidence/task2/paired_bootstrap/manifest.json",
        'bootstrap_manifest["holdout_opened"] is False',
        'bootstrap_manifest["new_candidates_allowed"] is False',
        'bootstrap_manifest["random_seed_generalisability_claim_allowed"] is False',
        'bootstrap_manifest["ultimate_winner_frozen"] is False',
        'macro_intervals["replicates"].eq(10000).all()',
        'bootstrap_groups.loc[0, "unique_group_count"] == 22885',
        "conservative_dependency_block_not_verified_sku",
        "ResNet-BERT",
        "Condition-CNN",
        "journal.pone.0324621",
        "S0957417421006291",
        "Selvaraju_Grad-CAM_Visual_Explanations",
        "294a8ed24b1ad22ec2e7efea049b8737",
        "10.1111/j.1467-9868.2007.00593.x",
        '"direct_score_comparison": "no"',
    ):
        assert required in combined
    assert "train_fold" not in combined
    assert "bootstrap_draws.csv" not in combined

    findings = cells["s13-01-finding"].source + cells["s13-02-finding"].source
    for required in (
        "+0.017651",
        "[0.013050, 0.022453]",
        "+0.011607",
        "[0.005793, 0.017341]",
        "does not prove I2 wins under every training seed",
        "Fall and Spring class intervals include zero",
        "support architecture ideas, not a numeric benchmark",
        "cannot validate or rank our macro-F1",
        "non-causal, human-reviewed interpretation",
    ):
        assert required in findings


def test_task2_ultimate_judgement_cells_load_verified_freeze_evidence() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = ("s14-01-code", "s14-02-code")

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "load_verified_ultimate_judgement_manifest",
        "results/evidence/task2/ultimate_judgement/manifest.json",
        'ultimate_manifest["ultimate_winner_frozen"] is True',
        'ultimate_manifest["holdout_opened"] is False',
        'ultimate_manifest["selected_candidate"] == "I2"',
        'decision["direct_selection_rule_passed"] is True',
        'decision["cost_tie_break_used"] is False',
        'scorecard["selected"].sum() == 1',
        "load_verified_selection_freeze",
        "TASK2_SELECTION_FREEZE_JSON",
        'selection_freeze["refit_rule"]["epochs"] == 24',
        'selection_freeze["selected_model"]["scratch"] is True',
        'selection_freeze["selected_model"]["weights"] is None',
        'selection_freeze["selected_model"]["inference_inputs"] == ["image"]',
        'selection_freeze["holdout_metrics_present"] is False',
        'ultimate_artifacts["selection_freeze"] == selection_freeze_path',
    ):
        assert required in combined
    assert "train_fold" not in combined
    assert "run_or_load_experiment" not in combined

    findings = cells["s14-01-finding"].source + cells["s14-02-finding"].source
    for required in (
        "I2 is the frozen Task 2 winner",
        "0.752687",
        "0.735036",
        "+0.017651",
        "both seeds",
        "smaller and faster",
        "brightness `0.85`",
        "Spring recall to `0.003010`",
        "all `32,753` valid development rows",
        "exactly `24` epochs",
        "no validation or holdout early stopping",
        "image-only",
        "holdout remains sealed",
    ):
        assert required in findings


def test_task2_refit_cell_loads_verified_bundle_without_evaluation_leakage() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    code = cells["s14-03-code"].source

    assert not code.startswith("# TODO:")
    compile(code, "03_task2_season.ipynb:s14-03-code", "exec")
    for required in (
        "load_verified_development_refit_manifest",
        'refit_manifest["gate"] == "G8-DEVELOPMENT-REFIT"',
        'refit_manifest["selected_candidate"] == "I2"',
        'refit_manifest["scratch"] is True',
        'refit_manifest["weights"] is None',
        'refit_manifest["valid_development_rows"] == 32753',
        'refit_manifest["final_epoch"] == 24',
        'refit_manifest["validation_used"] is False',
        'refit_manifest["early_stopping_used"] is False',
        'refit_manifest["holdout_opened"] is False',
        'refit_manifest["holdout_metrics_present"] is False',
        'refit_manifest["primary_metric_name"] is None',
        'refit_bundle["inference"]["inputs"] == ["image"]',
        'refit_bundle["auxiliary"]["used_at_inference"] is False',
        'not any("validation" in column.lower()',
        "development_refit_training_curve.png",
    ):
        assert required in code
    for forbidden in (
        "train_fold",
        "train_masked_multitask_refit",
        "run_or_load_development_refit",
        "load_splits_for_final_evaluation",
    ):
        assert forbidden not in code

    finding = cells["s14-03-finding"].source
    for required in (
        "all `32,753` valid development rows",
        "frozen `24` epochs",
        "`1,206,112` parameters",
        "`1.789050` to `0.551781`",
        "`0.559124` to `0.820505`",
        "optimisation diagnostics, not unbiased performance estimates",
        "five-fold CV curves in Section 8.4",
        "no validation selection, no early stopping, and no holdout labels",
        "temperature `1.365002`",
        "ArticleType is not an inference input",
        "holdout remains sealed",
        "task2-season-i2-refit-fall-s2753-637dd6378be9",
    ):
        assert required in finding


def test_task2_markdown_matches_final_model_and_recovered_registry() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    model_manifest_path = ROOT / "models/task2_season.manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    handoff = json.loads(
        (ROOT / "results/evidence/task2/final_handoff/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    recovery = json.loads(
        (ROOT / "results/evidence/task2/registry_recovery.json").read_text(
            encoding="utf-8"
        )
    )

    refit_finding = cells["s14-03-finding"].source
    for required in (
        model_manifest["selected_candidate"],
        model_manifest["selected_experiment_id"],
        model_manifest["run_id"],
        model_manifest["bundle"]["sha256"],
        compute_sha256(model_manifest_path),
        f'{(ROOT / model_manifest["bundle"]["path"]).stat().st_size:,} bytes',
        f'{model_manifest["parameter_count"]:,}',
        f'{model_manifest["valid_development_rows"]:,}',
        f'{model_manifest["final_epoch"]} epochs',
        f'{model_manifest["temperature"]:.6f}',
        "weights=None",
    ):
        assert required in refit_finding

    registry_findings = "\n".join(
        cells[cell_id].source
        for cell_id in ("s07-02-finding", "s07-02-output-1-finding")
    )
    status_counts = recovery["final_task2_status_counts"]
    for required in (
        str(recovery["recovery"]["task2_rows"]),
        str(status_counts["completed"]),
        str(status_counts["failed"]),
        str(status_counts["interrupted"]),
        recovery["active_refit_run_id"],
        "historical eligibility snapshot",
        "final recovered Task 2 ledger",
    ):
        assert required in registry_findings

    assert handoff["run_id"] == model_manifest["run_id"]
    assert handoff["artifacts"]["model_bundle"] == model_manifest["bundle"]
    assert handoff["status"] == "ready_for_group_freeze"
    assert handoff["holdout_opened"] is False


def test_task2_final_cells_build_only_the_locked_component_handoff() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}
    cell_ids = ("s15-01-01-code", "s15-01-02-code", "s15-02-code")

    for cell_id in cell_ids:
        code = cells[cell_id].source
        assert not code.startswith("# TODO:")
        compile(code, f"03_task2_season.ipynb:{cell_id}", "exec")

    combined = "\n".join(cells[cell_id].source for cell_id in cell_ids)
    for required in (
        "audit_task2_artifacts",
        'task2_artifact_audit["status"].eq("PASS").all()',
        "load_season_bundle",
        "predict_season",
        'eq("1163")',
        "task2_smoke_prediction.review_required is None",
        "load_verified_task2_handoff",
        'task2_handoff["status"] == "ready_for_group_freeze"',
        'task2_handoff["group_freeze_verified"] is False',
        'task2_handoff["notebook_06_unlocked"] is False',
        'task2_handoff["holdout_opened"] is False',
        'task2_handoff["model_change_allowed"] is False',
    ):
        assert required in combined
    for forbidden in (
        "build_task2_handoff_evidence",
        "evaluation_unlocked",
        "load_splits_for_final_evaluation",
        "styles_prediction.csv",
        "HTML export",
    ):
        assert forbidden not in combined

    findings = "\n".join(
        cells[cell_id].source
        for cell_id in ("s15-01-01-finding", "s15-01-02-finding", "s15-02-finding")
    )
    for required in (
        "All `10/10` Task 2 component checks pass",
        "not claim holdout performance",
        "development product `1163` as Summer",
        "not an accuracy estimate",
        "`ready_for_group_freeze`",
        "`group_freeze_verified=False`",
        "`notebook_06_unlocked=False`",
        "`holdout_opened=False`",
        "No Task 2 model change is allowed",
    ):
        assert required in findings


def test_task2_analysis_cells_verify_all_declared_inputs_and_exact_claims() -> None:
    notebook = nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4)
    cells = {cell.id: cell for cell in notebook.cells}

    manifest_loader = cells["s01-01-code"].source
    for required in (
        "def walk_declarations",
        '{"path", "sha256"} <= set(value)',
        "walk_declarations(manifest)",
        'verified_declarations[f"artifacts.{name}"]',
    ):
        assert required in manifest_loader

    robustness = cells["s11-01-code"].source
    assert "development_stress_and_machine_cost_diagnosis_only" in robustness
    assert 'clean_reconciliation["support"].eq(32753).all()' in robustness
    assert 'clean_reconciliation["clean_reference"]' in robustness
    assert '["candidate", "condition"], observed=True' in robustness

    bootstrap = cells["s13-01-code"].source
    assert 'macro_intervals["ci_lower"] > macro_intervals["practical_tie_threshold"]' in bootstrap

    gradcam_finding = cells["s12-02-finding"].source
    assert "mean border lift stays below `1.0`" in gradcam_finding
    assert "below area share" not in gradcam_finding


def test_task_metric_placeholders_are_explicit() -> None:
    task1 = _source(nbformat.read(ROOT / "notebooks/02_task1_article_type.ipynb", as_version=4))
    assert "Primary development metric: TODO(owner)" in task1

    task2 = _source(nbformat.read(ROOT / "notebooks/03_task2_season.ipynb", as_version=4))
    assert "Primary development metric:** pooled out-of-fold macro-F1" in task2

    task3 = _source(nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4))
    assert "Primary development metric for `gender`: pooled five-fold OOF macro-F1" in task3
    assert "Primary development metric for `usage`: pooled five-fold OOF macro-F1" in task3

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
        or not any(output.get("output_type") == "error" for output in cell["outputs"])
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
    assert (
        "Model quality remains unknown until the baseline and learned search methods are run."
        not in source
    )
    assert "Only learned-model quality remains unknown." in source
    assert (ROOT / "results/figures/task4/retrieval_protocol_overview.png").exists()


def test_active_task4_notebooks_use_canonical_import_owners() -> None:
    for filename in ("01_v1_eda.ipynb", "05_task4_visual_search.ipynb"):
        notebook = nbformat.read(ROOT / "notebooks/task-4" / filename, as_version=4)
        code_source = _code_source(notebook)

        assert "fashion.retrieval" not in code_source
        assert "fashion.task4" in code_source


def test_task4_preprocessing_milestone_is_frozen_and_executed() -> None:
    notebook = nbformat.read(_task_path("05_task4_visual_search.ipynb"), as_version=4)
    source = _source(notebook)
    preprocessing = source.split("## 5. Task-specific preprocessing and leakage rules", maxsplit=1)[
        1
    ].split("## 7. Hypotheses and baseline", maxsplit=1)[0]

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
    recomputed = primary_selection_rows.groupby("size")["value"].mean().sort_values(ascending=False)
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
    contract = json.loads((evidence / "preprocessing_contract.json").read_text(encoding="utf-8"))
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


def test_task3_clean_slate_eda_is_separate_and_label_safe() -> None:
    notebook = nbformat.read(ROOT / "notebooks/task3_training/clean_slate_eda.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — Clean-Slate EDA"
    assert "write_clean_slate_eda_tables" in code
    assert "load_splits" in code
    assert "train_test_split" not in source
    assert "load_splits_for_final_evaluation" not in source
    assert "pretrained=True" not in source
    assert "observability gate" in source.lower()
    assert "high-resolution" not in source.lower()
    assert "external_image" not in source
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert {
        "t3-clean-eda-setup",
        "t3-clean-eda-run",
        "t3-clean-eda-observability",
        "t3-clean-eda-foreground",
        "t3-clean-eda-nuisance",
        "t3-clean-eda-family",
        "t3-clean-eda-neighbourhoods",
        "t3-clean-eda-folds",
        "t3-clean-eda-gate",
    } == {cell.id for cell in notebook.cells if cell.cell_type == "code"}


def test_task3_micro_swin_screen_is_separate_colab_gpu_work() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/micro_swin_clean_slate_screen_2.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — Scratch Micro-Swin Clean-Slate Screen 2"
    assert "Run All starts four fits" in source
    assert "check_micro_swin_screen_setup" in code
    assert code.count("run_micro_swin_screen(") == 2
    assert 'device_name="cuda"' in code
    assert "reuse_completed=True" in code
    assert "folds 0 and 4" in source
    assert "train_test_split" not in source
    assert "pretrained=True" not in source
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    assert {
        "t3-cs2-config",
        "t3-cs2-repository",
        "t3-cs2-data",
        "t3-cs2-check",
        "t3-cs2-usage",
        "t3-cs2-gender",
        "t3-cs2-summary",
    } == {cell.id for cell in notebook.cells if cell.cell_type == "code"}


def test_task3_baseline_training_runner_is_foreground_and_complete() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/smallcnn_baseline_training.ipynb", as_version=4
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
        ROOT / "notebooks/task3_training/smallcnn_child_experiments.ipynb", as_version=4
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
        ROOT / "notebooks/task3_training/smallcnn_e3_experiments.ipynb", as_version=4
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
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e4_runner_only_trains_tinyresnet_children() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/tinyresnet18_pm_e4_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — TinyResNet-18-PM E4 Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert '"gender_tinyresnet18_pm"' in source
    assert '"usage_tinyresnet18_pm"' in source
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert 'parameter_count"] > 410_000' in source
    assert 'architecture_macs"] > 105_000_000' in source
    assert "run_task3_baseline_cv" not in source
    assert 'run_task3_child_cv(\n    "gender_brightness"' not in source
    assert 'run_task3_child_cv(\n    "usage_class_balanced"' not in source
    assert 'run_task3_child_cv(\n    "gender_class_balanced"' not in source
    assert 'run_task3_child_cv(\n    "usage_classifier_dropout"' not in source
    assert "START_BASELINE_TRAINING" not in code
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e5_runner_only_trains_frozen_e5_children() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/compactblurcnn_label_smoothing_e5_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert (
        notebook.metadata["title"] == "Task 3 — CompactBlurCNN and Label-Smoothing E5 Experiments"
    )
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert '"gender_compact_blur_cnn"' in source
    assert '"usage_label_smoothing"' in source
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert 'gender_e5_check["parameter_count"] > 100_000' in source
    assert 'gender_e5_check["architecture_macs"] > 35_000_000' in source
    assert 'usage_e5_check["label_smoothing"] != 0.05' in source
    assert "run_task3_baseline_cv" not in source
    assert "gender_tinyresnet18_pm" not in source
    assert "usage_tinyresnet18_pm" not in source
    assert "START_BASELINE_TRAINING" not in code
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e6_runner_only_trains_gem_and_focal_children() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/gem_focal_e6_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — GeM and Focal-Loss E6 Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert '"gender_gem_p3"' in source
    assert '"usage_focal_gamma1"' in source
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert source.count("audit_completed_registry_rows(") == 2
    assert 'gender_e6_check["parameter_count"] != 390_181' in source
    assert 'usage_e6_check["focal_gamma"] != 1.0' in source
    assert "run_task3_baseline_cv" not in source
    assert "gender_compact_blur_cnn" not in source
    assert "usage_label_smoothing" not in source
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e7_runner_only_trains_frozen_architecture_children() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/tinyconvnext_tinyhrnet_e7_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — TinyConvNeXt and TinyHRNet E7 Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert source.index('run_task3_child_cv(\n    "usage_tinyconvnext18"') < source.index(
        'run_task3_child_cv(\n    "gender_tinyhrnet20"'
    )
    assert "latest_completed_baseline_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert source.count("audit_completed_registry_rows(") == 2
    assert 'usage_e7_check["parameter_count"] != 384_345' in source
    assert 'usage_e7_check["architecture_macs"] != 95_297_616' in source
    assert 'gender_e7_check["parameter_count"] != 374_445' in source
    assert 'gender_e7_check["architecture_macs"] != 104_064_700' in source
    assert "run_task3_baseline_cv" not in source
    assert "gender_gem_p3" not in source
    assert "usage_focal_gamma1" not in source
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e8_runner_only_trains_early_stopping_and_translation() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/early_stopping_translation_e8_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — Early Stopping and Translation E8 Experiments"
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert source.index('run_task3_child_cv(\n    "gender_gem_p3_early_stopping"') < source.index(
        'run_task3_child_cv(\n    "usage_translation_2px"'
    )
    assert "latest_completed_gender_e6_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert source.count("audit_completed_registry_rows(") == 2
    assert 'gender_child["checkpoint_policy"] != "best_validation_macro_f1"' in source
    assert 'usage_child["training_augmentation"] != "translation_uniform_2px_p05"' in source
    assert "run_task3_baseline_cv" not in source
    assert "usage_tinyconvnext18" not in source
    assert "gender_tinyhrnet20" not in source
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e9_runner_trains_only_e9_with_deterministic_gender_audit() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/semantic_filter_exception_balance_e9_experiments.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert (
        notebook.metadata["title"]
        == "Task 3 — Semantic Filter and Exception Balance E9 Experiments"
    )
    assert source.count("folds=range(5)") == 2
    assert source.count("run_task3_child_cv(") == 2
    assert source.index('run_task3_child_cv(\n    "usage_exception_balance"') < source.index(
        'run_task3_child_cv(\n    "gender_semantic_filter"'
    )
    assert "write_task3_e9_prerun_evidence" in source
    assert "Deterministic E9 evidence ready in Drive; optimizer steps: 0" in source
    assert source.index("gender_contract = e9_prerun") < source.index(
        'run_task3_child_cv(\n    "gender_semantic_filter"'
    )
    assert "GENDER_E9_APPROVED" not in source
    assert "require_gender_e9_training_approval" not in source
    assert "three-rater" not in source
    assert "human_rating_gate_required" in source
    assert "latest_completed_gender_e6_parent_run_ids" in source
    assert "latest_completed_usage_e2_parent_run_ids" in source
    assert "no switch or merge was attempted" in source
    assert source.count("audit_completed_registry_rows(") == 2
    assert "run_task3_baseline_cv" not in source
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e10_runner_trains_only_the_gender_audience_child() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/task3_training/audience_aux_e10_experiment.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == ("Task 3 — Gender Audience-Auxiliary E10 Experiment")
    assert source.count("folds=range(5)") == 1
    assert source.count("run_task3_child_cv(") == 1
    assert 'run_task3_child_cv(\n    "gender_audience_aux"' in source
    assert "latest_completed_gender_e6_parent_run_ids" in source
    assert "write_task3_e10_prerun_evidence" in source
    assert source.index("write_task3_e10_prerun_evidence(") < source.index(
        'run_task3_child_cv(\n    "gender_audience_aux"'
    )
    assert source.count("audit_completed_registry_rows(") == 1
    assert "usage_exception_balance" not in source
    assert "gender_semantic_filter" not in source
    assert "run_task3_baseline_cv" not in source
    assert "nohup" not in code
    assert all(cell.source.strip() for cell in notebook.cells if cell.cell_type == "code")
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_final_metric_contracts_are_explicit():
    task3 = _source(nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4))
    assert "Primary development metric for `gender`: pooled five-fold OOF macro-F1" in task3
    assert "Primary development metric for `usage`: pooled five-fold OOF macro-F1" in task3
    stages = [
        "Start with the baseline",
        "Keep the pooling lesson",
        "Reduce sensitivity to small shifts",
        "Add dropout",
        "Repair the dark-image weakness",
        "Reduce colour dependence",
        "Review the labels",
        "Add MixUp",
        "Choose the SAM25 trade-off",
        "Confirm the fixed recipe on five folds",
        "Freeze the full prediction recipe",
        "Read the reserved holdout results",
        "Explain the remaining failures",
    ]
    positions = [task3.index(stage) for stage in stages]
    assert positions == sorted(positions)
    for contract in (
        "0020-task3-gender-sam25-final-model.md",
        "single_explicit_gender_cue_v1",
        "original-label",
        "folds 0 and 4",
        "not a best epoch",
        "teacher did not supply test labels",
        "146 of 311 Unisex",
        "id,gender,articleType,season,usage",
        "0022-task3-usage-e1-final-model.md",
        "original teacher-only E1",
        "equal probability average of its five saved fold models",
        "13,110 teacher images",
        "E1 misses every test NA",
        "final E1 acceptance followed holdout/test review",
    ):
        assert contract in task3
    assert "run_task3_baseline_cv" not in task3
    for internal_result in (
        "61.85",
        "92.02",
        "797 test names",
        "19 of 84 Unisex",
        'figure("matches")',
        'figure("evaluation")',
        "report.final_table",
    ):
        assert internal_result not in task3
    notebook = nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4)
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html", "")
            assert "<td>test</td>" not in html
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_final_report_structure() -> None:
    for filename in ("04_task3_gender_usage.ipynb",):
        spec = TASK_SPECS[filename]
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
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            assert code_cells
            assert max(len(cell.source.splitlines()) for cell in code_cells) <= 25
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    assert notebook.cells[index - 1].cell_type == "markdown"
                    assert "### " in notebook.cells[index - 1].source
            code = "\n".join(cell.source for cell in code_cells)
            assert "from fashion" not in code
            assert "import fashion" not in code
            assert "def " not in code
            assert "class Task3" not in code
            assert "plt.subplots" in code
            assert "analysis_assets.json" in code
            assert "earlier_investigation.ipynb" in source
        else:
            assert all(cell.cell_type == "markdown" for cell in notebook.cells)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        assert [
            int(match.group(1))
            for heading in headings
            if (match := re.fullmatch(r"## (\d+)\. .+", heading))
        ] == list(range(1, spec["sections"] + 1))

        for required in ("data/processed/splits.csv", "results/runs.csv"):
            assert required in source
        if filename == "04_task3_gender_usage.ipynb":
            assert "TODO(owner)" not in source
            assert "0022-task3-usage-e1-final-model.md" in source
        else:
            assert "TODO(owner)" in source
        assert all(token.lower() in lowered for token in spec["tokens"])
        assert "train_test_split" not in source
        assert "pretrained=True" not in source
        assert "Final metric selected: yes" not in source
        if filename != "04_task3_gender_usage.ipynb":
            for unselected in ("macro-F1", "nDCG@", "Recall@", "Adam", "cross-entropy"):
                assert unselected not in source
