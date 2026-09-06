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


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


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
        *TASK_SPECS,
    }
    present = {
        path.relative_to(ROOT / "notebooks").as_posix()
        for path in (ROOT / "notebooks").rglob("*.ipynb")
    }
    assert present == allowed


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


def test_task_metric_contracts_are_explicit() -> None:
    for filename in ("02_task1_article_type.ipynb", "03_task2_season.ipynb"):
        source = _source(nbformat.read(ROOT / "notebooks" / filename, as_version=4))
        assert "Primary development metric: TODO(owner)" in source

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

    task4 = _source(nbformat.read(ROOT / "notebooks/05_task4_visual_search.ipynb", as_version=4))
    assert "Primary ranking-quality metric: TODO(owner)" in task4
    assert "Cutoff, averaging, zero-positive rule, and tie-break: TODO(owner)" in task4
