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
        "sections": 32,
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
        "04d_task3_tinyresnet18_pm_e4_experiments.ipynb",
        "04e_task3_compactblurcnn_label_smoothing_e5_experiments.ipynb",
        "04f_task3_gem_focal_e6_experiments.ipynb",
        "04g_task3_tinyconvnext_tinyhrnet_e7_experiments.ipynb",
        "04h_task3_early_stopping_translation_e8_experiments.ipynb",
        "04i_task3_semantic_filter_exception_balance_e9_experiments.ipynb",
        "04j_task3_audience_aux_e10_experiment.ipynb",
        "04k_task3_clean_slate_eda.ipynb",
        "04l_task3_clean_slate_screen_1.ipynb",
        "04m_task3_micro_swin_clean_slate_screen_2.ipynb",
        "04n_task3_gem_gender_v2_g1_foreground_mask.ipynb",
        "04o_task3_gem_gender_v2_g2_translation.ipynb",
        "04p_task3_gem_gender_v2_g3_component_weight.ipynb",
        "04q_task3_smallcnn_usage_v2_u1_component_weight.ipynb",
        "04r_task3_gem_gender_v2_g2_confirmation.ipynb",
        "04s_task3_usage_v2_u2_full_rgb_hog_svm.ipynb",
        "04t_task3_gender_gd1_mild_darkening.ipynb",
        *TASK_SPECS,
    }
    present = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert present == allowed


def test_task3_clean_slate_eda_is_separate_and_label_safe() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/04k_task3_clean_slate_eda.ipynb", as_version=4
    )
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
        ROOT / "notebooks/04m_task3_micro_swin_clean_slate_screen_2.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    source = _source(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — Scratch Micro-Swin Clean-Slate Screen 2"
    assert "Run All starts four fits" in source
    assert "check_micro_swin_screen_setup" in code
    assert code.count("run_micro_swin_screen(") == 2
    assert "device_name=\"cuda\"" in code
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
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_task3_e4_runner_only_trains_tinyresnet_children() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks/04d_task3_tinyresnet18_pm_e4_experiments.ipynb",
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
        ROOT / "notebooks/04e_task3_compactblurcnn_label_smoothing_e5_experiments.ipynb",
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
        ROOT / "notebooks/04f_task3_gem_focal_e6_experiments.ipynb",
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
        ROOT / "notebooks/04g_task3_tinyconvnext_tinyhrnet_e7_experiments.ipynb",
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
        ROOT / "notebooks/04h_task3_early_stopping_translation_e8_experiments.ipynb",
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
        ROOT / "notebooks/04i_task3_semantic_filter_exception_balance_e9_experiments.ipynb",
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
        ROOT / "notebooks/04j_task3_audience_aux_e10_experiment.ipynb",
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
                "t3-e3-results-load",
                "t3-e3-comparison",
                "t3-e3-bootstrap",
                "t3-e3-gates",
                "t3-e3-curves",
                "t3-e3-diagnostics",
                "t3-e3-failure-gallery",
                "t3-e4-results-load",
                "t3-e4-comparison",
                "t3-e4-bootstrap",
                "t3-e4-gates",
                "t3-e4-curves",
                "t3-e4-diagnostics",
                "t3-e4-failure-gallery",
                "t3-e5-results-load",
                "t3-e5-comparison",
                "t3-e5-bootstrap",
                "t3-e5-gates",
                "t3-e5-curves",
                "t3-e5-diagnostics",
                "t3-e5-failure-gallery",
                "t3-e6-load",
                "t3-e6-tables",
                "t3-e6-plots",
                "t3-e7-load",
                "t3-e7-tables",
                "t3-e7-gates",
                "t3-e7-plots",
                "t3-e7-failure-gallery",
                "sc-e8-analysis",
                "sc-e8-plots",
                "t3-e9-prerun-audits",
                "t3-e9-prerun-plots",
                "t3-e9-deterministic-gate",
                "t3-e9-results-load",
                "t3-e9-gates",
                "t3-e9-plots",
                "t3-e10-results-load",
                "t3-e10-gates",
                "t3-e10-plots",
                "t3-v2-results-load",
                "t3-v2-results-figures",
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
    assert task3.count("keep_default_na=False") >= 6
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
    assert "Loaded completed E3 evidence and accepted parents from Drive" in task3
    assert "paired_family_bootstrap_macro_f1" in task3
    assert "E3 result: reject both children under the frozen rules" in task3
    assert "0.7081" in task3
    assert "0.4161" in task3
    assert "e3_learning_curves.png" in task3
    assert "e3_finished_model_gap.png" in task3
    assert "primary and paired gates failed" in task3
    assert "E4 hypotheses — parameter-matched TinyResNet-18-PM" in task3
    assert "394,865" in task3
    assert "395,253" in task3
    assert "94,268,640" in task3
    assert "94,269,024" in task3
    assert "paired product-family bootstrap lower 95% bound" in task3
    assert "If any gate fails, stop and retain Gender E1 or Usage E2" in task3
    assert "E4 result: reject TinyResNet for both targets" in task3
    assert "Registry check: ten completed E4 folds" in task3
    assert "e4_paired_family_bootstrap" in task3
    assert "0.7121" in task3
    assert "0.3878" in task3
    assert "−0.0400 to −0.0034" in task3
    assert "2.5688 ms against 1.00 ms" in task3
    assert "e4_learning_curves.png" in task3
    assert "e4_fold_macro_f1.png" in task3
    assert "e4_per_class_f1_comparison.png" in task3
    assert "e4_normalised_confusion.png" in task3
    assert "e4_robustness_comparison.png" in task3
    assert "e4_finished_model_gap.png" in task3
    assert "e4_{target}_high_confidence_errors.png" in task3
    assert "E5 hypotheses — test the remaining causes directly" in task3
    assert "Gender E5 hypothesis — CompactBlurCNN" in task3
    assert "67,069 trainable parameters" in task3
    assert "does **not** prove that every architecture change is exhausted" in task3
    assert "Usage E5 hypothesis — class-balanced label smoothing" in task3
    assert "E2 accuracy is 0.6843 versus 0.8996" in task3
    assert "wrong predictions at confidence ≥ 0.99" in task3
    assert "E6 hypotheses — isolate pooling and hard-example focus" in task3
    assert "Gender E6 — fixed GeM pooling with `p=3`" in task3
    assert "Usage E6 — class-balanced focal loss with `gamma=1`" in task3
    assert "E6 result: Gender improves; Usage focal loss is rejected" in task3
    assert "0.7335" in task3
    assert "0.4094" in task3
    assert "e6_bootstrap" in task3
    assert "E7 result: reject both scratch architectures" in task3
    assert "0.7025" in task3
    assert "0.3451" in task3
    assert "e7_learning_curves.png" in task3
    assert "e7_fold_macro_f1.png" in task3
    assert "e7_per_class_f1_comparison.png" in task3
    assert "e7_normalised_confusion.png" in task3
    assert "e7_robustness_comparison.png" in task3
    assert "e7_finished_model_gap.png" in task3
    assert "e7_confidence_diagnostics.png" in task3
    assert "E8 hypotheses — checkpoint selection and small translation" in task3
    assert "Gender E8G — E6 GeM with best-checkpoint early stopping" in task3
    assert "Usage E8U — E2 plus two-pixel random translation" in task3
    assert "96.8% of those wrong disagreement cases" in task3
    assert "Do not add the ArticleType head" in task3
    assert "2.0 GiB PyTorch peak-allocation ceiling per fold" in task3
    assert "04h_task3_early_stopping_translation_e8_experiments.ipynb` resolves" in task3
    assert "E9 pre-run audits and conditional hypotheses" in task3
    assert "optional qualitative review" not in task3
    assert "The rule finds **305** development rows" in task3
    assert "242, 224, 259, 263, and 232" in task3
    assert "No human-rating gate is used" in task3
    assert "97.21% of 1,614 wrong exception rows" in task3
    assert "fewer than 179 clean child-to-adult errors" in task3
    assert "exception error below 45.05%" in task3
    assert "wrong-exception shortcut below 92.21%" in task3
    assert "mixed-label-family accuracy above 0.6743" in task3
    assert "GENDER_E9_APPROVED" not in task3
    assert "e9_prerun_audit_summary.png" in task3
    assert "training remains stopped" in task3
    assert "04i_task3_semantic_filter_exception_balance_e9_experiments.ipynb`" in task3
    assert "E10 result: reject the auxiliary audience head" in task3
    assert "Loaded completed E10 evidence from" in task3
    assert "0.7269" in task3
    assert "only fold 3 improves" in task3
    assert "fails 10" in task3
    assert "e10_learning_curves.png" in task3
    assert "e10_fold_and_class_f1.png" in task3
    assert "e10_mechanism_and_overfit.png" in task3
    assert "e10_robustness.png" in task3
    assert "e6_learning_curves.png" in task3
    assert "e6_per_class_f1_comparison.png" in task3
    assert "E7 hypotheses — change the feature representation" in task3
    assert "Gender E7 — TinyHRNet-20" in task3
    assert "Usage E7 — TinyConvNeXt-18" in task3
    assert "374,445 parameters" in task3
    assert "384,345 parameters" in task3
    assert "Run Usage TinyConvNeXt-18 first" in task3
    assert "mean Boys/Girls/Unisex F1 at least 0.6013" in task3
    assert "mixed-family NLL at most 1.6475" in task3
    assert "Casual 0.9218, Ethnic 0.8440" in task3
    assert "if TinyConvNeXt clearly beats E2" in task3
    assert "04g_task3_tinyconvnext_tinyhrnet_e7_experiments.ipynb` trains Usage" in task3
    assert "reject: worse pooled, paired, fold, minority" in task3
    assert "strongest observed score, but reject as finalist" in task3
    assert "reject: paired, primary, NA, and dark-image no-harm gates fail" in task3
    assert "04f_task3_gem_focal_e6_experiments.ipynb` loads E1/E2" in task3
    assert "results/evidence/task3" in task3
    assert "04a_task3_smallcnn_baseline_training.ipynb` reproduces only" in task3
    assert "04b_task3_smallcnn_child_experiments.ipynb` loads those saved parents" in task3
    assert "04c_task3_smallcnn_e3_experiments.ipynb` loads the accepted E1/E2 parents" in task3
    assert (
        "04d_task3_tinyresnet18_pm_e4_experiments.ipynb` loads the same accepted parents" in task3
    )
    assert "04e_task3_compactblurcnn_label_smoothing_e5_experiments.ipynb` loads E1/E2" in task3
    assert "run_task3_baseline_cv" not in task3

    notebook = nbformat.read(ROOT / "notebooks/04_task3_gender_usage.ipynb", as_version=4)
    child_cells = "\n".join(
        cell.source for cell in notebook.cells if cell.id.startswith("t3-child-")
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
