from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pandas as pd
from PIL import Image

from fashion.config import ROOT
from fashion.data.pipeline import (
    _BASE_ARTIFACTS,
    _HIGH_RESOLUTION_ARTIFACTS,
    CACHE_FILENAME,
    refresh_protected_safe_tabular_artifacts,
)

PROBLEM_NOTEBOOK = ROOT / "notebooks/00_problem_definition.ipynb"
NOTEBOOK = ROOT / "notebooks/01_data_preparation.ipynb"
ALLOWED_TASK_NOTEBOOKS = {
    ROOT / "notebooks/02_task1_article_type.ipynb",
    ROOT / "notebooks/03_task2_season.ipynb",
    ROOT / "notebooks/04_task3_gender_usage.ipynb",
    ROOT / "notebooks/05_task4_visual_search.ipynb",
    ROOT / "notebooks/06_final_evaluation.ipynb",
}
FIGURE_NAMES = {
    "article_type_long_tail.png",
    "bias_risk.png",
    "data_quality_examples.png",
    "development_balance.png",
    "hierarchy_overlap.png",
    "image_profile.png",
    "image_quality_correlation.png",
    "joint_target_relationships.png",
    "label_image_sanity.png",
    "near_duplicate_boundary.png",
    "family_policy.png",
    "season_ambiguity.png",
    "target_distributions.png",
    "task4_protocol.png",
    "train_image_quality.png",
    "transform_comparison.png",
    "variant_alignment_examples.png",
}
REQUIRED_HEADINGS = (
    "# Data Preparation and Shared Dataset Analysis",
    "## Executive summary",
    "## 1. Scope, contracts, and execution",
    "### 1.1 Notebook boundary",
    "### 1.2 Shared implementation setup",
    "### 1.3 Execution modes and prepared-data contract",
    "## 2. Raw inventory, hashing, and reconciliation",
    "### 2.1 Source and prepared-contract inventory",
    "### 2.2 Raw SHA-256, decode order, and CSV-to-image reconciliation",
    "## 3. Duplicate control, canonical split, and leakage safety",
    "### 3.1 Canonical split and train-only statistics",
    "### 3.2 Label-image semantic sanity check",
    "### 3.3 Exact, perceptual, and family duplicate controls",
    "## 4. Train-only targets and imbalance",
    "### 4.1 Class support and development balance",
    "### 4.2 Imbalance handoff to task owners",
    "## 5. Train-only target association and shortcut risk",
    "### 5.1 Categorical target association",
    "### 5.2 Hierarchy, ambiguity, and shortcut slices",
    "## 6. Shared image quality and preprocessing",
    "### 6.1 Image properties and paired-variant alignment",
    "### 6.2 Numeric image-feature correlation",
    "### 6.3 Common transformation contract",
    "## 7. Task 4 data boundary and retrieval handoff",
    "### 7.1 Query and gallery isolation",
    "### 7.2 Relevance proxy and metric-selection boundary",
    "## 8. Reproducibility, provenance, and limitations",
    "### 8.1 Shared decision record",
    "### 8.2 Saved evidence and provenance",
    "## 9. Task notebook handoff",
    "### 9.1 Fixed shared contracts",
    "### 9.2 Decisions left to task owners",
    "## 10. Data-preparation completion gate",
)
PROBLEM_REQUIRED_HEADINGS = (
    "# Fashion Intelligence: Problem Definition",
    "## Executive summary",
    "## 1. Real-world context and users",
    "### 1.1 User needs",
    "### 1.2 Decisions supported",
    "## 2. Problem statement and prediction unit",
    "### 2.1 Prediction unit",
    "### 2.2 Inputs available at prediction time",
    "### 2.3 Outputs",
    "## 3. Machine-learning task definitions",
    "### 3.1 Task 1 - Fashion item type classification",
    "### 3.2 Task 2 - Fashion season classification",
    "### 3.3 Task 3 - Gender and usage classification",
    "### 3.4 Task 4 - Fashion visual search",
    "## 4. Data roles and evaluation boundary",
    "## 5. Success criteria and evaluation framework",
    "### 5.1 Classification metric-selection criteria",
    "### 5.2 Retrieval metric-selection criteria",
    "### 5.3 Final judgement criteria",
    "## 6. Constraints and non-goals",
    "## 7. Assumptions, risks, and failure costs",
    "## 8. End-to-end system design",
    "## 9. Deliverables and notebook handoff",
    "## 10. Problem-definition completion gate",
)
EVIDENCE_NAMES = {
    "bias_summary.csv",
    "decision_table.csv",
    "duplicate_summary.csv",
    "family_basis_summary.csv",
    "family_policy_summary.csv",
    "family_size_distribution.csv",
    "hierarchy_summary.csv",
    "high_resolution_summary.csv",
    "image_reconciliation.csv",
    "image_quality_spearman.csv",
    "image_summary.csv",
    "imbalance_handoff.csv",
    "joint_target_nmi.csv",
    "label_image_sanity.csv",
    "normalization_summary.csv",
    "partition_summary.csv",
    "preparation_stage_order.csv",
    "processed_inventory.csv",
    "provenance.json",
    "raw_hashing_summary.csv",
    "raw_inventory.csv",
    "season_family_basis_summary.csv",
    "support_threshold_sensitivity.csv",
    "summary.json",
    "target_summary.csv",
    "task4_protocol.json",
    "task4_summary.csv",
    "taxonomy_exclusions.csv",
    "taxonomy_summary.csv",
    "train_image_quality_sample.csv",
    "train_image_quality_summary.csv",
    "transform_benchmark.json",
    "transform_summary.csv",
    "validation_summary.csv",
    "validation_family_coverage.csv",
    "variant_alignment_summary.csv",
}


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def _write_high_resolution_fixture(root: Path) -> None:
    dataset = root / "data/fashion-dataset"
    images = dataset / "images"
    styles = dataset / "styles"
    images.mkdir(parents=True)
    styles.mkdir(parents=True)
    train = pd.read_csv(
        root / "data/raw/teacher/train/styles_train.csv",
        keep_default_na=False,
    )
    prediction_ids = pd.read_csv(
        root / "data/raw/teacher/test/styles_prediction.csv",
        usecols=["id"],
    )["id"].astype(int)
    metadata = train[
        [
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
            "productDisplayName",
        ]
    ].copy()
    prediction_metadata = pd.DataFrame(
        {
            "id": prediction_ids,
            "gender": "",
            "masterCategory": "",
            "subCategory": "",
            "articleType": "",
            "baseColour": "",
            "season": "",
            "year": "",
            "usage": "",
            "productDisplayName": "",
        }
    )
    metadata = pd.concat([metadata, prediction_metadata], ignore_index=True)
    links = []
    for item_id in metadata["id"].astype(int):
        size = (170, 210) if item_id == 3 else (120, 160)
        Image.new(
            "RGB",
            size,
            color=((item_id * 17) % 255, (item_id * 53) % 255, (item_id * 91) % 255),
        ).save(images / f"{item_id}.jpg", "JPEG", quality=95)
        links.append({"filename": f"{item_id}.jpg", "link": f"https://example.test/{item_id}"})
        (styles / f"{item_id}.json").write_text(
            json.dumps({"meta": {"code": 200}, "data": {"id": item_id}}),
            encoding="utf-8",
        )
    metadata.to_csv(dataset / "styles.csv", index=False)
    pd.DataFrame(links).to_csv(dataset / "images.csv", index=False)


def _copy_tiny_project(source_root: Path, destination: Path) -> Path:
    shutil.copytree(source_root / "data/raw", destination / "data/raw")
    shutil.copytree(
        ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    _write_high_resolution_fixture(destination)
    (destination / "notebooks").mkdir(parents=True)
    shutil.copy2(NOTEBOOK, destination / "notebooks/01_data_preparation.ipynb")
    return destination


def _copy_delivered_prepared_artifacts(source_root: Path, destination: Path) -> None:
    relative_paths = (
        *_BASE_ARTIFACTS,
        *_HIGH_RESOLUTION_ARTIFACTS,
        "data/processed/splits.csv",
        f"data/processed/{CACHE_FILENAME}",
    )
    for relative in relative_paths:
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _git_deliverable_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    return {path.decode("utf-8") for path in completed.stdout.split(b"\0") if path}


def _read_public_text(path: Path) -> str:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    return path.read_text(encoding="utf-8")


def _execute_notebook(
    project_root: Path,
    *,
    rebuild: bool = True,
) -> tuple[dict, nbformat.NotebookNode]:
    notebook_path = project_root / "notebooks/01_data_preparation.ipynb"
    if rebuild:
        assert not (project_root / "data/processed").exists()
        assert not (project_root / "results").exists()
    environment = os.environ.copy()
    python_path = [str(project_root / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "FASHION_PROJECT_ROOT": str(project_root),
            "FASHION_DATA_PREPARATION_NOTEBOOK_PATH": str(notebook_path),
            "FASHION_DATA_PREPARATION_WORKERS": "2",
            "FASHION_DATA_PREPARATION_BENCHMARK_SIZE": "4",
            "FASHION_DATA_PREPARATION_BENCHMARK_REPETITIONS": "1",
        }
    )
    if rebuild:
        environment["FASHION_DATA_PREPARATION_MODE"] = "full"
    else:
        environment.pop("FASHION_DATA_PREPARATION_MODE", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(notebook_path),
        ],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--no-input",
            "--TagRemovePreprocessor.enabled=True",
            "--TagRemovePreprocessor.remove_cell_tags=report-hide",
            "--output",
            "01_data_preparation.html",
            "--output-dir",
            str(project_root / "results/notebooks"),
            str(notebook_path),
        ],
        check=True,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    summary_path = project_root / "results/evidence/data_preparation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, nbformat.read(notebook_path, as_version=4)


def _hashes_from_summary(summary: dict) -> dict[str, str]:
    return {record["path"]: record["sha256"] for record in summary["deterministic_outputs"]}


def test_problem_definition_notebook_is_complete_and_data_independent() -> None:
    notebook = nbformat.read(PROBLEM_NOTEBOOK, as_version=4)
    source = _source(notebook)

    assert notebook.cells
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert all(heading in source for heading in PROBLEM_REQUIRED_HEADINGS)
    heading_positions = [source.index(heading) for heading in PROBLEM_REQUIRED_HEADINGS]
    assert heading_positions == sorted(heading_positions)
    assert all(source.count(heading) == 1 for heading in PROBLEM_REQUIRED_HEADINGS)
    assert "articleType" in source
    assert "season" in source
    assert "gender" in source
    assert "usage" in source
    assert "task owner" in source
    assert "Primary development metric:" not in source
    assert "Ranking quality: nDCG" not in source
    assert "data/processed/splits.csv" in source
    assert "trained from scratch" in source


def test_the_official_notebook_contains_the_complete_workflow() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    normalized_markdown = " ".join(markdown.split())
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    code_cell_indices = [
        index for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
    ]
    guide_cells = [cell for cell in notebook.cells if "code-guide" in cell.metadata.get("tags", [])]

    assert notebook.metadata["title"] == "Data Preparation and Shared Dataset Analysis"
    notebook_paths = set((ROOT / "notebooks").glob("*.ipynb"))
    assert {PROBLEM_NOTEBOOK, NOTEBOOK}.issubset(notebook_paths)
    assert notebook_paths <= {PROBLEM_NOTEBOOK, NOTEBOOK, *ALLOWED_TASK_NOTEBOOKS}
    assert all(heading in source for heading in REQUIRED_HEADINGS)
    assert len(code_cells) == 38
    assert len(guide_cells) == len(code_cells)
    assert all(
        notebook.cells[index - 1].cell_type == "markdown"
        and notebook.cells[index - 1].source.startswith("**Code step — ")
        and "code-guide" in notebook.cells[index - 1].metadata.get("tags", [])
        and "report-hide" in notebook.cells[index - 1].metadata.get("tags", [])
        for index in code_cell_indices
    )
    assert all("hide-input" in cell.metadata.get("tags", []) for cell in code_cells)
    assert all(cell.metadata.get("jupyter", {}).get("source_hidden") for cell in code_cells)
    technical_cells = [
        cell for cell in code_cells if "report-hide" in cell.metadata.get("tags", [])
    ]
    assert technical_cells
    assert max(len(cell.source.splitlines()) for cell in code_cells) < 150
    assert "**Question.**" not in markdown
    assert "?" not in markdown
    assert source.count("> **Finding.**") == 14
    assert source.count("> **Modelling consequence.**") == 14
    assert all(source.count(f"Figure {number}. ") == 1 for number in range(1, 18))
    assert all(
        count in normalized_markdown
        for count in (
            "53,984 training",
            "11,562 validation",
            "11,556 holdout",
            "26,992, 5,781, and 5,778 unique products",
        )
    )
    assert "Product counts describe reporting units" in markdown
    assert "image-input counts describe the actual loader size" in markdown
    provenance = json.loads(
        (ROOT / "results/evidence/data_preparation/provenance.json").read_text(encoding="utf-8")
    )
    stable_count = len(provenance["deterministic_outputs"])
    figure_count = sum(
        record["path"].startswith("results/figures/data_preparation/")
        and record["path"].endswith(".png")
        for record in provenance["deterministic_outputs"]
    )
    saved_count_sentence = (
        f"The notebook records {stable_count} stable evidence files, "
        f"including all {figure_count} report figures,"
    )
    assert saved_count_sentence in normalized_markdown
    rendered_report = " ".join(
        (ROOT / "results/notebooks/01_data_preparation.html").read_text(encoding="utf-8").split()
    )
    assert saved_count_sentence in rendered_report
    assert "fashion.data.pipeline import prepare_data" in source
    assert "validate_prepared_data_cache" in source
    assert 'PREPARATION_MODE = "cached"' in source
    assert "include_high_resolution_variants=True" in source
    assert "Raw inventory and SHA-256" in source
    assert "Perceptual dHash and aHash" in source
    assert "metadata_without_decoded_image" in source
    assert "label_image_sanity.csv" in source
    assert "automatic_label_changes" in source
    assert 'method="spearman"' in source
    assert "Correlation alone does not" in source
    assert "TODO in task notebook after baseline evidence" in source
    assert "never resample or rebalance" in source
    assert '"fixed_method": False' in source
    assert "paired_normalization.json" in source
    assert "normalization_original_only.json" in source
    assert "all_alignment_pairs.csv.gz" in source
    assert "taxonomy.json" in source
    assert "build_development_retrieval_variant_sets" in source
    assert '"metrics":' not in source
    assert "nDCG@5" not in source
    assert "Recall@5" not in source
    assert '"final_metric_selected": False' in source
    assert "TODO in Task 4 notebook after baseline evidence" in source
    assert "load_splits(SPLITS_PATH)" in source
    assert "keep_default_na=False" in source
    assert '"NA"' in source
    forbidden_review_workflow = {
        "pending_team_signoff",
        "signoff_status",
        "signed_off",
        "reviewer_initials",
        "review_date",
        "review_summary.csv",
        "review_contact_sheet",
    }
    assert all(token not in source for token in forbidden_review_workflow)
    assert "fashion.eda" not in source
    assert "load_splits_for_final_evaluation" not in source
    assert "raw_train = pd.read_csv" not in source
    assert 'usecols=["id"]' in source
    assert "raw_teacher_target_columns_read_by_data_preparation" in source
    assert "def atomic_write_text" in source
    assert "temporary.replace(path)" in source
    assert "this notebook does not train or select a model" in normalized_markdown
    assert all(name in source for name in FIGURE_NAMES)

    for cell in code_cells:
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr == "read_csv":
                assert not (node.args and "SPLITS_PATH" in ast.unparse(node.args[0])), (
                    "the notebook must use the protected split loader"
                )
                if node.args and "RAW_TRAIN_CSV" in ast.unparse(node.args[0]):
                    keywords = {
                        keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords
                    }
                    assert (
                        keywords.get("usecols") in {'["id"]', "['id']"}
                        or keywords.get("nrows") == "0"
                    )


def test_old_eda_modules_and_entry_points_are_retired() -> None:
    old_scripts = {
        "audit_perceptual_duplicates.py",
        "benchmark_transforms.py",
        "generate_eda.py",
        "generate_review_contact_sheet.py",
        "render_eda_notebook.py",
        "verify_eda_provenance.py",
        "prepare_data.py",
    }
    assert old_scripts.isdisjoint(path.name for path in (ROOT / "scripts").glob("*.py"))
    assert not list((ROOT / "src/fashion/eda").glob("*.py"))


def test_notebook_runs_twice_with_stable_outputs(prepared_project, tmp_path: Path) -> None:
    first_root = _copy_tiny_project(prepared_project.root, tmp_path / "first")
    second_root = _copy_tiny_project(prepared_project.root, tmp_path / "second")

    first_summary, first_notebook = _execute_notebook(first_root)
    second_summary, second_notebook = _execute_notebook(second_root)

    assert _hashes_from_summary(first_summary) == _hashes_from_summary(second_summary)
    assert first_summary["scope"] == {
        "development_partition": "val",
        "modelling_partition": "train",
        "prediction_images_in_target_analysis": 0,
        "prediction_role": "label-free duplicate audit only",
        "protected_partitions": ["holdout", "quarantine"],
        "raw_teacher_target_columns_read_by_data_preparation": 0,
    }
    assert first_summary["leakage_checks"]["protected_target_fields_visible"] == 0
    assert (
        first_summary["leakage_checks"]["raw_teacher_target_columns_read_by_data_preparation"]
        == 0
    )
    assert first_summary["leakage_checks"]["prediction_ids_in_labelled_splits"] == 0
    assert first_summary["leakage_checks"]["prediction_ids_in_variant_manifest"] == 0
    assert first_summary["leakage_checks"]["quarantine_ids_in_variant_manifest"] == 0
    assert [row["order"] for row in first_summary["preparation_stage_order"]] == list(
        range(1, 9)
    )
    assert first_summary["preparation_stage_order"][0]["stage"] == (
        "Raw inventory and SHA-256"
    )
    assert all(
        row["decoded_files"] == row["files_with_raw_sha256"]
        for row in first_summary["raw_hashing"]
    )
    assert all(
        row["unique_raw_sha256_values"] <= row["files_with_raw_sha256"]
        for row in first_summary["raw_hashing"]
    )
    assert all(
        row["decoded_image_without_metadata"] == 0
        for row in first_summary["image_reconciliation"]
    )
    assert first_summary["label_image_sanity"]["source_partition"] == "train"
    assert first_summary["label_image_sanity"]["automatic_label_changes"] == 0
    assert first_summary["image_feature_correlation"]["source_partition"] == "train"
    assert first_summary["image_feature_correlation"]["method"] == "spearman"
    assert first_summary["image_feature_correlation"]["causal_claim"] is False
    assert first_summary["imbalance_handoff"]["source_partition"] == "train"
    assert first_summary["imbalance_handoff"]["fixed_method"] is False
    assert first_summary["imbalance_handoff"]["task_owner_choice_required"] is True
    assert first_summary["imbalance_handoff"]["validation_holdout_rebalanced"] is False
    assert first_summary["imbalance_handoff"]["raw_pixel_smote"] is False
    assert first_summary["task4"]["shared_ids"] == 0
    assert first_summary["task4"]["shared_sha256"] == 0
    assert first_summary["task4"]["shared_product_families"] == 0
    assert first_summary["task4"]["total_validation_products"] == (
        first_summary["task4"]["eligible_supported_query_products"]
        + first_summary["task4"]["excluded_unsupported_query_products"]
    )
    assert first_summary["task4"]["query_exclusion_reason"] == (
        "articleType is outside the train-fitted supported slice"
    )
    assert (
        first_summary["task4"]["query_image_variants"]
        == 2 * first_summary["task4"]["query_products"]
    )
    assert (
        first_summary["task4"]["gallery_image_variants"]
        == 2 * first_summary["task4"]["gallery_products"]
    )
    task4_coverage = first_summary["task4"]["strict_relevance_coverage_diagnostic"]
    assert task4_coverage["diagnostic_k"] == 5
    assert task4_coverage["strict_relevance_grade"] == 2
    assert first_summary["task4"]["metric_selection"]["final_metric_selected"] is False
    assert first_summary["task4"]["metric_selection"]["owner"] == "Task 4 teammate"
    assert first_summary["task4"]["metric_selection"]["status"] == (
        "TODO in Task 4 notebook after baseline evidence"
    )
    assert "metrics" not in first_summary["task4"]
    assert "reviews" not in first_summary
    assert first_summary["image_variants"]["policy"] == (
        "complete_low_high_pairs_in_train_val_holdout"
    )
    assert first_summary["normalization"]["default_policy"] == (
        "pair_weighted_original_and_high_resolution"
    )
    assert first_summary["image_variants"]["alignment"]["blocks_preparation"] is False
    assert "human_signoff" not in first_summary["image_variants"]["alignment"]
    assert first_summary["family_policy"]["open_decisions_block_execution"] is False
    assert first_summary["task4"]["evidence_kind"] == "metadata_proxy"
    assert first_summary["execution"]["preparation_mode"] == "full"
    for partition, counts in first_summary["image_variants"]["partition_coverage"].items():
        assert counts["variant_count"] == 2 * counts["product_count"], partition
        assert counts["complete_pair_count"] == counts["product_count"], partition
        assert counts["incomplete_pair_count"] == 0, partition
    for partition, inventory_name in (
        ("train", "train"),
        ("val", "validation"),
        ("holdout", "holdout"),
    ):
        counts = first_summary["image_variants"]["partition_coverage"][partition]
        assert (
            first_summary["inventory"][f"{inventory_name}_product_ids"] == counts["product_count"]
        )
        assert (
            first_summary["inventory"][f"{inventory_name}_image_inputs"] == counts["variant_count"]
        )
    assert first_summary["inventory"]["quarantine_image_inputs"] == 0
    assert first_summary["inventory"]["target_distribution_unit"] == "unique product ID"
    assert first_summary["inventory"]["model_loader_unit"] == "image input row"

    for root, notebook in (
        (first_root, first_notebook),
        (second_root, second_notebook),
    ):
        figures = root / "results/figures/data_preparation"
        assert {path.name for path in figures.glob("*.png")} == FIGURE_NAMES
        assert all(path.stat().st_size > 1_000 for path in figures.glob("*.png"))
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells)
        assert sum(bool(cell.outputs) for cell in code_cells) >= 10
        assert not any(
            output.output_type == "execute_result" for cell in code_cells for output in cell.outputs
        )
        final_text = json.dumps(code_cells[-1].outputs)
        assert "Run All complete" in final_text
        assert "Protected target values materialised by data preparation: 0" in final_text

        report = (root / "results/notebooks/01_data_preparation.html").read_text(encoding="utf-8")
        assert not re.search(r'class="[^"]*jp-InputArea-editor', report)
        assert report.count('<figure class="data-preparation-figure">') == len(FIGURE_NAMES)
        assert all(heading.lstrip("# ") in report for heading in REQUIRED_HEADINGS)
        assert "Code step —" not in report
        assert "The next helpers keep tables" not in report

        assert not list((root / "results/reviews").glob("*"))

        saved_summary = root / "results/evidence/data_preparation/summary.json"
        assert hashlib.sha256(saved_summary.read_bytes()).hexdigest()
        evidence = root / "results/evidence/data_preparation"
        assert {path.name for path in evidence.iterdir()} == EVIDENCE_NAMES
        assert not (evidence / "data_reconciliation.json").exists()
        variants = pd.read_csv(root / "data/processed/training_image_variants.csv.gz")
        assert variants.groupby("id")["variant"].agg(set).eq({"original", "high_resolution"}).all()
        assert variants.groupby("id")["per_product_weight"].sum().eq(1).all()
        assert not set(variants["id"]).intersection({101, 102})
        paired = json.loads(
            (root / "data/processed/paired_normalization.json").read_text(encoding="utf-8")
        )
        assert (
            paired["variant_manifest_sha256"]
            == hashlib.sha256(
                (root / "data/processed/training_image_variants.csv.gz").read_bytes()
            ).hexdigest()
        )
        taxonomy = json.loads((root / "data/processed/taxonomy.json").read_text(encoding="utf-8"))
        assert taxonomy["fit_partition"] == "train"
        provenance = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
        assert str(root) not in json.dumps(provenance)


def test_protected_target_sentinels_do_not_change_normal_eda_state(
    prepared_project,
    tmp_path: Path,
) -> None:
    baseline_root = _copy_tiny_project(prepared_project.root, tmp_path / "baseline")
    baseline_summary, _ = _execute_notebook(baseline_root)
    prepared_paths = (
        *_BASE_ARTIFACTS,
        *_HIGH_RESOLUTION_ARTIFACTS,
        f"data/processed/{CACHE_FILENAME}",
    )
    baseline_prepared_hashes = {
        relative: hashlib.sha256((baseline_root / relative).read_bytes()).hexdigest()
        for relative in prepared_paths
    }

    sentinel_root = tmp_path / "sentinel"
    shutil.copytree(baseline_root, sentinel_root)
    shutil.copy2(NOTEBOOK, sentinel_root / "notebooks/01_data_preparation.ipynb")
    shutil.rmtree(sentinel_root / "results")

    splits_path = sentinel_root / "data/processed/splits.csv"
    splits = pd.read_csv(splits_path, keep_default_na=False)
    protected = splits["partition"].isin({"holdout", "quarantine"})
    protected_ids = set(splits.loc[protected, "id"].astype(int))
    for target in ("articleType", "season", "gender", "usage"):
        for column in (target, f"{target}_deployed", f"{target}_supported"):
            splits.loc[protected, column] = f"SENTINEL_{target}"
        for column in (
            f"has_{target}_label",
            f"has_{target}_deployed_label",
            f"has_{target}_supported_label",
        ):
            splits.loc[protected, column] = True
    splits.to_csv(splits_path, index=False)

    raw_path = sentinel_root / "data/raw/teacher/train/styles_train.csv"
    raw = pd.read_csv(raw_path, keep_default_na=False)
    for target in ("articleType", "season", "gender", "usage"):
        raw.loc[raw["id"].astype(int).isin(protected_ids), target] = f"SENTINEL_{target}"
    raw.to_csv(raw_path, index=False)

    # These files are deliberately made stale. The refresh must rebuild them; a cache-only
    # shortcut would leave the markers behind and fail the equality checks below.
    for relative in (
        "data/processed/audit/csv_summary.json",
        "data/processed/audit/missing_values.csv",
        "data/processed/audit/target_class_counts.csv",
        "data/processed/development_class_summary.csv",
        "data/processed/label_maps.json",
        "data/processed/preparation_cache.json",
        "data/processed/split_summary.json",
        "data/processed/taxonomy.json",
    ):
        (sentinel_root / relative).write_text("STALE_SENTINEL\n", encoding="utf-8")

    refresh_protected_safe_tabular_artifacts(
        root=sentinel_root,
        include_high_resolution_variants=True,
    )

    sentinel_summary, sentinel_notebook = _execute_notebook(sentinel_root, rebuild=False)
    assert _hashes_from_summary(sentinel_summary) == _hashes_from_summary(baseline_summary)
    assert {
        relative: hashlib.sha256((sentinel_root / relative).read_bytes()).hexdigest()
        for relative in prepared_paths
    } == baseline_prepared_hashes
    saved_state = json.dumps(sentinel_summary) + json.dumps(sentinel_notebook)
    assert "SENTINEL_" not in saved_state
    for relative in prepared_paths:
        path = sentinel_root / relative
        if path.suffix in {".json", ".csv", ".gz"}:
            assert "SENTINEL_" not in _read_public_text(path)
    for path in (sentinel_root / "results").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".html", ".ipynb"}:
            assert "SENTINEL_" not in path.read_text(encoding="utf-8")


def test_clean_delivered_package_runs_once_in_default_cached_mode(
    prepared_project,
    tmp_path: Path,
) -> None:
    built_root = _copy_tiny_project(prepared_project.root, tmp_path / "built")
    built_summary, _ = _execute_notebook(built_root)

    package_root = tmp_path / "delivered"
    package_root.mkdir()
    for relative in ("pyproject.toml", "notebooks/01_data_preparation.ipynb", "data/raw/README.md"):
        source = ROOT / relative
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(
        ROOT / "src",
        package_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    shutil.copytree(
        built_root / "data/raw/teacher",
        package_root / "data/raw/teacher",
        copy_function=os.link,
    )
    shutil.copytree(
        built_root / "data/fashion-dataset",
        package_root / "data/fashion-dataset",
        copy_function=os.link,
    )
    _copy_delivered_prepared_artifacts(built_root, package_root)
    assert not (package_root / "results").exists()
    deliverable_paths = _git_deliverable_paths()
    local_roots = ("data/raw/teacher/", "data/fashion-dataset/")
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if not relative.startswith(local_roots):
            assert relative in deliverable_paths
    raw_source = next((built_root / "data/raw/teacher/train/images_train").glob("*.jpg"))
    raw_link = package_root / raw_source.relative_to(built_root)
    assert os.path.samefile(raw_source, raw_link)

    started = time.perf_counter()
    packaged_summary, packaged_notebook = _execute_notebook(package_root, rebuild=False)
    elapsed = time.perf_counter() - started
    assert elapsed < 300
    assert packaged_summary["execution"]["preparation_mode"] == "cached"
    assert packaged_summary["execution"]["cache_validation"]["status"] == "validated"
    assert _hashes_from_summary(packaged_summary) == _hashes_from_summary(built_summary)
    assert all(
        cell.execution_count is not None
        for cell in packaged_notebook.cells
        if cell.cell_type == "code"
    )


def test_saved_notebook_and_html_are_report_ready() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert sum(bool(cell.outputs) for cell in code_cells) >= 10
    assert not any(
        output.output_type == "execute_result" for cell in code_cells for output in cell.outputs
    )

    report = ROOT / "results/notebooks/01_data_preparation.html"
    assert report.is_file()
    document = report.read_text(encoding="utf-8")
    assert "<title>Data Preparation and Shared Dataset Analysis</title>" in document
    assert not re.search(r'class="[^"]*jp-InputArea-editor', document)
    assert document.count('<figure class="data-preparation-figure">') == len(FIGURE_NAMES)
    assert all(heading.lstrip("# ") in document for heading in REQUIRED_HEADINGS)
    assert "Code step —" not in document
    assert "PosixPath(" not in document
    assert "The next helpers keep tables" not in document

    summary_path = ROOT / "results/evidence/data_preparation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task4_coverage = summary["task4"]["strict_relevance_coverage_diagnostic"]
    assert task4_coverage["queries_with_zero_strict_positives"] == 110
    assert task4_coverage["queries_with_fewer_than_diagnostic_k_strict_positives"] == 412
    assert summary["task4"]["metric_selection"]["final_metric_selected"] is False
    assert summary["task4"]["total_validation_products"] == 5_781
    assert summary["task4"]["eligible_supported_query_products"] == 5_768
    assert summary["task4"]["excluded_unsupported_query_products"] == 13
    assert summary["task4"]["total_validation_products"] == (
        summary["task4"]["eligible_supported_query_products"]
        + summary["task4"]["excluded_unsupported_query_products"]
    )
    assert summary["image_variants"]["model_product_count"] == 38_551
    assert summary["image_variants"]["model_variant_count"] == 77_102
    assert summary["inventory"]["train_product_ids"] == 26_992
    assert summary["inventory"]["train_image_inputs"] == 53_984
    assert summary["inventory"]["validation_product_ids"] == 5_781
    assert summary["inventory"]["validation_image_inputs"] == 11_562
    assert summary["inventory"]["holdout_product_ids"] == 5_778
    assert summary["inventory"]["holdout_image_inputs"] == 11_556
    assert summary["inventory"]["quarantine_product_ids"] == 61
    assert summary["inventory"]["quarantine_image_inputs"] == 0
    family_policy = pd.read_csv(
        ROOT / "results/evidence/data_preparation/family_policy_summary.csv"
    )
    family_values = family_policy.set_index("measure")["value"]
    assert int(family_values["active product rows"]) == 38_551
    assert int(family_values["independent family units"]) == 27_009
    assert int(family_values["units removed by blocking"]) == 11_542
    assert int(family_values["multi-row families"]) == 4_567
    assert int(family_values["largest family"]) == 80
    validation = pd.read_csv(
        ROOT / "results/evidence/data_preparation/validation_summary.csv"
    ).set_index("target")
    assert int(validation.loc["articleType", "classes_with_one_val_family"]) == 18
    assert int(
        validation.loc["articleType", "classes_with_fewer_than_three_val_families"]
    ) == 26
    assert int(validation.loc["usage", "classes_with_two_val_families"]) == 1
    sensitivity = pd.read_csv(
        ROOT / "results/evidence/data_preparation/support_threshold_sensitivity.csv"
    )
    article_sensitivity = sensitivity[sensitivity["target"].eq("articleType")].set_index(
        "minimum_train_families"
    )
    assert article_sensitivity["supported_classes"].to_dict() == {
        2: 105,
        3: 100,
        4: 98,
        5: 94,
    }
    label_image_sanity = pd.read_csv(
        ROOT / "results/evidence/data_preparation/label_image_sanity.csv",
        keep_default_na=False,
    )
    assert len(label_image_sanity) == 12
    assert label_image_sanity["source_partition"].eq("train").all()
    assert {"id", "articleType", "season", "gender", "usage", "sample_reason"}.issubset(
        label_image_sanity.columns
    )
    image_quality_spearman = pd.read_csv(
        ROOT / "results/evidence/data_preparation/image_quality_spearman.csv"
    )
    assert len(image_quality_spearman) == 36
    correlation_matrix = image_quality_spearman.pivot(
        index="feature_x", columns="feature_y", values="spearman_rho"
    )
    assert correlation_matrix.shape == (6, 6)
    assert (correlation_matrix.values.diagonal() == 1).all()
    assert correlation_matrix.equals(correlation_matrix.T)
    imbalance_handoff = pd.read_csv(
        ROOT / "results/evidence/data_preparation/imbalance_handoff.csv",
        keep_default_na=False,
    )
    assert set(imbalance_handoff["target"]) == {"articleType", "season", "gender", "usage"}
    assert imbalance_handoff["task_owner_decision"].eq(
        "TODO in task notebook after baseline evidence"
    ).all()
    assert imbalance_handoff["validation_holdout_policy"].eq(
        "never resample or rebalance"
    ).all()
    assert imbalance_handoff["raw_pixel_smote"].eq("not used").all()
    quality = pd.read_csv(ROOT / "results/evidence/data_preparation/train_image_quality_sample.csv")
    assert len(quality) == 2_048
    assert quality["source_partition"].eq("train").all()
    assert {
        "brightness_mean",
        "contrast_std",
        "laplacian_variance",
        "foreground_fraction",
        "foreground_box_fraction",
        "background_fraction",
    }.issubset(quality.columns)
    partition_summary = pd.read_csv(
        ROOT / "results/evidence/data_preparation/partition_summary.csv"
    )
    counts = partition_summary.set_index("partition")
    assert int(counts.loc["train", "image_inputs"]) == 53_984
    assert int(counts.loc["val", "image_inputs"]) == 11_562
    assert int(counts.loc["holdout", "image_inputs"]) == 11_556
    assert int(counts.loc["quarantine", "image_inputs"]) == 0
    assert "53,984 training" in document
    assert "11,562 validation" in document
    assert "11,556 holdout" in document
    assert "metadata proxy" in document
    assert "real-world similarity ground truth" in document
    provenance = json.loads(
        (ROOT / "results/evidence/data_preparation/provenance.json").read_text(encoding="utf-8")
    )
    provenance_paths = {row["path"] for row in provenance["inputs"]}
    assert {
        "data/processed/audit/issues.csv",
        "data/processed/audit/product_family_summary.json",
        "data/processed/high_resolution/catalogue.json",
        "data/processed/high_resolution/image_catalogue.csv.gz",
    }.issubset(provenance_paths)
    assert all(not Path(path).is_absolute() for path in provenance_paths)
    assert sorted(path.name for path in (ROOT / "docs/reviews").iterdir()) == [
        "open_decisions.md"
    ]
    assert not list((ROOT / "results/reviews").glob("*"))
    assert not (ROOT / "results/evidence/data_preparation/data_reconciliation.json").exists()


def test_delivered_prepared_and_evidence_pack_is_lean() -> None:
    relative_paths = {
        *_BASE_ARTIFACTS,
        *_HIGH_RESOLUTION_ARTIFACTS,
        f"data/processed/{CACHE_FILENAME}",
    }
    prepared = [ROOT / relative for relative in sorted(relative_paths)]
    evidence = [
        path
        for path in (ROOT / "results/evidence/data_preparation").iterdir()
        if path.is_file()
    ]
    assert all(path.is_file() for path in prepared)
    total_bytes = sum(path.stat().st_size for path in [*prepared, *evidence])
    assert total_bytes < 50 * 1024 * 1024
    largest = max(prepared, key=lambda path: path.stat().st_size)
    assert largest.name == "splits.csv"
    assert largest.stat().st_size < 30 * 1024 * 1024
    assert not (ROOT / "data/processed/train_manifest.csv").exists()
    assert not (ROOT / "data/processed/train_manifest.csv.gz").exists()
    for old_name in (
        "training_image_variants.csv",
        "audit/image_audit.csv",
        "audit/near_duplicate_candidates.csv",
        "audit/perceptual_hashes.csv",
        "audit/product_family_groups.csv",
        "high_resolution/image_catalogue.csv",
        "high_resolution/all_alignment_pairs.csv",
    ):
        assert not (ROOT / "data/processed" / old_name).exists()
