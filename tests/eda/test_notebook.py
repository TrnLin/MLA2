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

NOTEBOOK = ROOT / "notebooks/00_eda.ipynb"
FIGURE_NAMES = {
    "article_type_long_tail.png",
    "bias_risk.png",
    "data_quality_examples.png",
    "development_balance.png",
    "hierarchy_overlap.png",
    "image_profile.png",
    "joint_target_relationships.png",
    "near_duplicate_review.png",
    "product_name_review.png",
    "season_ambiguity.png",
    "target_distributions.png",
    "task4_protocol.png",
    "transform_comparison.png",
    "variant_alignment_examples.png",
}
REQUIRED_HEADINGS = (
    "# Exploratory Data Analysis",
    "## Executive summary",
    "## 1. Scope and execution",
    "## 2. Data provenance and inventory",
    "### 2.1 Prepared-data contract",
    "### 2.2 Dataset inventory",
    "## 3. Data integrity and leakage controls",
    "### 3.1 Split integrity",
    "### 3.2 Duplicate and entity controls",
    "## 4. Target distributions and class balance",
    "## 5. Target relationships and bias",
    "## 6. Image data quality and preprocessing",
    "### 6.1 Image quality and resolution variants",
    "### 6.2 Input transformation",
    "## 7. Retrieval evaluation",
    "## 8. Reproducibility and limitations",
    "## 9. Modelling recommendations",
    "## 10. Final assessment",
)
EVIDENCE_NAMES = {
    "bias_summary.csv",
    "decision_table.csv",
    "duplicate_summary.csv",
    "family_policy_summary.csv",
    "hierarchy_summary.csv",
    "high_resolution_summary.csv",
    "image_summary.csv",
    "joint_target_nmi.csv",
    "normalization_summary.csv",
    "partition_summary.csv",
    "processed_inventory.csv",
    "provenance.json",
    "raw_inventory.csv",
    "review_summary.csv",
    "season_family_basis_summary.csv",
    "summary.json",
    "target_summary.csv",
    "task4_protocol.json",
    "task4_summary.csv",
    "taxonomy_exclusions.csv",
    "taxonomy_summary.csv",
    "transform_benchmark.json",
    "transform_summary.csv",
    "validation_summary.csv",
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
    shutil.copy2(NOTEBOOK, destination / "notebooks/00_eda.ipynb")
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
    notebook_path = project_root / "notebooks/00_eda.ipynb"
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
            "FASHION_EDA_NOTEBOOK_PATH": str(notebook_path),
            "FASHION_EDA_WORKERS": "2",
            "FASHION_EDA_BENCHMARK_SIZE": "4",
            "FASHION_EDA_BENCHMARK_REPETITIONS": "1",
        }
    )
    if rebuild:
        environment["FASHION_EDA_MODE"] = "full"
    else:
        environment.pop("FASHION_EDA_MODE", None)
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
            "00_eda.html",
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
    summary = json.loads(
        (project_root / "results/evidence/eda/summary.json").read_text(encoding="utf-8")
    )
    return summary, nbformat.read(notebook_path, as_version=4)


def _hashes_from_summary(summary: dict) -> dict[str, str]:
    return {record["path"]: record["sha256"] for record in summary["deterministic_outputs"]}


def test_the_official_notebook_contains_the_complete_workflow() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert sorted((ROOT / "notebooks").glob("*.ipynb")) == [NOTEBOOK]
    assert all(heading in source for heading in REQUIRED_HEADINGS)
    assert all("hide-input" in cell.metadata.get("tags", []) for cell in code_cells)
    assert all(cell.metadata.get("jupyter", {}).get("source_hidden") for cell in code_cells)
    technical_cells = [
        cell for cell in code_cells if "report-hide" in cell.metadata.get("tags", [])
    ]
    assert len(technical_cells) == 7
    assert max(len(cell.source.splitlines()) for cell in technical_cells) < 350
    assert "**Question.**" not in markdown
    assert "?" not in markdown
    assert source.count("> **Finding.**") == 10
    assert source.count("> **Modelling consequence.**") == 10
    assert all(source.count(f"Figure {number}. ") == 1 for number in range(1, 15))
    assert "fashion.data.pipeline import prepare_data" in source
    assert "validate_prepared_data_cache" in source
    assert 'PREPARATION_MODE = "cached"' in source
    assert "include_high_resolution_variants=True" in source
    assert "paired_normalization.json" in source
    assert "normalization_original_only.json" in source
    assert "all_alignment_pairs.csv.gz" in source
    assert "taxonomy.json" in source
    assert "build_development_retrieval_variant_sets" in source
    assert "load_splits(SPLITS_PATH)" in source
    assert "keep_default_na=False" in source
    assert '"NA"' in source
    assert "pending_team_signoff" in source
    assert "fashion.eda" not in source
    assert "load_splits_for_final_evaluation" not in source
    assert "raw_train = pd.read_csv" not in source
    assert 'usecols=["id"]' in source
    assert "raw_teacher_target_columns_read_by_eda" in source
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
        "prediction_images_in_target_eda": 0,
        "prediction_role": "label-free duplicate audit only",
        "protected_partitions": ["holdout", "quarantine"],
        "raw_teacher_target_columns_read_by_eda": 0,
    }
    assert first_summary["leakage_checks"]["protected_target_fields_visible"] == 0
    assert first_summary["leakage_checks"]["raw_teacher_target_columns_read_by_eda"] == 0
    assert first_summary["leakage_checks"]["prediction_ids_in_labelled_splits"] == 0
    assert first_summary["leakage_checks"]["prediction_ids_in_variant_manifest"] == 0
    assert first_summary["leakage_checks"]["quarantine_ids_in_variant_manifest"] == 0
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
    assert first_summary["task4"]["recall_at_k"]["relevant_grades"] == [2]
    assert (
        first_summary["task4"]["recall_at_k"]["zero_positive_query_rule"]
        == "exclude from macro Recall@K and report in coverage"
    )
    assert all(row["signed_rows"] == 0 for row in first_summary["reviews"])
    assert first_summary["image_variants"]["policy"] == (
        "complete_low_high_pairs_in_train_val_holdout"
    )
    assert first_summary["normalization"]["default_policy"] == (
        "pair_weighted_original_and_high_resolution"
    )
    assert first_summary["image_variants"]["alignment"]["human_signoff"] == "pending"
    assert first_summary["execution"]["preparation_mode"] == "full"
    for partition, counts in first_summary["image_variants"]["partition_coverage"].items():
        assert counts["variant_count"] == 2 * counts["product_count"], partition
        assert counts["complete_pair_count"] == counts["product_count"], partition
        assert counts["incomplete_pair_count"] == 0, partition

    for root, notebook in (
        (first_root, first_notebook),
        (second_root, second_notebook),
    ):
        figures = root / "results/figures/eda"
        assert {path.name for path in figures.glob("*.png")} == FIGURE_NAMES
        assert all(path.stat().st_size > 1_000 for path in figures.glob("*.png"))
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells)
        assert sum(bool(cell.outputs) for cell in code_cells) >= 10
        final_text = json.dumps(code_cells[-1].outputs)
        assert "Run All complete" in final_text
        assert "Protected target values materialised by EDA: 0" in final_text

        report = (root / "results/notebooks/00_eda.html").read_text(encoding="utf-8")
        assert not re.search(r'class="[^"]*jp-InputArea-editor', report)
        assert report.count('<figure class="eda-figure">') == len(FIGURE_NAMES)
        assert all(heading.lstrip("# ") in report for heading in REQUIRED_HEADINGS)
        assert "The next helpers keep tables" not in report

        contact_sheet = root / "results/reviews/review_contact_sheet.html"
        assert contact_sheet.is_file()
        assert "Blind image-review contact sheet" in contact_sheet.read_text(encoding="utf-8")

        saved_summary = root / "results/evidence/eda/summary.json"
        assert hashlib.sha256(saved_summary.read_bytes()).hexdigest()
        evidence = root / "results/evidence/eda"
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
    shutil.copy2(NOTEBOOK, sentinel_root / "notebooks/00_eda.ipynb")
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
    for relative in ("pyproject.toml", "notebooks/00_eda.ipynb", "data/raw/README.md"):
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

    report = ROOT / "results/notebooks/00_eda.html"
    assert report.is_file()
    document = report.read_text(encoding="utf-8")
    assert not re.search(r'class="[^"]*jp-InputArea-editor', document)
    assert document.count('<figure class="eda-figure">') == len(FIGURE_NAMES)
    assert all(heading.lstrip("# ") in document for heading in REQUIRED_HEADINGS)
    assert "The next helpers keep tables" not in document

    summary = json.loads((ROOT / "results/evidence/eda/summary.json").read_text(encoding="utf-8"))
    assert summary["task4"]["recall_at_k"]["queries_with_zero_grade2_positives"] == 110
    assert summary["task4"]["recall_at_k"]["queries_with_fewer_than_k_grade2_positives"] == 412
    assert summary["task4"]["total_validation_products"] == 5_781
    assert summary["task4"]["eligible_supported_query_products"] == 5_768
    assert summary["task4"]["excluded_unsupported_query_products"] == 13
    assert summary["task4"]["total_validation_products"] == (
        summary["task4"]["eligible_supported_query_products"]
        + summary["task4"]["excluded_unsupported_query_products"]
    )
    assert summary["image_variants"]["model_product_count"] == 38_551
    assert summary["image_variants"]["model_variant_count"] == 77_102
    assert not (ROOT / "results/evidence/eda/data_reconciliation.json").exists()


def test_delivered_prepared_and_evidence_pack_is_lean() -> None:
    relative_paths = {
        *_BASE_ARTIFACTS,
        *_HIGH_RESOLUTION_ARTIFACTS,
        f"data/processed/{CACHE_FILENAME}",
    }
    prepared = [ROOT / relative for relative in sorted(relative_paths)]
    evidence = [path for path in (ROOT / "results/evidence/eda").iterdir() if path.is_file()]
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
