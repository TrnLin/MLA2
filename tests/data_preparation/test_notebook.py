from __future__ import annotations

import hashlib
import json
import re

import nbformat
import pandas as pd
import pytest

from fashion.config import RANDOM_SEED, ROOT, TARGET_COLUMNS
from fashion.data.evidence import family_profile, fold_support_tables, shortcut_benchmarks
from fashion.data.pipeline import validate_prepared_data_cache
from fashion.data.splits import cv_assignment_digest

PROBLEM_NOTEBOOK = ROOT / "notebooks/00_problem_definition.ipynb"
NOTEBOOK = ROOT / "notebooks/01_data_preparation.ipynb"
HTML = ROOT / "results/notebooks/01_data_preparation.html"

DATA_HEADINGS = (
    "# 01 — Shared Data Preparation",
    "## 1. Runtime setup",
    "## 2. Input/output map and run modes",
    "## 3. Holdout boundary",
    "## 4. Discover teacher sources",
    "## 5. Hash raw bytes before image decode",
    "## 6. Image integrity and exact ID reconciliation",
    "## 7. Exact duplicates, perceptual candidates, families, and quarantine",
    "## 8. Development, internal holdout, quarantine, and five CV folds",
    "## 9. Development-only target analysis",
    "## 10. Imbalance and class support by fold",
    "## 11. Shortcut risk and categorical association",
    "## 12. Development image profile and pixel diagnostics",
    "## 13. Labelled development contact sheet",
    "## 14. Duplicate, missing-image, and quality-extreme examples",
    "## 15. Transform-risk illustration",
    "## 16. Artifact registry and lineage",
    "## 17. Using the prepared data for model development",
    "## 18. Completion gate",
)

EXPECTED_COUNTS = {"development": 32773, "holdout": 5778, "quarantine": 61}
EXPECTED_DIGESTS = {
    "development": "2639d731f89942fd9598f9552bb092dd634830c3e6a63d0b569cfc2aa4362d01",
    "holdout": "48a9e2e389657744d2771a3ab5ca34e85e015b21b18ef4d0ed29c2e648cf8fc0",
    "quarantine": "5a672e34e485c8ddad0cef37c2dc37d99f8774f8874279a5d0ed69aff7022f43",
}
EXPECTED_CV_DIGEST = "bad7bc4ae65fbbfd815567f4ccfa308d6e57dc650bc15c0b8e798867a335f2fd"
REDUNDANT_GRAPH_TABLES = (
    "article_type_gender_shares.csv",
    "article_type_season_shares.csv",
    "article_type_usage_shares.csv",
    "class_support_points.csv",
    "development_family_profile.csv",
    "development_family_sizes.csv",
    "family_source_profile.csv",
    "metadata_pattern_audit.csv",
    "near_threshold_review.csv",
    "rare_class_fold_support.csv",
    "shortcut_majority_benchmarks.csv",
    "untrainable_fold_flags.csv",
)
HELPER_FILE_BY_CODE_FRAGMENT = (
    ("from fashion.config import", "src/fashion/config.py"),
    ("from fashion.data.audit import", "src/fashion/data/audit.py"),
    ("from fashion.data.dataset import", "src/fashion/data/dataset.py"),
    ("from fashion.data.evidence import", "src/fashion/data/evidence.py"),
    ("from fashion.data.hashing import", "src/fashion/data/hashing.py"),
    ("from fashion.data.pipeline import", "src/fashion/data/pipeline.py"),
    ("from fashion.data.splits import", "src/fashion/data/splits.py"),
    ("cache_status = validate_prepared_data_cache(ROOT)", "src/fashion/data/pipeline.py"),
    ("splits = load_splits(ROOT /", "src/fashion/data/dataset.py"),
    ("audit_source = inspect.getsource(audit_image)", "src/fashion/data/audit.py"),
    ("audit_source = inspect.getsource(audit_image)", "src/fashion/data/hashing.py"),
    ("family_sizes, development_family_profile", "src/fashion/data/evidence.py"),
    ("threshold_review = near_threshold_review", "src/fashion/data/evidence.py"),
    ("current_cv_digest = cv_assignment_digest(splits)", "src/fashion/data/splits.py"),
    ("for fold, training, validation in iter_cv_folds(splits):", "src/fashion/data/dataset.py"),
    ("support_points, rare_fold_counts", "src/fashion/data/evidence.py"),
    ("shortcut_summary = shortcut_benchmarks(development)", "src/fashion/data/evidence.py"),
    ("target: article_target_heatmap(development", "src/fashion/data/evidence.py"),
)


def _source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def _headings(notebook: nbformat.NotebookNode) -> list[str]:
    return [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]


def _id_digest(ids: pd.Series) -> str:
    payload = "".join(f"{item_id}\n" for item_id in sorted(ids.astype(int)))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_problem_definition_is_narrative_only() -> None:
    notebook = nbformat.read(PROBLEM_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    source = _source(notebook).lower()

    assert notebook.metadata["title"] == "Fashion Intelligence: Problem Definition"
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert [h for h in _headings(notebook) if h.startswith("## ")] == [
        f"## {index}. {title}"
        for index, title in enumerate(
            (
                "Executive summary",
                "Real-world users and decisions supported",
                "Problem statement and prediction unit",
                "Assignment tasks",
                "Inputs available at prediction time",
                "Required outputs",
                "Evaluation principles and independent evidence",
                "Success dimensions and ownership",
                "Constraints",
                "Assumptions, risks, and failure costs",
                "Non-goals",
                "Deliverables and notebook handoff",
                "Problem-definition readiness gate",
            ),
            start=1,
        )
    ]
    for empirical_claim in ("we achieved", "our accuracy", "best model", "final metric"):
        assert empirical_claim not in source


def test_data_preparation_notebook_is_valid_executed_and_narrative() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)

    assert notebook.metadata["title"] == "Shared Data Preparation"
    assert _headings(notebook) == list(DATA_HEADINGS)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)

    code_indexes = [i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code"]
    assert len(code_indexes) == 28
    for index in code_indexes:
        cell = notebook.cells[index]
        assert cell.execution_count is not None
        assert cell.outputs
        assert not any(output.output_type == "error" for output in cell.outputs)
        assert index + 1 < len(notebook.cells)
        assert notebook.cells[index + 1].cell_type == "markdown"
        assert re.match(
            r"\*\*(Finding|Definition only|Completion result)\.",
            notebook.cells[index + 1].source,
        )


def test_notebook_keeps_shared_scope_teacher_only_and_holdout_sealed() -> None:
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    lowered = source.lower()

    assert "data/raw/teacher" in source
    assert "load_splits_for_final_evaluation" not in source
    assert "evaluation_unlocked=true" not in lowered
    for retired in (
        "high-resolution",
        "high_resolution",
        "paired view",
        "paired-view",
        "task 4 protocol",
        "global normalization",
        "supported mask",
        "deployed mask",
        "data/fashion-dataset",
        "data/raw/external",
    ):
        assert retired not in lowered

    pipeline_source = (ROOT / "src/fashion/data/pipeline.py").read_text(encoding="utf-8")
    for external_input in ("data/raw/external", "data/fashion-dataset", "images.csv"):
        assert external_input not in pipeline_source


def test_hashing_reconciliation_and_analysis_contracts_are_visible() -> None:
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    lowered = source.lower()

    assert source.index('"compute_sha256(path)"') < source.index('"Image.open(path)"')
    for check in (
        "duplicate IDs in CSV",
        "numeric stems with multiple files",
        "non-numeric filename stems",
        "IDs with multiple extensions",
        "metadata IDs without valid image",
        "extra valid image IDs",
        "unsupported extensions",
        "corrupt supported images",
    ):
        assert check in source
    assert RANDOM_SEED == 2753
    assert "from fashion.config import RANDOM_SEED" in source
    assert "random_state=RANDOM_SEED" in source
    assert "development.sample" in source
    contact_index = pd.read_csv(
        ROOT / "results/evidence/data_preparation/development_contact_sheet_index.csv"
    )
    assert set(TARGET_COLUMNS).issubset(contact_index.columns)
    assert "association" in lowered and "caus" in lowered
    assert "QUALITY_SAMPLE_SIZE" in source
    assert "spearman" in lowered
    assert "allowed_for_model_fit" in source
    assert "family_profile" in source
    assert "fold_support_tables" in source
    assert "shortcut_benchmarks" in source
    assert "near_threshold_review" in source


def test_outside_helper_files_are_named_at_each_call_site() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for fragment, helper_path in HELPER_FILE_BY_CODE_FRAGMENT:
        matching_indexes = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and fragment in cell.source
        ]
        assert matching_indexes, fragment
        for index in matching_indexes:
            assert index > 0
            previous = notebook.cells[index - 1]
            assert previous.cell_type == "markdown"
            assert helper_path in previous.source, (fragment, helper_path)


def test_split_membership_digests_folds_and_protected_cells_are_frozen() -> None:
    splits = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)
    assert splits["partition"].value_counts().to_dict() == EXPECTED_COUNTS

    for partition, expected in EXPECTED_DIGESTS.items():
        rows = splits[splits["partition"].eq(partition)]
        assert _id_digest(rows["id"]) == expected

    development = splits[splits["partition"].eq("development")]
    protected = splits[splits["partition"].isin(["holdout", "quarantine"])]
    assert set(development["cv_fold"].astype(int)) == set(range(5))
    assert protected["cv_fold"].eq("").all()
    for target in TARGET_COLUMNS:
        assert protected[target].eq("").all()
        assert protected[f"has_{target}_label"].astype(str).str.lower().eq("false").all()

    assert development.groupby("product_family_group")["cv_fold"].nunique().max() == 1
    assert splits.groupby("product_family_group")["partition"].nunique().max() == 1
    assert development.groupby("duplicate_group")["cv_fold"].nunique().max() == 1
    assert cv_assignment_digest(splits) == EXPECTED_CV_DIGEST

    raw = pd.read_csv(ROOT / "data/raw/teacher/train/styles_train.csv", keep_default_na=False)
    missing_name_ids = raw.loc[
        raw["productDisplayName"].astype(str).str.strip().str.casefold().eq("na"), "id"
    ]
    missing_names = splits[splits["id"].isin(missing_name_ids)]
    assert len(missing_names) == 7
    assert missing_names["product_name_key"].eq("").all()
    assert missing_names["product_family_group"].nunique() == 7


def test_computed_data_preparation_figures_and_statistics_are_complete() -> None:
    evidence = ROOT / "results/evidence/data_preparation"
    figures = ROOT / "results/figures/data_preparation"
    splits = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)

    _, profile_table, _ = family_profile(splits)
    profile = profile_table.iloc[0]
    assert profile["product_rows"] == 32773
    assert profile["independent_families"] == 22905
    assert profile["active_partition_crossings"] == 0
    assert profile["development_fold_crossings"] == 0
    assert profile["largest_family_products"] == 80

    class_summary = pd.read_csv(
        ROOT / "data/processed/development_class_summary.csv", keep_default_na=False
    )
    support, _, _ = fold_support_tables(class_summary)
    article = support[support["target"].eq("articleType")]
    assert article["rare_warning"].ne("").sum() == 26
    assert pd.to_numeric(article["untrainable_fold_count"]).gt(0).sum() == 12
    home = support[support["target"].eq("usage") & support["class"].eq("Home")]
    assert pd.to_numeric(home["untrainable_fold_count"]).item() == 1

    development = splits[splits["partition"].eq("development")]
    shortcut = shortcut_benchmarks(development).set_index("predicted_target")
    assert shortcut.loc["season", "group_majority_accuracy"] == pytest.approx(0.6505, abs=0.0001)
    assert shortcut.loc["usage", "group_majority_accuracy"] == pytest.approx(0.9004, abs=0.0001)
    assert shortcut.loc["gender", "group_majority_accuracy"] == pytest.approx(0.7854, abs=0.0001)

    spearman = pd.read_csv(evidence / "image_quality_spearman.csv")
    assert not {"width", "height", "aspect_ratio"}.intersection(spearman["feature_1"])
    assert not any((evidence / filename).exists() for filename in REDUNDANT_GRAPH_TABLES)
    for filename in (
        "family_duplicate_profile.png",
        "near_duplicate_threshold_review.png",
        "class_support_by_fold.png",
        "shortcut_risk_heatmaps.png",
        "target_distributions.png",
        "image_quality_and_spearman.png",
        "development_contact_sheet.png",
        "transform_risk.png",
    ):
        assert (figures / filename).is_file()


def test_delivered_cache_and_artifact_registry_match_bytes() -> None:
    result = validate_prepared_data_cache(ROOT)
    assert result["status"] == "validated"
    assert result["shared_source_policy"] == "teacher_only"
    assert result["protected_target_values_hashed"] == 0

    registry = pd.read_csv(ROOT / "results/evidence/data_preparation/artifact_registry.csv")
    assert len(registry) == 44
    for row in registry.itertuples(index=False):
        path = ROOT / row.repository_path
        assert path.is_file(), row.repository_path
        assert path.stat().st_size == row.bytes, row.repository_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256, row.repository_path

    summary = json.loads(
        (ROOT / "results/evidence/data_preparation/summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ready"
    assert summary["model_training_runs"] == 0
    assert summary["protected_target_values_used"] == 0


def test_saved_html_is_current_portable_and_hides_inputs() -> None:
    html = HTML.read_text(encoding="utf-8")
    lowered = html.lower()

    assert "DATA PREPARATION READY" in html
    assert "Shared Data Preparation" in html
    assert html.count("jp-mod-noInput") >= 28
    assert "from __future__ import annotations" not in html
    assert len(re.findall(r"<h2[ >]", html)) == 18
    assert "/home/" not in html
    assert "c:\\" not in lowered
    for retired in (
        "high-resolution",
        "paired normalization",
        "task 4 protocol",
        "global normalization",
        "supported mask",
        "deployed mask",
    ):
        assert retired not in lowered
