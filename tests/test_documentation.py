from __future__ import annotations

import ast
import subprocess

from fashion.config import ROOT
from fashion.data.pipeline import _BASE_ARTIFACTS, CACHE_FILENAME


def test_active_handoff_docs_use_the_current_contract() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "notebooks/README.md",
        ROOT / "data/processed/README.md",
        ROOT / "results/evidence/data_preparation/README.md",
        ROOT / "results/figures/data_preparation/README.md",
        ROOT / "docs/assignment-breakdown.html",
        ROOT / "docs/reviews/open_decisions.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    lowered = combined.lower()

    for current in (
        "data/raw/teacher",
        "data/processed/splits.csv",
        "development",
        "cv_fold",
        "32,773",
        "5,778",
        "quarantine",
        "teacher-only",
    ):
        assert current.lower() in lowered

    for stale in (
        "data/fashion-dataset",
        "paired normalization",
        "paired high-resolution",
        "shared normalization",
        "train | val | holdout",
        "supported mask",
        "deployed mask",
        "validation queries against a train-only",
    ):
        assert stale not in lowered

    assert "22,905 conservative split groups" in combined
    assert "not verified independent products" in lowered
    assert "22,905 independent families" not in lowered
    assert "acquisition_shortcut_risk.png" in combined
    assert "broad_name_family_review.png" in combined


def test_assignment_breakdown_lists_the_canonical_notebooks() -> None:
    document = (ROOT / "docs/assignment-breakdown.html").read_text(encoding="utf-8")
    for name in (
        "00_problem_definition.ipynb",
        "01_data_preparation.ipynb",
        "02_task1_article_type.ipynb",
        "03_task2_season.ipynb",
        "04_task3_gender_usage.ipynb",
        "01_v1_eda.ipynb",
        "05_task4_visual_search.ipynb",
        "06_final_evaluation.ipynb",
    ):
        assert name in document
    for fixed_metric in ("Macro-F1", "nDCG@5", "Recall@5"):
        assert fixed_metric not in document


def test_task4_preprocessing_decision_and_progress_are_frozen() -> None:
    decision_index = (ROOT / "docs/decisions/README.md").read_text(encoding="utf-8")
    decision = (
        ROOT / "docs/decisions/0020-task4-image-preprocessing.md"
    ).read_text(encoding="utf-8")
    progress = (ROOT / "notebooks/task-4/PROGRESS.md").read_text(encoding="utf-8")
    evidence_readme = (
        ROOT / "results/evidence/task4/README.md"
    ).read_text(encoding="utf-8")
    figure_readme = (
        ROOT / "results/figures/task4/README.md"
    ).read_text(encoding="utf-8")
    open_decisions = (ROOT / "docs/reviews/open_decisions.md").read_text(
        encoding="utf-8"
    )

    assert "0020-task4-image-preprocessing.md" in decision_index
    assert "- Status: Accepted" in decision
    assert "holdout" in decision.lower() and "sealed" in decision.lower()
    for completed in (
        "- [x] Compare useful image sizes",
        "- [x] Choose resize, padding, and colour handling",
        "- [x] Define arbitrary query-image handling",
        "- [x] Fit any learned image values on training folds only",
    ):
        assert completed in progress
    for artifact in (
        "preprocessing_comparison.csv",
        "preprocessing_size_selection.csv",
        "preprocessing_stability.csv",
        "preprocessing_robustness.csv",
        "preprocessing_contract.json",
        "preprocessing_normalization_fold1.json",
    ):
        assert artifact in evidence_readme
    assert "preprocessing_comparison.png" in figure_readme
    assert "addition to ADR 0019" in decision
    assert "does not reopen" in decision
    assert "arbitrary-query image policy" not in open_decisions


def test_locked_setup_and_single_split_rule_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "requirements/constraints-py312.txt" in readme
    assert 'python -m pip install -c requirements/constraints-py312.txt -e ".[dev]"' in readme

    searched_files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "notebooks/01_data_preparation.ipynb",
    ]
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in searched_files
        if "train_test_split" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert not (ROOT / "scripts/prepare_data.py").exists()


def test_delivered_prepared_cache_is_not_ignored() -> None:
    required = {
        *_BASE_ARTIFACTS,
        "data/processed/splits.csv",
        f"data/processed/{CACHE_FILENAME}",
    }
    ignored = [
        path
        for path in sorted(required)
        if subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    ]
    assert ignored == []


def test_runtime_code_cannot_read_or_unseal_the_protected_split_directly() -> None:
    runtime_files = [
        *sorted((ROOT / "src/fashion/eda").rglob("*.py")),
        *sorted((ROOT / "src/fashion/retrieval").rglob("*.py")),
    ]
    for optional in (ROOT / "src/fashion/models", ROOT / "src/fashion/train"):
        if optional.exists():
            runtime_files.extend(sorted(optional.rglob("*.py")))

    direct_split_reads = []
    unlock_calls = []
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "read_csv":
                continue
            source = ast.unparse(node.args[0])
            if "SPLITS_CSV" in source or "splits.csv" in source or "['splits']" in source:
                direct_split_reads.append(path.relative_to(ROOT).as_posix())
        if "load_splits_for_final_evaluation" in text or "_load_unredacted_splits" in text:
            unlock_calls.append(path.relative_to(ROOT).as_posix())

    assert direct_split_reads == []
    assert unlock_calls == []

    dataset_source = (ROOT / "src/fashion/data/dataset.py").read_text(encoding="utf-8")
    assert "raw_teacher_csv" in dataset_source
    assert "evaluation_unlocked" in dataset_source
    assert 'usecols=["id", *TARGET_COLUMNS]' in dataset_source
