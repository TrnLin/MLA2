import ast
import subprocess

from fashion.config import ROOT
from fashion.data.pipeline import _BASE_ARTIFACTS, CACHE_FILENAME


def test_assignment_breakdown_uses_canonical_data_contract():
    document = (ROOT / "docs/assignment-breakdown.html").read_text(encoding="utf-8")

    for stale_claim in (
        "01_preprocessing.ipynb",
        "10_task1_articletype.ipynb",
        "20_task2_season.ipynb",
        "30_task3_gender_usage.ipynb",
        "40_task4_visual_search.ipynb",
        "90_final_predictions.ipynb",
        "clean.csv",
        "train | val | test",
        "pixels, every image",
        "Every image is 4,800 pixels",
        "official EDA",
        "protected EDA",
        "Macro-F1",
        "nDCG@5",
        "Recall@5",
    ):
        assert stale_claim not in document

    for current_claim in (
        "notebooks/01_data_preparation.ipynb",
        "00_problem_definition.ipynb",
        "02_task1_article_type.ipynb",
        "03_task2_season.ipynb",
        "04_task3_gender_usage.ipynb",
        "05_task4_visual_search.ipynb",
        "06_final_evaluation.ipynb",
        "data/processed/splits.csv",
        "train | val | holdout | quarantine",
        "38,595",
        "17 at other resolutions",
        "124 official outputs",
        "27,009 family units",
        "18 have only one",
        "real-world similarity ground truth",
        "Owner TODO after baseline evidence",
        "The Task 4 owner must choose the final cutoff",
    ):
        assert current_claim in document


def test_task4_documentation_has_one_isolated_development_contract():
    document = (ROOT / "docs/assignment-breakdown.html").read_text(encoding="utf-8")
    decision = (ROOT / "docs/decisions/0009-task4-retrieval-isolation.md").read_text(
        encoding="utf-8"
    )
    assert "same images used for evaluation" not in document
    assert "validation queries against a train-only" in document
    assert "ID, SHA, and family isolation" in document
    assert "gallery-coverage diagnostics only" in document
    assert "must select\nand freeze the exact ranking-quality" in decision
    assert "not the final\nmetric or image-view fusion choice" in decision
    for fixed_metric in ("nDCG@5", "Recall@5"):
        assert fixed_metric not in document
        assert fixed_metric not in decision


def test_locked_setup_and_single_split_rule_are_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "requirements/constraints-py312.txt" in readme
    assert 'python -m pip install -c requirements/constraints-py312.txt -e ".[dev]"' in readme

    searched_files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    searched_files.append(ROOT / "notebooks/01_data_preparation.ipynb")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in searched_files
        if "train_test_split" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert not (ROOT / "scripts/prepare_data.py").exists()


def test_delivered_prepared_cache_is_not_ignored():
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


def test_runtime_code_cannot_read_or_unseal_the_protected_split_directly():
    """Only the data access boundary may unseal protected partition targets."""
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
        if (
            "load_splits_for_final_evaluation" in text
            or "_load_unredacted_splits" in text
            or "_normalise_manifest_frame" in text
        ):
            unlock_calls.append(path.relative_to(ROOT).as_posix())

    assert direct_split_reads == []
    assert unlock_calls == []

    dataset_source = (ROOT / "src/fashion/data/dataset.py").read_text(encoding="utf-8")
    assert "raw_teacher_csv" in dataset_source
    assert "evaluation_unlocked" in dataset_source
    assert 'usecols=["id", *TARGET_COLUMNS]' in dataset_source
    assert "redact_protected_targets: bool" not in dataset_source
