from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task4 import report_figures

CHART_NAMES = (
    "method_quality_comparison.png",
    "stability_folds.png",
    "gallery_policy_comparison.png",
    "cost_quality_tradeoff.png",
    "failure_slice_profile.png",
    "canvas_robustness.png",
)
RETRIEVAL_NAMES = (
    "r5_retrieval_success.png",
    "r5_retrieval_failure.png",
    "r5_retrieval_slices.png",
)


def test_report_figure_names_carry_captions_that_name_source_and_transformation() -> None:
    assert report_figures.REPORT_FIGURE_NAMES == CHART_NAMES + RETRIEVAL_NAMES
    assert report_figures.FIGURE_DIR_RELATIVE == Path("results/figures/task4/final")

    for name in report_figures.REPORT_FIGURE_NAMES:
        caption = report_figures.FIGURE_CAPTIONS[name]
        assert "Source:" in caption, name
        assert any(
            word in caption
            for word in ("mean", "p95", "normalised", "normalized", "sample SD")
        ), name


def test_resolve_image_rows_refuses_sealed_partitions_and_missing_ids() -> None:
    catalogue = pd.DataFrame(
        {
            "id": [1, 2],
            "partition": ["development", "holdout"],
            "articleType": ["Tshirts", "Tshirts"],
            "baseColour": ["Red", "Red"],
            "product_family_group": ["family_a", "family_b"],
            "external_path": ["a.jpg", "b.jpg"],
            "teacher_path": ["a.jpg", "b.jpg"],
        }
    )
    with pytest.raises(ValueError, match="sealed"):
        report_figures.resolve_image_rows(catalogue, [1, 2])
    with pytest.raises(ValueError, match="missing"):
        report_figures.resolve_image_rows(catalogue, [1, 99])
    kept = report_figures.resolve_image_rows(catalogue, [1])
    assert kept["id"].tolist() == [1]


def test_graded_relevance_marks_exact_partial_and_incorrect_results() -> None:
    query = pd.Series(
        {
            "id": 1,
            "articleType": "Tshirts",
            "baseColour": "Red",
            "product_family_group": "family_a",
        }
    )
    candidates = pd.DataFrame(
        {
            "id": [2, 3, 4],
            "articleType": ["Tshirts", "Tshirts", "Handbags"],
            "baseColour": ["Red", "Blue", "Red"],
            "product_family_group": ["family_a", "family_c", "family_d"],
        }
    )
    grades = report_figures.graded_relevance(query, candidates)
    assert grades.tolist() == [2, 1, 0]
    families = report_figures.same_family(query, candidates)
    assert families.tolist() == [True, False, False]


def test_winner_retrieval_panels_are_deterministic_and_development_only() -> None:
    evidence = report_figures.load_winner_retrieval_evidence(root=ROOT)
    assert evidence.method == "R5"
    assert evidence.run_id == "task4-candidate-r5-task9-preexec"

    first = report_figures.select_retrieval_panels(evidence, kind="success")
    second = report_figures.select_retrieval_panels(evidence, kind="success")
    failures = report_figures.select_retrieval_panels(evidence, kind="failure")

    assert [panel.query_id for panel in first] == [panel.query_id for panel in second]
    assert len(first) == len(failures) == report_figures.PANEL_COUNT
    assert all(len(panel.results) == report_figures.RETRIEVAL_TOP_K for panel in first)
    assert {panel.query_variant for panel in first} == {"clean"}
    assert all(panel.metric == "ndcg_at_10" for panel in first)

    success_scores = [panel.value for panel in first]
    failure_scores = [panel.value for panel in failures]
    assert min(success_scores) > max(failure_scores)
    assert len({panel.article_type for panel in first}) == len(first)
    assert any(result.grade == 2 for panel in first for result in panel.results)
    assert any(result.grade == 0 for panel in failures for result in panel.results)

    catalogue = report_figures.development_catalogue(root=ROOT)
    assert set(catalogue["partition"]) == {"development"}
    for panel in (*first, *failures):
        ids = [panel.query_id, *(result.candidate_id for result in panel.results)]
        assert not report_figures.resolve_image_rows(catalogue, ids).empty


def test_slice_panels_cover_the_pre_registered_failure_slices() -> None:
    evidence = report_figures.load_winner_retrieval_evidence(root=ROOT)
    panels = report_figures.slice_retrieval_panels(evidence)
    labels = [panel.label for panel in panels]

    assert labels == [
        "normal_success",
        "grayscale",
        "unusual_geometry",
        "canvas_failure",
        "weak_family",
        "family_unavailable",
        "rare_article_type",
        "rare_type_colour",
    ]
    assert all(len(panel.results) == 5 for panel in panels)
    variants = {panel.label: panel.query_variant for panel in panels}
    assert variants["canvas_failure"] == "wide"
    assert variants["normal_success"] == "clean"


def test_chart_figures_are_regenerable_and_byte_identical(tmp_path: Path) -> None:
    first = report_figures.build_chart_figures(root=ROOT, destination=tmp_path / "a")
    second = report_figures.build_chart_figures(root=ROOT, destination=tmp_path / "b")

    assert sorted(first) == sorted(CHART_NAMES)
    for name in CHART_NAMES:
        assert first[name].read_bytes() == second[name].read_bytes(), name
        assert first[name].stat().st_size > 5_000, name


def test_every_cited_report_figure_is_present_and_git_trackable() -> None:
    for name in report_figures.REPORT_FIGURE_NAMES:
        relative = report_figures.FIGURE_DIR_RELATIVE / name
        assert (ROOT / relative).is_file(), relative
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", str(relative)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert ignored.returncode == 1, ignored.stdout
