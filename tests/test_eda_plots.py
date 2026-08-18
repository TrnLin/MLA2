from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.config import PROCESSED_DATA_DIR
import fashion.eda as eda_module
from fashion.eda import EdaPaths, GENERATED_EDA_OUTPUTS, run_eda
from fashion.eda_plots import (
    build_report_summary,
    plot_article_type_support,
    plot_association_matrix,
    plot_drift,
    plot_duplicate_summary,
    plot_image_profiles,
    plot_relationship_heatmap,
    plot_review_grid,
    plot_target_distributions,
)


HEADER = (
    "id,gender,masterCategory,subCategory,articleType,baseColour,"
    "season,year,usage,productDisplayName"
)


def _write_metadata(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([HEADER, *rows]), encoding="utf-8")


def _save_image(path: Path, color: tuple[int, int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        Image.new("L", (8, 6), color[0]).save(path)
    else:
        Image.new("RGB", (8, 6), color).save(path)


@pytest.fixture
def eda_paths(tmp_path: Path) -> EdaPaths:
    teacher_csv = tmp_path / "teacher.csv"
    original_csv = tmp_path / "original.csv"
    test_csv = tmp_path / "test.csv"
    original_images = tmp_path / "original-images"
    lowres_images = tmp_path / "lowres-images"
    rows = [
        "1,Men,Apparel,Topwear,Tshirts,Blue,Summer,2012,Casual,Blue shirt",
        "2,Women,Apparel,Topwear,Tshirts,Red,Winter,2013,Casual,Red shirt",
        "3,Men,Apparel,Bottomwear,Jeans,Black,Spring,2014,Formal,Black jeans",
        "4,Women,Accessories,Bags,Bags,Green,Fall,2015,NA,Green bag",
    ]
    _write_metadata(teacher_csv, rows)
    _write_metadata(original_csv, rows)
    test_csv.write_text("id,gender,articleType,season,usage\n90,,,,\n", encoding="utf-8")
    colors = [(20, 40, 200), (20, 40, 200), (40, 40, 40), (200, 180, 20)]
    for product_id, color in enumerate(colors, start=1):
        _save_image(original_images / f"{product_id}.png", color)
        _save_image(
            lowres_images / f"{product_id}.png",
            color,
            mode="L" if product_id == 3 else "RGB",
        )
    return EdaPaths(
        teacher_train_csv=teacher_csv,
        original_csv=original_csv,
        test_csv=test_csv,
        original_image_dir=original_images,
        lowres_image_dir=lowres_images,
    )


def _plot_tables() -> dict[str, pd.DataFrame]:
    distribution = pd.DataFrame(
        {"label": ["Tshirts", "Jeans"], "count": [3, 1], "share": [0.75, 0.25]}
    )
    return {
        "distributions": {"articleType": distribution, "gender": distribution},
        "support": pd.DataFrame(
            {"band": ["1", "2", "3–4", "5–9", "10+"], "class_count": [1, 0, 1, 0, 0]}
        ),
        "associations": pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]], index=["gender", "usage"], columns=["gender", "usage"]
        ),
        "relationship": pd.DataFrame(
            [[0.75, 0.25], [0.25, 0.75]],
            index=["Men", "Women"],
            columns=["Casual", "Formal"],
        ),
        "drift": pd.DataFrame({"year": ["2012", "2013"], "total_variation": [0.1, 0.2]}),
        "images": pd.DataFrame(
            {
                "brightness_low": [30.0, 40.0],
                "brightness_high": [50.0, 60.0],
                "contrast_low": [3.0, 4.0],
                "contrast_high": [5.0, 6.0],
            }
        ),
        "duplicates": pd.DataFrame({"kind": ["exact", "near"], "count": [1, 2]}),
    }


def test_plot_functions_return_figures_without_showing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notebook callers need composable figures and batch jobs must never open a window."""
    tables = _plot_tables()
    shown = False

    def record_show() -> None:
        nonlocal shown
        shown = True

    monkeypatch.setattr(plt, "show", record_show)
    image = Image.new("RGB", (4, 4), "blue")
    figures = [
        plot_target_distributions(tables["distributions"]),
        plot_article_type_support(tables["distributions"]["articleType"], tables["support"]),
        plot_association_matrix(tables["associations"]),
        plot_relationship_heatmap(tables["relationship"], "gender", "usage"),
        plot_drift(tables["drift"], "year"),
        plot_image_profiles(tables["images"]),
        plot_duplicate_summary(tables["duplicates"]),
        plot_review_grid([(1, image, "example")], "review"),
    ]

    assert all(figure.axes for figure in figures)
    assert not shown
    for figure in figures:
        plt.close(figure)


def test_target_distribution_uses_readable_ranked_labels() -> None:
    """Putting all 124 article labels on one x-axis makes the chart unreadable."""
    article = pd.DataFrame(
        {
            "label": [f"class-{index:02d}" for index in range(30)],
            "count": list(range(30, 0, -1)),
        }
    )
    article["share"] = article["count"] / article["count"].sum()
    usage = pd.DataFrame(
        {
            "label": ["Casual", "Formal", "Home"],
            "count": [1000, 20, 1],
            "share": [1000 / 1021, 20 / 1021, 1 / 1021],
        }
    )

    figure = plot_target_distributions({"articleType": article, "usage": usage})

    article_axis, usage_axis = figure.axes
    article_labels = [label.get_text() for label in article_axis.get_yticklabels()]
    assert "Top 15 of 30" in article_axis.get_title()
    assert len(article_labels) == 15
    assert "class-00" in article_labels
    assert usage_axis.get_xscale() == "log"
    assert {label.get_text() for label in usage_axis.get_yticklabels()} == {
        "Casual",
        "Formal",
        "Home",
    }
    plt.close(figure)


def test_article_support_names_the_head_and_summarizes_the_tail() -> None:
    """The long-tail view needs readable class names without labeling every rank."""
    article = pd.DataFrame(
        {
            "label": [f"class-{index:02d}" for index in range(30)],
            "count": list(range(30, 0, -1)),
        }
    )
    article["share"] = article["count"] / article["count"].sum()
    support = pd.DataFrame(
        {
            "band": ["1", "2", "3–4", "5–9", "10+"],
            "class_count": [1, 1, 2, 5, 21],
            "product_count": [1, 2, 7, 35, 420],
        }
    )

    figure = plot_article_type_support(article, support)

    head_axis, rank_axis, cumulative_axis, support_axis = figure.axes
    assert "20 largest" in head_axis.get_title()
    assert len(head_axis.get_yticklabels()) == 20
    assert rank_axis.get_yscale() == "log"
    assert rank_axis.get_xlabel() == "Class rank"
    assert cumulative_axis.get_ylim()[1] == pytest.approx(1.0)
    assert len(support_axis.patches) == 5
    plt.close(figure)


def test_report_summary_has_exactly_four_main_axes() -> None:
    """The fixed report compares targets using normalized entropy skew."""
    tables = _plot_tables()

    figure = build_report_summary(
        tables["distributions"],
        tables["associations"],
        tables["images"],
    )

    assert len(figure.axes) == 4
    assert all(axis.has_data() for axis in figure.axes)
    expected_skew = 1 - (
        -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
    ) / math.log2(2)
    skew_axis = figure.axes[0]
    assert [bar.get_height() for bar in skew_axis.patches] == pytest.approx(
        [expected_skew, expected_skew]
    )
    assert skew_axis.get_ylabel() == "Normalized skew (0 balanced, 1 concentrated)"
    plt.close(figure)


def test_report_quality_panel_uses_visible_within_metric_median_normalization() -> None:
    """Only resolution-independent metrics belong in the low/high report panel."""
    tables = _plot_tables()
    paired = pd.DataFrame(
        {
            "edge_sharpness_low": [1_000.0, 2_000.0, 3_000.0],
            "edge_sharpness_high": [2_000.0, 4_000.0, 6_000.0],
            "brightness_low": [20.0, 40.0, 60.0],
            "brightness_high": [40.0, 80.0, 120.0],
            "contrast_low": [5.0, 10.0, 15.0],
            "contrast_high": [10.0, 20.0, 30.0],
            "colorfulness_low": [10.0, 20.0, 30.0],
            "colorfulness_high": [20.0, 40.0, 60.0],
            "saturation_low": [0.1, 0.2, 0.3],
            "saturation_high": [0.2, 0.4, 0.6],
        }
    )

    figure = build_report_summary(
        tables["distributions"],
        tables["associations"],
        paired,
    )

    assert len(figure.axes) == 4
    quality_axis = figure.axes[3]
    heights = np.array([patch.get_height() for patch in quality_axis.patches])
    assert [label.get_text() for label in quality_axis.get_xticklabels()] == [
        "brightness",
        "contrast",
        "colorfulness",
        "saturation",
    ]
    assert "edge_sharpness" not in quality_axis.get_xticklabels()
    assert "excluded" in quality_axis.get_title().lower()
    assert len(heights) == 8
    assert np.isfinite(heights).all()
    assert ((heights >= 0) & (heights <= 1)).all()
    assert heights.tolist() == pytest.approx([0.5] * 4 + [1.0] * 4)
    assert quality_axis.get_ylabel() == "Normalized median (within metric, 0–1)"
    plt.close(figure)


def test_plot_drift_orders_id_ranges_by_numeric_lower_bound() -> None:
    """Connected catalogue trends must follow numeric, not input or lexical, order."""
    drift = pd.DataFrame(
        {
            "id_bin": ["100–109", "2–9", "10–19"],
            "total_variation": [0.3, 0.1, 0.2],
        }
    )

    figure = plot_drift(drift, "id_bin")

    axis = figure.axes[0]
    assert [label.get_text() for label in axis.get_xticklabels()] == ["2–9", "10–19", "100–109"]
    assert axis.lines[0].get_ydata().tolist() == pytest.approx([0.1, 0.2, 0.3])
    plt.close(figure)


def test_plot_image_profiles_excludes_native_sharpness_from_low_high_comparison() -> None:
    """Native-resolution sharpness is descriptive evidence, not a fair paired comparison."""
    paired = pd.DataFrame(
        {
            "edge_sharpness_low": [1_000.0, 2_000.0],
            "edge_sharpness_high": [2_000.0, 4_000.0],
            "edge_sharpness_delta": [1_000.0, 2_000.0],
            "brightness_low": [20.0, 40.0],
            "brightness_high": [40.0, 80.0],
            "brightness_delta": [20.0, 40.0],
            "contrast_low": [5.0, 10.0],
            "contrast_high": [10.0, 20.0],
            "contrast_delta": [5.0, 10.0],
            "colorfulness_low": [10.0, 20.0],
            "colorfulness_high": [20.0, 40.0],
            "colorfulness_delta": [10.0, 20.0],
            "saturation_low": [0.1, 0.2],
            "saturation_high": [0.2, 0.4],
            "saturation_delta": [0.1, 0.2],
        }
    )

    figure = plot_image_profiles(paired)

    expected_metrics = ["brightness", "contrast", "colorfulness", "saturation"]
    for axis in figure.axes:
        assert [label.get_text() for label in axis.get_xticklabels()] == expected_metrics
        assert "edge_sharpness" not in [label.get_text() for label in axis.get_xticklabels()]
        assert "excluded" in axis.get_title().lower()
    plt.close(figure)


def test_review_grid_caps_rendered_examples_at_twenty_four() -> None:
    """An unbounded candidate list makes the review grid too small to read."""
    examples = [
        (product_id, Image.new("RGB", (2, 2), "blue"), "example")
        for product_id in range(1, 31)
    ]

    figure = plot_review_grid(examples, "review")

    assert len(figure.axes) == 24
    assert sum(bool(axis.get_title()) for axis in figure.axes) == 24
    plt.close(figure)


def test_strongest_relationship_skips_pairs_with_unreadable_cardinality() -> None:
    """A numerically strong 25×25 hierarchy table is not a readable detail plot."""
    frame = pd.DataFrame(
        {
            "gender": ["Men", "Women"] * 13,
            "usage": ["Casual", "Formal"] * 13,
            "articleType": [f"type-{index}" for index in range(26)],
            "baseColour": [f"colour-{index}" for index in range(26)],
        }
    )
    names = ["gender", "usage", "articleType", "baseColour"]
    associations = pd.DataFrame(0.0, index=names, columns=names)
    associations.loc["articleType", "baseColour"] = 1.0
    associations.loc["baseColour", "articleType"] = 1.0
    associations.loc["gender", "usage"] = 0.6
    associations.loc["usage", "gender"] = 0.6

    left, right, relationship = eda_module._strongest_pair(frame, associations)

    assert (left, right) == ("gender", "usage")
    assert relationship.shape == (2, 2)


def test_run_eda_writes_stable_review_grids_and_replaces_stale_ones(
    eda_paths: EdaPaths, tmp_path: Path
) -> None:
    """Every run needs all six review files, even when a candidate group is empty."""
    output_dir = tmp_path / "eda-output"
    split_path = PROCESSED_DATA_DIR / "splits.csv"
    split_before = split_path.read_bytes() if split_path.is_file() else None
    review_files = {
        "review-common.png",
        "review-rare.png",
        "review-unusual.png",
        "review-grayscale.png",
        "review-exact-duplicate.png",
        "review-near-duplicate.png",
    }
    result = run_eda(paths=eda_paths, output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    low_cache = output_dir / "lowres-measurements.csv"
    provenance_path = output_dir / "lowres-measurements.provenance.json"

    required = {
        "summary.json",
        "target-distributions.csv",
        "target-skew.csv",
        "article-type-support.csv",
        "associations.csv",
        "year-drift.csv",
        "id-range-drift.csv",
        "lowres-measurements.csv",
        "highres-measurements.csv",
        "exact-duplicates.csv",
        "near-duplicates.csv",
        "paired-image-comparison.csv",
        "eda-report-summary.png",
        "master-category-distribution.png",
    }
    assert required | review_files <= {path.name for path in output_dir.iterdir()}
    assert all(path.stat().st_size > 0 for path in output_dir.iterdir() if path.is_file())
    assert result.summary_path == summary_path
    assert result.output_dir == output_dir
    assert low_cache.is_file() and provenance_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "master-category-distribution.png" in summary["output_manifest"]["files"]
    assert summary["output_manifest"]["review_grids"] == sorted(review_files)
    assert summary["population"]["source_train_ids"] == 4
    assert summary["population"]["usable_products"] == 4
    assert summary["test_quarantine"]["overlap_count"] == 0
    assert summary["cache_provenance"]["lowres"]["metric_version"]
    assert summary["split_provenance"]["before"] == summary["split_provenance"]["after"]
    assert summary["split_provenance"]["unchanged"] is True
    assert summary["duplicates"]["exact_group_count"] == 1
    assert summary["duplicates"]["exact_product_count"] == 2
    assert summary["duplicates"]["near_candidate_pair_count"] >= 0
    assert summary["duplicates"]["near_sample_product_count"] == 4
    json.dumps(summary, allow_nan=False)
    assert (split_path.read_bytes() if split_path.is_file() else None) == split_before
    initial_mtime = low_cache.stat().st_mtime_ns

    reused = run_eda(paths=eda_paths, output_dir=output_dir)
    assert reused.cache_status["lowres"] == "reused"
    assert low_cache.stat().st_mtime_ns == initial_mtime

    for filename in review_files:
        (output_dir / filename).write_bytes(b"stale review")
    replaced = run_eda(paths=eda_paths, output_dir=output_dir)
    assert all((output_dir / filename).read_bytes() != b"stale review" for filename in review_files)

    _save_image(eda_paths.lowres_image_dir / "1.png", (250, 250, 250))
    invalidated = run_eda(paths=eda_paths, output_dir=output_dir)
    assert invalidated.cache_status["lowres"] == "recomputed"
    assert low_cache.stat().st_mtime_ns > initial_mtime

    refreshed = run_eda(paths=eda_paths, output_dir=output_dir, refresh=True)
    assert refreshed.cache_status["lowres"] == "recomputed"


def test_unusual_review_collects_all_measurement_extremes_and_unusual_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unusual review must not silently collapse to brightness extremes only."""
    captured: dict[str, list[int]] = {}

    def fake_open_examples(
        frame: pd.DataFrame, ids: list[int], title_column: str = "articleType"
    ) -> list[tuple[int, object, str]]:
        return [(product_id, Image.new("RGB", (1, 1)), "example") for product_id in ids]

    def fake_review_grid(
        examples: list[tuple[int, object, str]], title: str, columns: int = 4
    ) -> object:
        if title == "Unusual image review":
            captured["ids"] = [product_id for product_id, _, _ in examples]
        return plt.figure()

    monkeypatch.setattr(eda_module, "_open_examples", fake_open_examples)
    monkeypatch.setattr(eda_module, "plot_review_grid", fake_review_grid)
    monkeypatch.setattr(eda_module, "_save_figure", lambda figure, path: plt.close(figure))
    frame = pd.DataFrame(
        {
            "id": list(range(1, 61)),
            "articleType": ["Tshirts"] * 60,
            "lowres_image_path": [tmp_path / f"{item}.png" for item in range(1, 61)],
        }
    )
    lowres = pd.DataFrame(
        {
            "id": list(range(1, 61)),
            "width": [0 if item == 20 else 200 if item == 21 else 100 for item in range(1, 61)],
            "height": [0 if item == 22 else 200 if item == 23 else 100 for item in range(1, 61)],
            "mode": ["L" if item >= 34 else "RGB" for item in range(1, 61)],
            "brightness": [0 if item == 24 else 200 if item == 25 else 100 for item in range(1, 61)],
            "contrast": [0 if item == 26 else 200 if item == 27 else 100 for item in range(1, 61)],
            "colorfulness": [0 if item == 28 else 200 if item == 29 else 100 for item in range(1, 61)],
            "saturation": [0 if item == 30 else 200 if item == 31 else 100 for item in range(1, 61)],
            "edge_sharpness": [0 if item == 32 else 200 if item == 33 else 100 for item in range(1, 61)],
        }
    )

    eda_module._write_review_grids(
        frame,
        lowres,
        pd.DataFrame(columns=["ids"]),
        pd.DataFrame(columns=["id_left", "id_right"]),
        tmp_path,
    )

    assert captured["ids"] == list(range(20, 44))


def test_run_eda_serializes_complete_data_audit_evidence(
    eda_paths: EdaPaths, tmp_path: Path,
) -> None:
    """A summary-only population total cannot prove malformed source rows were audited."""
    teacher_lines = eda_paths.teacher_train_csv.read_text(encoding="utf-8").splitlines()
    teacher_lines[0] += ",,"
    teacher_lines[1:] = [f"{line},," for line in teacher_lines[1:]]
    eda_paths.teacher_train_csv.write_text("\n".join(teacher_lines), encoding="utf-8")
    original_lines = eda_paths.original_csv.read_text(encoding="utf-8").splitlines()
    original_lines[1] += ",spilled"
    eda_paths.original_csv.write_text("\n".join(original_lines), encoding="utf-8")
    _save_image(eda_paths.original_image_dir / "invalid-name.png", (1, 2, 3))
    _save_image(eda_paths.lowres_image_dir / "1.jpg", (1, 2, 3))

    result = run_eda(paths=eda_paths, output_dir=tmp_path / "eda-output")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    audit_path = result.output_dir / "data-audit.json"
    data_audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert "data-audit.json" in GENERATED_EDA_OUTPUTS
    assert summary["data_audit"] == data_audit
    assert data_audit["teacher_csv_audit"]["row_count"] == 4
    assert data_audit["teacher_csv_audit"]["physical_columns"][-2:] == [
        "Unnamed: 10",
        "Unnamed: 11",
    ]
    assert data_audit["teacher_csv_audit"]["blank_counts"]["usage"] == 0
    assert data_audit["teacher_csv_audit"]["literal_na_usage_count"] == 1
    assert data_audit["original_csv_audit"]["row_count"] == 4
    assert data_audit["original_csv_audit"]["phantom_nonempty_counts"] == {
        "Overflow: 10": 1
    }
    assert data_audit["invalid_original_image_filenames"] == ["invalid-name.png"]
    assert data_audit["duplicate_lowres_image_ids"] == [1]
    assert "unmatched_original_image_ids" in data_audit
    assert "unmatched_lowres_image_ids" in data_audit


def test_run_eda_records_split_before_and_after_and_fails_on_change(
    eda_paths: EdaPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run must prove it neither creates nor changes the official split."""
    absent = {"status": "absent", "path": "/tmp/splits.csv"}
    unchanged_states = iter([absent.copy(), absent.copy()])
    monkeypatch.setattr(
        eda_module, "_split_provenance", lambda: next(unchanged_states)
    )
    result = run_eda(paths=eda_paths, output_dir=tmp_path / "unchanged")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert summary["split_provenance"] == {
        "before": absent,
        "after": absent,
        "unchanged": True,
    }

    changed_states = iter(
        [
            absent.copy(),
            {
                "status": "present",
                "path": "/tmp/splits.csv",
                "sha256": "changed",
            },
        ]
    )
    monkeypatch.setattr(
        eda_module, "_split_provenance", lambda: next(changed_states)
    )
    with pytest.raises(RuntimeError, match="split.*changed|created"):
        run_eda(paths=eda_paths, output_dir=tmp_path / "changed")


def test_cache_reuse_preserves_leading_zero_hash_strings(
    eda_paths: EdaPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric CSV inference would strip leading zeroes and corrupt dHash distances."""
    output_dir = tmp_path / "eda-output"
    run_eda(paths=eda_paths, output_dir=output_dir)
    near_cache = output_dir / "near-lowres-measurements.csv"
    cached = pd.read_csv(near_cache, dtype="string")
    cached["dhash"] = [
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
        "0000000000000004",
    ]
    cached.to_csv(near_cache, index=False)
    observed: list[str] = []
    real_candidates = eda_module.near_duplicate_candidates

    def capture_hashes(measurements: pd.DataFrame) -> pd.DataFrame:
        observed.extend(measurements["dhash"].tolist())
        return real_candidates(measurements)

    monkeypatch.setattr(eda_module, "near_duplicate_candidates", capture_hashes)

    result = run_eda(paths=eda_paths, output_dir=output_dir)

    assert result.cache_status["near_lowres"] == "reused"
    assert observed == [
        "0000000000000001",
        "0000000000000002",
        "0000000000000003",
        "0000000000000004",
    ]


def test_corrupt_measurement_cache_schema_ids_and_paths_recompute(
    eda_paths: EdaPaths, tmp_path: Path,
) -> None:
    """Matching provenance must not bless a malformed or mismatched measurement CSV."""
    output_dir = tmp_path / "eda-output"
    run_eda(paths=eda_paths, output_dir=output_dir)
    low_cache = output_dir / "lowres-measurements.csv"

    missing_column = pd.read_csv(low_cache).drop(columns="mode")
    missing_column.to_csv(low_cache, index=False)
    schema_result = run_eda(paths=eda_paths, output_dir=output_dir)
    assert schema_result.cache_status["lowres"] == "recomputed"

    wrong_contract = pd.read_csv(low_cache, dtype={"path": "string"})
    wrong_contract.loc[0, "id"] = 999
    wrong_contract.loc[0, "path"] = "/wrong/image.png"
    wrong_contract.to_csv(low_cache, index=False)
    contract_result = run_eda(paths=eda_paths, output_dir=output_dir)
    assert contract_result.cache_status["lowres"] == "recomputed"


def _load_verify_module() -> object:
    """Load the directly runnable verifier as the real synthetic test boundary."""
    path = Path(__file__).parents[1] / "scripts" / "verify_eda.py"
    spec = importlib.util.spec_from_file_location("verify_eda_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_eda_contract_is_complete_and_excludes_arbitrary_stale_files(
    eda_paths: EdaPaths, tmp_path: Path
) -> None:
    """A prior unrelated output must never be reported as evidence from this run."""
    output_dir = tmp_path / "eda-output"
    output_dir.mkdir()
    (output_dir / "old-experiment.csv").write_text("stale", encoding="utf-8")

    result = run_eda(paths=eda_paths, output_dir=output_dir)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert set(GENERATED_EDA_OUTPUTS) == set(result.manifest)
    assert set(summary["output_manifest"]["files"]) == set(GENERATED_EDA_OUTPUTS)
    assert "old-experiment.csv" not in result.manifest
    assert (output_dir / "old-experiment.csv").exists()
    assert all((output_dir / name).stat().st_size > 0 for name in GENERATED_EDA_OUTPUTS)
    assert set(summary["output_manifest"]["review_grids"]) == {
        "review-common.png",
        "review-rare.png",
        "review-unusual.png",
        "review-grayscale.png",
        "review-exact-duplicate.png",
        "review-near-duplicate.png",
    }


def test_image_cache_provenance_detects_restored_timestamp_corruption_and_missing_files(
    eda_paths: EdaPaths, tmp_path: Path
) -> None:
    """Cache reuse must depend on inventory content, complete provenance, and existing files."""
    output_dir = tmp_path / "eda-output"
    first = run_eda(paths=eda_paths, output_dir=output_dir)
    assert first.cache_status["lowres"] == "recomputed"

    low_image = eda_paths.lowres_image_dir / "1.png"
    original_stat = low_image.stat()
    _save_image(low_image, (250, 250, 250))
    os.utime(low_image, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    content_changed = run_eda(paths=eda_paths, output_dir=output_dir)
    assert content_changed.cache_status["lowres"] == "recomputed"

    low_provenance = output_dir / "lowres-measurements.provenance.json"
    low_provenance.write_text("{not json", encoding="utf-8")
    corrupt = run_eda(paths=eda_paths, output_dir=output_dir)
    assert corrupt.cache_status["lowres"] == "recomputed"

    high_provenance = output_dir / "highres-measurements.provenance.json"
    high_provenance.unlink()
    missing = run_eda(paths=eda_paths, output_dir=output_dir)
    assert missing.cache_status["highres"] == "recomputed"

    (eda_paths.lowres_image_dir / "4.png").unlink()
    disappeared = run_eda(paths=eda_paths, output_dir=output_dir)
    assert disappeared.cache_status["lowres"] == "recomputed"


def test_synthetic_verifier_rejects_missing_contract_and_split_changes(
    eda_paths: EdaPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifier accepts a complete synthetic bundle but catches missing evidence and split drift."""
    output_dir = tmp_path / "eda-output"
    run_eda(paths=eda_paths, output_dir=output_dir)
    verifier = _load_verify_module()
    split_dir = tmp_path / "processed"
    split_dir.mkdir()
    split_path = split_dir / "splits.csv"
    split_path.write_text("id,split\n1,train\n", encoding="utf-8")
    monkeypatch.setattr(verifier, "PROCESSED_DATA_DIR", split_dir)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_state = {
        "status": "present",
        "path": str(split_path),
        "sha256": __import__("hashlib").sha256(split_path.read_bytes()).hexdigest(),
    }
    summary["split_provenance"] = {
        "before": split_state,
        "after": split_state,
        "unchanged": True,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert verifier.verify_eda(output_dir) == []

    (output_dir / "eda-report-summary.png").unlink()
    assert any("eda-report-summary.png" in failure for failure in verifier.verify_eda(output_dir))

    run_eda(paths=eda_paths, output_dir=output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_state = {
        "status": "present",
        "path": str(split_path),
        "sha256": __import__("hashlib").sha256(split_path.read_bytes()).hexdigest(),
    }
    summary["split_provenance"] = {
        "before": split_state,
        "after": split_state,
        "unchanged": True,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    split_path.write_text("id,split\n1,test\n", encoding="utf-8")
    assert "split content changed after EDA" in verifier.verify_eda(output_dir)

    summary["split_provenance"]["unchanged"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert "EDA run reported split provenance changed" in verifier.verify_eda(output_dir)


def test_eda_scripts_bootstrap_src_when_run_from_repo_root() -> None:
    """Users can invoke both scripts without manually exporting PYTHONPATH."""
    root = Path(__file__).parents[1]
    for script in ("run_eda.py", "verify_eda.py"):
        completed = subprocess.run(
            [sys.executable, f"scripts/{script}", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
