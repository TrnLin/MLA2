from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import fashion.task4 as task4
from fashion.task4.baseline import BaselineEvaluation
from fashion.task4.baseline_evidence import (
    EXAMPLE_COLUMNS,
    write_baseline_artifacts,
)

SUMMARY_COLUMNS = [
    "method",
    "fold",
    "size",
    "query_source",
    "gallery_source",
    "protocol",
    "scope",
    "width",
    "height",
    "query_transform_seconds",
    "gallery_transform_seconds",
    "query_source_bytes",
    "gallery_source_bytes",
    "feature_bytes_per_image",
    "uint8_tensor_bytes_per_image",
    "float32_tensor_bytes_per_image",
    "metric",
    "k",
    "aggregation",
    "value",
    "query_count",
    "class_count",
    "passed",
]
QUERY_COLUMNS = [
    "method",
    "fold",
    "size",
    "query_source",
    "gallery_source",
    "protocol",
    "scope",
    "query_id",
    "articleType",
    "ndcg_at_5",
    "precision_any_at_5",
    "precision_strict_at_5",
    "tie_rate_at_5",
    "ndcg_at_10",
    "precision_any_at_10",
    "precision_strict_at_10",
    "tie_rate_at_10",
    "ndcg_at_20",
    "precision_any_at_20",
    "precision_strict_at_20",
    "tie_rate_at_20",
    "recall_at_10",
    "hit_rate_at_10",
    "precision_at_10",
]
SLICE_COLUMNS = [
    "scope",
    "fold",
    "size",
    "query_source",
    "gallery_source",
    "protocol",
    "slice",
    "metric",
    "k",
    "aggregation",
    "value",
    "total_queries",
    "scored_queries",
    "excluded_queries",
    "coverage",
]
TIMING_COLUMNS = [
    "scope",
    "fold",
    "query_id",
    "query_source",
    "gallery_source",
    "encoding_seconds",
    "search_seconds",
    "end_to_end_seconds",
]


def _baseline() -> BaselineEvaluation:
    summary = pd.DataFrame(
        [
            [
                "spatial-probe-v1",
                1,
                "240x320",
                query_source,
                gallery_source,
                "primary",
                "development",
                240,
                320,
                1.0,
                1.0,
                100,
                100,
                1600,
                230400,
                921600,
                "ndcg",
                10,
                "query_mean",
                value,
                2,
                pd.NA,
                pd.NA,
            ]
            for query_source, gallery_source, value in (
                ("v1", "v1", 0.8),
                ("teacher", "teacher", 0.7),
            )
        ],
        columns=SUMMARY_COLUMNS,
    )
    query_metrics = pd.DataFrame(
        [
            [
                "spatial-probe-v1",
                1,
                "240x320",
                "v1",
                "v1",
                protocol,
                "development",
                query_id,
                "Shirts",
                *([score] * 15),
            ]
            for query_id, score in ((20, 0.2), (10, 0.9))
            for protocol in ("family", "primary")
        ],
        columns=QUERY_COLUMNS,
    )
    return BaselineEvaluation(
        summary=summary,
        query_metrics=query_metrics,
        pair_evaluations={},
        random_rankings=pd.DataFrame(),
    )


def _slice_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "development",
                1,
                "240x320",
                "v1",
                "v1",
                "primary",
                slice_name,
                "ndcg",
                10,
                "query_mean",
                value,
                2,
                2,
                0,
                1.0,
            ]
            for slice_name, value in (("canvas_tall", 0.2), ("grayscale", 0.4))
        ],
        columns=SLICE_COLUMNS,
    )


def _timings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["development", 1, 20, "v1", "v1", 0.2, 0.02, 0.22],
            ["development", 1, 10, "teacher", "teacher", 0.1, 0.01, 0.11],
        ],
        columns=TIMING_COLUMNS,
    )


def _cost() -> dict[str, object]:
    directions = (
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    )
    timing_summary = [
        {
            "query_source": query_source,
            "gallery_source": gallery_source,
            "metric": metric,
            "percentile": percentile,
            "value_seconds": 0.1,
            "timed_queries": 2,
        }
        for query_source, gallery_source in directions
        for metric in ("encoding", "search", "end_to_end")
        for percentile in ("p50", "p95")
    ]
    index_costs = {
        source: {
            "source": source,
            "contract": {
                "width": 240,
                "height": 320,
                "pad_color": [255, 255, 255],
                "colour_mode": "RGB",
                "resize": "aspect_preserving_letterbox",
                "resample": "LANCZOS",
            },
            "rows": 3,
            "dimension": 400,
            "payload_bytes": 4_800,
            "index_bytes": 4_824,
            "build_seconds": 0.2,
            "peak_rss_bytes": 10_000,
        }
        for source in ("teacher", "v1")
    }
    return {
        "schema_version": 1,
        "scope": "development",
        "parameters": 0,
        "checkpoint_bytes": 0,
        "timed_queries": 2,
        "timing_summary": timing_summary,
        "per_source_index_cost": index_costs,
    }


def test_canvas_failure_summary_counts_only_queries_with_defined_ndcg() -> None:
    runner = runpy.run_path(
        Path(__file__).resolve().parents[2] / "scripts/task4/run_baseline.py",
        run_name="task5_canvas_test",
    )
    summary = pd.DataFrame(
        {
            "scope": ["development"] * 3,
            "fold": [1] * 3,
            "size": ["240x320"] * 3,
            "query_source": ["v1"] * 3,
            "gallery_source": ["v1"] * 3,
            "query_variant": ["clean", "wide", "tall"],
            "queries": [3, 3, 3],
            "ndcg_at_10": [0.7, 0.3, 0.5],
        }
    )
    per_query = pd.DataFrame(
        {
            "scope": ["development"] * 9,
            "fold": [1] * 9,
            "size": ["240x320"] * 9,
            "query_source": ["v1"] * 9,
            "gallery_source": ["v1"] * 9,
            "query_variant": ["clean"] * 3 + ["wide"] * 3 + ["tall"] * 3,
            "query_id": [1, 2, 3] * 3,
            "canvas_ndcg_at_10": [
                0.8,
                float("nan"),
                0.6,
                float("nan"),
                float("nan"),
                0.3,
                0.4,
                0.5,
                0.6,
            ],
        }
    )

    result = runner["_canvas_failure_summary"](summary, per_query).set_index("slice")

    assert result.loc[
        ["canvas_clean", "canvas_wide", "canvas_tall"],
        ["total_queries", "scored_queries", "excluded_queries", "coverage"],
    ].values.tolist() == [
        [3, 2, 1, pytest.approx(2 / 3)],
        [3, 1, 2, pytest.approx(1 / 3)],
        [3, 3, 0, pytest.approx(1.0)],
    ]


def _example_inputs(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows: list[dict[str, object]] = []
    for product_id in (1, 2, 3, 4, 5, 10, 20):
        path = image_dir / f"{product_id}.png"
        Image.new(
            "RGB",
            (12, 8),
            (product_id * 10 % 255, product_id * 20 % 255, 80),
        ).save(path)
        rows.append(
            {
                "id": product_id,
                "partition": "development",
                "external_path": str(path),
            }
        )
    image_rows = pd.DataFrame(rows)

    examples = pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": 1,
                "size": "240x320",
                "query_source": "v1",
                "gallery_source": "v1",
                "slice": slice_name,
                "query_variant": query_variant,
                "query_id": query_id,
                "metric": "ndcg_at_10",
                "value": score,
                "rank": rank,
                "candidate_id": rank,
                "distance": rank / 10,
            }
            for slice_name, query_variant, query_id, score in (
                ("canvas_failure", "tall", 20, 0.2),
                ("normal_success", "clean", 10, 0.9),
            )
            for rank in range(5, 0, -1)
        ],
        columns=EXAMPLE_COLUMNS,
    )
    rankings = {
        slice_name: group.loc[
            :, ["query_id", "candidate_id", "distance", "rank"]
        ].reset_index(drop=True)
        for slice_name, group in examples.groupby("slice", sort=False)
    }
    return image_rows, examples, rankings


def test_writer_creates_complete_deterministic_artifact_set(tmp_path: Path) -> None:
    image_rows, examples, rankings = _example_inputs(tmp_path)
    first_evidence = tmp_path / "first/evidence"
    first_figures = tmp_path / "first/figures"

    write_baseline_artifacts(
        baseline=_baseline(),
        slice_summary=_slice_summary(),
        timings=_timings(),
        cost=_cost(),
        examples=examples,
        evidence_dir=first_evidence,
        figure_dir=first_figures,
        image_rows=image_rows,
        example_rankings=rankings,
    )
    write_baseline_artifacts(
        baseline=_baseline(),
        slice_summary=_slice_summary(),
        timings=_timings(),
        cost=_cost(),
        examples=examples,
        evidence_dir=tmp_path / "second/evidence",
        figure_dir=tmp_path / "second/figures",
        image_rows=image_rows,
        example_rankings=rankings,
    )

    expected_names = {
        "baseline_summary.csv",
        "baseline_query_metrics.csv",
        "baseline_failure_slices.csv",
        "baseline_timing.csv",
        "baseline_cost.json",
        "baseline_examples.csv",
    }
    assert {path.name for path in first_evidence.iterdir()} == expected_names
    assert (first_figures / "baseline_examples.png").is_file()
    with Image.open(first_figures / "baseline_examples.png") as figure:
        assert figure.size == (2160, 2592)
    for name in expected_names:
        first = (first_evidence / name).read_bytes()
        second = (tmp_path / "second/evidence" / name).read_bytes()
        assert first == second
        assert first.endswith(b"\n")

    summary = pd.read_csv(first_evidence / "baseline_summary.csv")
    assert summary[["query_source", "gallery_source"]].values.tolist() == [
        ["teacher", "teacher"],
        ["v1", "v1"],
    ]
    queries = pd.read_csv(first_evidence / "baseline_query_metrics.csv")
    assert queries[["query_source", "gallery_source", "protocol", "query_id"]].values.tolist() == [
        ["v1", "v1", "family", 10],
        ["v1", "v1", "family", 20],
        ["v1", "v1", "primary", 10],
        ["v1", "v1", "primary", 20],
    ]
    slices = pd.read_csv(first_evidence / "baseline_failure_slices.csv")
    assert slices["slice"].tolist() == ["grayscale", "canvas_tall"]
    timing = pd.read_csv(first_evidence / "baseline_timing.csv")
    assert timing["query_id"].tolist() == [10, 20]
    persisted_examples = pd.read_csv(first_evidence / "baseline_examples.csv")
    assert persisted_examples.groupby("slice")["rank"].apply(list).to_dict() == {
        "normal_success": [1, 2, 3, 4, 5],
        "canvas_failure": [1, 2, 3, 4, 5],
    }
    assert len(persisted_examples) == 10

    cost = json.loads((first_evidence / "baseline_cost.json").read_text())
    assert cost["schema_version"] == 1
    assert cost["scope"] == "development"
    assert cost["parameters"] == 0
    assert cost["checkpoint_bytes"] == 0
    for csv_path in first_evidence.glob("*.csv"):
        frame = pd.read_csv(csv_path)
        if "scope" in frame:
            assert set(frame["scope"]) == {"development"}
        if "partition" in frame:
            assert not frame["partition"].isin({"holdout", "quarantine"}).any()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_scope", "columns"),
        ("missing_fold", "columns"),
        ("wrong_scope", "scope must be development"),
        ("wrong_fold", "frozen fold 1"),
    ],
)
def test_writer_rejects_missing_or_wrong_timing_provenance(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    timings = _timings()
    if mutation == "missing_scope":
        timings = timings.drop(columns="scope")
    elif mutation == "missing_fold":
        timings = timings.drop(columns="fold")
    elif mutation == "wrong_scope":
        timings["scope"] = "holdout"
    elif mutation == "wrong_fold":
        timings["fold"] = 2
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    image_rows, examples, rankings = _example_inputs(tmp_path)

    with pytest.raises(ValueError, match=message):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=timings,
            cost=_cost(),
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


def test_writer_rejects_non_top_five_example_rows(tmp_path: Path) -> None:
    image_rows, examples, rankings = _example_inputs(tmp_path)
    examples.loc[examples.index[-1], "rank"] = 6

    with pytest.raises(ValueError, match="Top-5"):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=_timings(),
            cost=_cost(),
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


def test_writer_rejects_non_frozen_example_size(tmp_path: Path) -> None:
    image_rows, examples, rankings = _example_inputs(tmp_path)
    examples["size"] = "rgb-letterbox-lanczos-240x320"

    with pytest.raises(ValueError, match="240x320"):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=_timings(),
            cost=_cost(),
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "scope or partition"),
        ("protected", "partition must be development"),
        ("other", "partition must be development"),
        ("protected_scope", "scope must be development"),
    ],
)
def test_writer_rejects_missing_or_non_development_image_provenance(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    image_rows, examples, rankings = _example_inputs(tmp_path)
    if mutation == "missing":
        image_rows = image_rows.drop(columns="partition")
    elif mutation == "protected":
        image_rows["partition"] = "holdout"
    elif mutation == "other":
        image_rows["partition"] = "training"
    elif mutation == "protected_scope":
        image_rows = image_rows.drop(columns="partition").assign(scope="holdout")
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(ValueError, match=message):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=_timings(),
            cost=_cost(),
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("width", 241),
        ("height", 321),
        ("pad_color", [0, 0, 0]),
        ("colour_mode", "BGR"),
        ("resize", "stretch"),
        ("resample", "BILINEAR"),
    ],
)
def test_writer_rejects_any_changed_nested_preprocessing_contract_field(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    cost = _cost()
    source_costs = cost["per_source_index_cost"]
    assert isinstance(source_costs, dict)
    teacher_cost = source_costs["teacher"]
    assert isinstance(teacher_cost, dict)
    contract = teacher_cost["contract"]
    assert isinstance(contract, dict)
    contract[field] = bad_value
    image_rows, examples, rankings = _example_inputs(tmp_path)

    with pytest.raises(ValueError, match="frozen 240x320 contract"):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=_timings(),
            cost=cost,
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("incomplete_timing", "complete"),
        ("duplicate_timing", "exactly one"),
        ("non_finite_timing", "finite"),
        ("missing_source_cost", "teacher and v1"),
        ("inconsistent_timed_counts", "consistent"),
        ("mismatched_top_level_count", "must match"),
        ("inconsistent_index_counts", "row and dimension counts"),
    ],
)
def test_writer_rejects_malformed_cost_evidence(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    cost = _cost()
    timing_summary = cost["timing_summary"]
    source_costs = cost["per_source_index_cost"]
    assert isinstance(timing_summary, list)
    assert isinstance(source_costs, dict)
    if case == "incomplete_timing":
        timing_summary.pop()
    elif case == "duplicate_timing":
        timing_summary.append(dict(timing_summary[0]))
    elif case == "non_finite_timing":
        timing_summary[0]["value_seconds"] = float("nan")
    elif case == "missing_source_cost":
        source_costs.pop("v1")
    elif case == "inconsistent_timed_counts":
        timing_summary[0]["timed_queries"] = 3
    elif case == "mismatched_top_level_count":
        cost["timed_queries"] = 3
    elif case == "inconsistent_index_counts":
        v1_cost = source_costs["v1"]
        assert isinstance(v1_cost, dict)
        v1_cost["rows"] = 4
        v1_cost["payload_bytes"] = 6_400
        v1_cost["index_bytes"] = 6_432
    else:
        raise AssertionError(f"unknown test case: {case}")
    image_rows, examples, rankings = _example_inputs(tmp_path)

    with pytest.raises(ValueError, match=message):
        write_baseline_artifacts(
            baseline=_baseline(),
            slice_summary=_slice_summary(),
            timings=_timings(),
            cost=cost,
            examples=examples,
            evidence_dir=tmp_path / "evidence",
            figure_dir=tmp_path / "figures",
            image_rows=image_rows,
            example_rankings=rankings,
        )


def test_evidence_api_is_exported_from_task4_package() -> None:
    assert task4.write_baseline_artifacts is write_baseline_artifacts
    assert task4.render_baseline_examples is not None


def test_baseline_runner_help_uses_only_the_thin_public_cli() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    environment.update(
        dict.fromkeys(thread_variables, "9")
    )

    result = subprocess.run(
        [sys.executable, str(root / "scripts/task4/run_baseline.py"), "--help"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Rebuild development-only Task 4 baseline evidence." in result.stdout
    assert "--workers" in result.stdout
    assert result.stderr == ""

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, runpy; "
                "runpy.run_path('scripts/task4/run_baseline.py', run_name='task5_probe'); "
                f"print('|'.join(os.environ[name] for name in {thread_variables!r}))"
            ),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() == "1|1|1|1"
