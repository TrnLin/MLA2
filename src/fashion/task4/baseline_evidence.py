"""Deterministic tracked evidence for the frozen Task 4 baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fashion.config import ROOT
from fashion.data.hashing import write_deterministic_csv
from fashion.task4.baseline import BaselineEvaluation
from fashion.task4.preprocessing_experiment import build_odd_aspect_canvas

SUMMARY_COLUMNS = (
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
)
QUERY_METRIC_COLUMNS = (
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
)
FAILURE_SLICE_COLUMNS = (
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
)
TIMING_COLUMNS = (
    "query_id",
    "query_source",
    "gallery_source",
    "encoding_seconds",
    "search_seconds",
    "end_to_end_seconds",
)
EXAMPLE_COLUMNS = (
    "scope",
    "fold",
    "size",
    "query_source",
    "gallery_source",
    "slice",
    "query_variant",
    "query_id",
    "metric",
    "value",
    "rank",
    "candidate_id",
    "distance",
)

_ARTIFACT_NAMES = (
    "baseline_summary.csv",
    "baseline_query_metrics.csv",
    "baseline_failure_slices.csv",
    "baseline_timing.csv",
    "baseline_cost.json",
    "baseline_examples.csv",
)
_SLICE_ORDER = (
    "grayscale",
    "rare_article_type",
    "rare_type_colour",
    "unusual_geometry",
    "family_unavailable",
    "weak_family",
    "canvas_clean",
    "canvas_wide",
    "canvas_tall",
)
_EXAMPLE_ORDER = (
    "normal_success",
    "grayscale",
    "unusual_geometry",
    "canvas_failure",
    "weak_family",
    "rare_article_type",
    "rare_type_colour",
    "family_unavailable",
)

__all__ = (
    "EXAMPLE_COLUMNS",
    "FAILURE_SLICE_COLUMNS",
    "QUERY_METRIC_COLUMNS",
    "SUMMARY_COLUMNS",
    "TIMING_COLUMNS",
    "render_baseline_examples",
    "write_baseline_artifacts",
)


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise ValueError(
            f"{label} columns must be {tuple(expected)}, found {tuple(frame.columns)}"
        )


def _require_development_scope(frame: pd.DataFrame, *, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} must not be empty")
    if "scope" in frame and set(frame["scope"].astype(str)) != {"development"}:
        raise ValueError(f"{label} scope must be development")
    if "partition" in frame and frame["partition"].isin({"holdout", "quarantine"}).any():
        raise ValueError(f"{label} contains a protected partition")


def _require_frozen_context(frame: pd.DataFrame, *, label: str) -> None:
    if "fold" in frame and set(pd.to_numeric(frame["fold"], errors="coerce")) != {1}:
        raise ValueError(f"{label} must use frozen fold 1")
    if "size" in frame and set(frame["size"].astype(str)) != {"240x320"}:
        raise ValueError(f"{label} must use frozen size 240x320")


def _require_finite_non_null(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        present = frame[column].notna()
        if numeric[present].isna().any() or not np.isfinite(numeric[present]).all():
            raise ValueError(f"{label} {column} values must be finite when present")


def _ordered_categories(
    frame: pd.DataFrame,
    *,
    column: str,
    categories: Sequence[str],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    unknown = sorted(set(frame[column].astype(str)).difference(categories))
    if unknown:
        raise ValueError(f"unknown {column} values: {unknown}")
    order = pd.Categorical(frame[column], categories=categories, ordered=True)
    return (
        frame.assign(_fixed_order=order)
        .sort_values([*sort_columns, "_fixed_order"], kind="mergesort")
        .drop(columns="_fixed_order")
        .reset_index(drop=True)
    )


def _validate_examples(examples: pd.DataFrame) -> pd.DataFrame:
    _require_exact_columns(examples, EXAMPLE_COLUMNS, label="baseline examples")
    _require_development_scope(examples, label="baseline examples")
    _require_frozen_context(examples, label="baseline examples")
    if set(examples["query_source"]) != {"v1"} or set(examples["gallery_source"]) != {
        "v1"
    }:
        raise ValueError("baseline examples must use V1-to-V1 retrieval")
    if not examples["query_variant"].isin({"clean", "wide", "tall"}).all():
        raise ValueError("baseline examples have an unknown query variant")
    _require_finite_non_null(
        examples,
        ("fold", "query_id", "value", "rank", "candidate_id", "distance"),
        label="baseline examples",
    )
    ranks = pd.to_numeric(examples["rank"], errors="coerce")
    if not ranks.mod(1).eq(0).all() or not ranks.between(1, 5).all():
        raise ValueError("baseline examples may persist Top-5 ranks only")
    for slice_name, rows in examples.groupby("slice", sort=False):
        if rows["rank"].astype(int).sort_values().tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"baseline example {slice_name} requires ranks 1 through 5")
        invariant = [
            "scope",
            "fold",
            "size",
            "query_source",
            "gallery_source",
            "query_variant",
            "query_id",
            "metric",
            "value",
        ]
        if any(rows[column].nunique(dropna=False) != 1 for column in invariant):
            raise ValueError(f"baseline example {slice_name} has inconsistent metadata")
    return _ordered_categories(
        examples,
        column="slice",
        categories=_EXAMPLE_ORDER,
        sort_columns=(),
    ).sort_values(["slice", "rank"], kind="mergesort", key=_example_sort_key).reset_index(
        drop=True
    )


def _example_sort_key(series: pd.Series) -> pd.Series:
    if series.name == "slice":
        return pd.Categorical(series, categories=_EXAMPLE_ORDER, ordered=True)
    return series


def _resolve_image_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_display_image(path: Path, query_variant: str) -> Image.Image:
    with Image.open(path) as image:
        if query_variant in {"wide", "tall"}:
            return build_odd_aspect_canvas(image, query_variant).copy()
        return image.convert("RGB").copy()


def _validate_example_rankings(
    examples: pd.DataFrame,
    example_rankings: Mapping[str, pd.DataFrame],
) -> None:
    expected_slices = set(examples["slice"].astype(str))
    if set(example_rankings) != expected_slices:
        raise ValueError("example rankings must match the selected example slices")
    for slice_name, persisted in examples.groupby("slice", sort=False):
        ranking = example_rankings[str(slice_name)]
        required = {"query_id", "candidate_id", "distance", "rank"}
        if missing := required.difference(ranking.columns):
            raise ValueError(f"example ranking is missing columns: {sorted(missing)}")
        expected = persisted.loc[
            :, ["query_id", "candidate_id", "distance", "rank"]
        ].sort_values("rank")
        query_id = int(persisted["query_id"].iloc[0])
        observed = ranking.loc[
            ranking["query_id"].eq(query_id) & ranking["rank"].between(1, 5),
            expected.columns,
        ].sort_values("rank")
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )


def render_baseline_examples(
    examples: pd.DataFrame,
    image_rows: pd.DataFrame,
    example_rankings: Mapping[str, pd.DataFrame],
    destination: str | Path,
) -> None:
    """Render fixed V1-to-V1 query and Top-5 rows from selected evidence."""
    ordered = _validate_examples(examples)
    _require_development_scope(image_rows, label="example image rows")
    if tuple(column for column in ("id", "external_path") if column not in image_rows):
        raise ValueError("example image rows require id and external_path")
    ids = pd.to_numeric(image_rows["id"], errors="coerce")
    if ids.isna().any() or not ids.mod(1).eq(0).all() or ids.duplicated().any():
        raise ValueError("example image row IDs must be unique integers")
    lookup = image_rows.assign(_numeric_id=ids.astype(int)).set_index("_numeric_id")
    needed_ids = set(ordered["query_id"].astype(int)) | set(
        ordered["candidate_id"].astype(int)
    )
    if missing := sorted(needed_ids.difference(lookup.index)):
        raise ValueError(f"example image rows are missing IDs: {missing}")
    _validate_example_rankings(ordered, example_rankings)

    row_groups = list(ordered.groupby("slice", sort=False))
    figure, axes = plt.subplots(
        len(row_groups),
        6,
        figsize=(12.0, 14.4),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (slice_name, rows) in enumerate(row_groups):
        rows = rows.sort_values("rank")
        first = rows.iloc[0]
        query_id = int(first["query_id"])
        variant = str(first["query_variant"])
        query_path = _resolve_image_path(lookup.loc[query_id, "external_path"])
        axes[row_index, 0].imshow(_load_display_image(query_path, variant))
        axes[row_index, 0].set_title(f"Query {query_id}\n{variant}", fontsize=8)
        score = (
            f"{float(first['value']):.3f}"
            if pd.notna(first["value"])
            else "not scorable"
        )
        axes[row_index, 0].set_ylabel(
            f"{slice_name}\n{first['metric']}={score}",
            fontsize=8,
        )
        result_ids: list[int] = []
        for column_index, result in enumerate(rows.itertuples(index=False), start=1):
            candidate_id = int(result.candidate_id)
            result_ids.append(candidate_id)
            candidate_path = _resolve_image_path(
                lookup.loc[candidate_id, "external_path"]
            )
            axes[row_index, column_index].imshow(
                _load_display_image(candidate_path, "clean")
            )
            axes[row_index, column_index].set_title(
                f"#{int(result.rank)} · {candidate_id}",
                fontsize=8,
            )
        axes[row_index, 0].set_xlabel(
            "Results: " + ", ".join(str(value) for value in result_ids),
            fontsize=7,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Task 4 fixed probe — V1 query to V1 Top-5", fontsize=11)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, metadata={"Software": "MLA2"})
    plt.close(figure)


def _prepare_frames(
    *,
    baseline: BaselineEvaluation,
    slice_summary: pd.DataFrame,
    timings: pd.DataFrame,
    examples: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    _require_exact_columns(baseline.summary, SUMMARY_COLUMNS, label="baseline summary")
    _require_exact_columns(
        baseline.query_metrics,
        QUERY_METRIC_COLUMNS,
        label="baseline query metrics",
    )
    _require_exact_columns(
        slice_summary,
        FAILURE_SLICE_COLUMNS,
        label="baseline failure slices",
    )
    _require_exact_columns(timings, TIMING_COLUMNS, label="baseline timings")
    for label, frame in (
        ("baseline summary", baseline.summary),
        ("baseline query metrics", baseline.query_metrics),
        ("baseline failure slices", slice_summary),
        ("baseline timings", timings),
    ):
        _require_development_scope(frame, label=label)
        _require_frozen_context(frame, label=label)
    _require_finite_non_null(
        baseline.summary,
        ("value",),
        label="baseline summary",
    )
    _require_finite_non_null(
        baseline.query_metrics,
        QUERY_METRIC_COLUMNS[9:],
        label="baseline query metrics",
    )
    _require_finite_non_null(
        slice_summary,
        ("value", "coverage"),
        label="baseline failure slices",
    )
    _require_finite_non_null(
        timings,
        TIMING_COLUMNS[3:],
        label="baseline timings",
    )
    if timings.loc[:, TIMING_COLUMNS[3:]].lt(0).any().any():
        raise ValueError("baseline timing values must be non-negative")

    ordered_slices = _ordered_categories(
        slice_summary,
        column="slice",
        categories=_SLICE_ORDER,
        sort_columns=("query_source", "gallery_source", "protocol"),
    )
    return {
        "baseline_summary.csv": baseline.summary.sort_values(
            [
                "method",
                "query_source",
                "gallery_source",
                "protocol",
                "metric",
                "k",
                "aggregation",
            ],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True),
        "baseline_query_metrics.csv": baseline.query_metrics.sort_values(
            ["query_source", "gallery_source", "protocol", "query_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        "baseline_failure_slices.csv": ordered_slices,
        "baseline_timing.csv": timings.sort_values(
            ["query_source", "gallery_source", "query_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        "baseline_examples.csv": _validate_examples(examples),
    }


def _validate_cost(cost: Mapping[str, object]) -> None:
    required = {
        "schema_version": 1,
        "scope": "development",
        "parameters": 0,
        "checkpoint_bytes": 0,
    }
    for key, expected in required.items():
        if cost.get(key) != expected:
            raise ValueError(f"baseline cost {key} must equal {expected!r}")
    if "timing_summary" not in cost:
        raise ValueError("baseline cost must contain timing percentiles")
    try:
        json.dumps(cost, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("baseline cost must contain finite JSON-safe values") from error


def _reopen_artifacts(
    evidence_path: Path,
    figure_path: Path,
    expected_frames: Mapping[str, pd.DataFrame],
) -> None:
    for name in _ARTIFACT_NAMES:
        path = evidence_path / name
        if not path.is_file() or not path.read_bytes().endswith(b"\n"):
            raise ValueError(f"baseline artifact is missing or not newline-terminated: {name}")
    frames = {
        name: pd.read_csv(evidence_path / name)
        for name in _ARTIFACT_NAMES
        if name.endswith(".csv")
    }
    expected_columns = {
        "baseline_summary.csv": SUMMARY_COLUMNS,
        "baseline_query_metrics.csv": QUERY_METRIC_COLUMNS,
        "baseline_failure_slices.csv": FAILURE_SLICE_COLUMNS,
        "baseline_timing.csv": TIMING_COLUMNS,
        "baseline_examples.csv": EXAMPLE_COLUMNS,
    }
    for name, frame in frames.items():
        _require_exact_columns(frame, expected_columns[name], label=name)
        _require_development_scope(frame, label=name)
        _require_frozen_context(frame, label=name)
        if len(frame) != len(expected_frames[name]):
            raise ValueError(f"{name} row count changed after writing")
    _require_finite_non_null(
        frames["baseline_summary.csv"],
        ("value",),
        label="baseline_summary.csv",
    )
    _require_finite_non_null(
        frames["baseline_query_metrics.csv"],
        QUERY_METRIC_COLUMNS[9:],
        label="baseline_query_metrics.csv",
    )
    _require_finite_non_null(
        frames["baseline_failure_slices.csv"],
        ("value", "coverage"),
        label="baseline_failure_slices.csv",
    )
    _require_finite_non_null(
        frames["baseline_timing.csv"],
        TIMING_COLUMNS[3:],
        label="baseline_timing.csv",
    )
    _validate_examples(frames["baseline_examples.csv"])
    _validate_cost(
        json.loads((evidence_path / "baseline_cost.json").read_text(encoding="utf-8"))
    )
    with Image.open(figure_path / "baseline_examples.png") as figure:
        figure.verify()


def write_baseline_artifacts(
    *,
    baseline: BaselineEvaluation,
    slice_summary: pd.DataFrame,
    timings: pd.DataFrame,
    cost: Mapping[str, object],
    examples: pd.DataFrame,
    evidence_dir: str | Path,
    figure_dir: str | Path,
    image_rows: pd.DataFrame,
    example_rankings: Mapping[str, pd.DataFrame],
) -> None:
    """Write and reopen the complete small development-only baseline evidence set."""
    frames = _prepare_frames(
        baseline=baseline,
        slice_summary=slice_summary,
        timings=timings,
        examples=examples,
    )
    _validate_cost(cost)
    evidence_path = Path(evidence_dir)
    figure_path = Path(figure_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        write_deterministic_csv(
            frame,
            evidence_path / name,
            index=False,
            float_format="%.8f",
        )
    (evidence_path / "baseline_cost.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    render_baseline_examples(
        frames["baseline_examples.csv"],
        image_rows,
        example_rankings,
        figure_path / "baseline_examples.png",
    )
    _reopen_artifacts(evidence_path, figure_path, frames)
