"""Frozen Task 4 baseline quality and sanity-floor evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    PairEvaluation,
    SourceName,
    evaluate_source_pair,
    source_directions,
)
from fashion.task4.probe import PROBE_VERSION
from fashion.task4.protocol import (
    RetrievalViews,
    build_development_views,
    evaluate_primary_rankings,
    prepare_rankings,
)

Direction: TypeAlias = tuple[SourceName, SourceName]

BASELINE_QUALITY_CHUNK_SIZE = 256
PREPROCESSING_REPRODUCTION_ATOL = 1e-5
_BASELINE_CONTRACT = PreprocessingContract(width=240, height=320)
_DIRECTIONS = source_directions()
_PREFIX_COLUMNS = (
    "method",
    "fold",
    "size",
    "query_source",
    "gallery_source",
    "protocol",
    "scope",
)

__all__ = (
    "BASELINE_QUALITY_CHUNK_SIZE",
    "PREPROCESSING_REPRODUCTION_ATOL",
    "BaselineEvaluation",
    "Direction",
    "build_baseline_summary",
    "build_headline_summary",
    "build_query_metrics",
    "build_random_primary_rankings",
    "evaluate_baseline",
    "verify_preprocessing_reproduction",
)


@dataclass(frozen=True)
class BaselineEvaluation:
    """Complete frozen-baseline quality evidence."""

    summary: pd.DataFrame
    query_metrics: pd.DataFrame
    pair_evaluations: dict[Direction, PairEvaluation]
    random_rankings: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    if missing := required.difference(frame.columns):
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _require_directions(
    pair_evaluations: Mapping[Direction, PairEvaluation],
) -> None:
    if set(pair_evaluations) != set(_DIRECTIONS):
        raise ValueError("baseline evidence requires exactly the four source directions")


def _single_value(series: pd.Series, *, label: str) -> object:
    values = series.drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"baseline evidence requires one {label}")
    return values.iloc[0]


def _pair_context(
    direction: Direction,
    evaluation: PairEvaluation,
) -> tuple[int, str]:
    _require_columns(
        evaluation.summary,
        {"fold", "size", "query_source", "gallery_source", "protocol"},
        label="pair summary",
    )
    fold = int(_single_value(evaluation.summary["fold"], label="fold"))
    size = str(_single_value(evaluation.summary["size"], label="size"))
    if (
        set(evaluation.summary["query_source"]) != {direction[0]}
        or set(evaluation.summary["gallery_source"]) != {direction[1]}
        or set(evaluation.summary["protocol"]) != {"primary", "family"}
    ):
        raise ValueError("pair summary source or protocol metadata is malformed")
    return fold, size


def _prefix(
    frame: pd.DataFrame,
    *,
    method: str,
    fold: int,
    size: str,
    query_source: str,
    gallery_source: str,
    protocol: str | None = None,
) -> pd.DataFrame:
    labelled = frame.copy()
    values: dict[str, object] = {
        "method": method,
        "fold": int(fold),
        "size": size,
        "query_source": query_source,
        "gallery_source": gallery_source,
    }
    if protocol is not None:
        values["protocol"] = protocol
    values["scope"] = "development"
    for position, (column, value) in enumerate(values.items()):
        if column in labelled:
            labelled[column] = value
        else:
            labelled.insert(position, column, value)
    prefix = [column for column in _PREFIX_COLUMNS if column in labelled]
    remaining = [column for column in labelled if column not in prefix]
    return labelled.loc[:, [*prefix, *remaining]]


def build_random_primary_rankings(
    views: RetrievalViews,
    *,
    seed: int = 2753,
    max_k: int = 20,
) -> pd.DataFrame:
    """Build one fixed Protocol A gallery permutation for every query."""
    gallery_ids = (
        pd.to_numeric(views.gallery["id"], errors="raise")
        .astype(np.int64)
        .sort_values()
        .to_numpy()
    )
    ordered_gallery = np.random.default_rng(seed).permutation(gallery_ids)
    selected = ordered_gallery[:max_k]
    records = [
        {
            "query_id": int(query_id),
            "candidate_id": int(candidate_id),
            "distance": float(position),
        }
        for query_id in sorted(pd.to_numeric(views.queries["id"], errors="raise"))
        for position, candidate_id in enumerate(selected, start=1)
    ]
    return prepare_rankings(
        pd.DataFrame.from_records(records),
        views,
        protocol="primary",
        max_k=max_k,
    )


def build_query_metrics(
    pair_evaluations: Mapping[Direction, PairEvaluation],
) -> pd.DataFrame:
    """Combine per-query values with their frozen baseline context."""
    _require_directions(pair_evaluations)
    frames: list[pd.DataFrame] = []
    for direction in _DIRECTIONS:
        evaluation = pair_evaluations[direction]
        fold, size = _pair_context(direction, evaluation)
        for protocol, per_query in (
            ("primary", evaluation.primary_per_query),
            ("family", evaluation.family_per_query),
        ):
            frames.append(
                _prefix(
                    per_query,
                    method=PROBE_VERSION,
                    fold=fold,
                    size=size,
                    query_source=direction[0],
                    gallery_source=direction[1],
                    protocol=protocol,
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_headline_summary(
    *,
    teacher_ndcg: float,
    v1_ndcg: float,
    teacher_to_v1_ndcg: float,
    v1_to_teacher_ndcg: float,
    random_ndcg: float,
) -> dict[str, float | bool]:
    """Calculate equal-source means and record both baseline hypotheses."""
    values = np.asarray(
        [
            teacher_ndcg,
            v1_ndcg,
            teacher_to_v1_ndcg,
            v1_to_teacher_ndcg,
            random_ndcg,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("headline nDCG values must be finite")
    same_source_mean = float((values[0] + values[1]) / 2.0)
    cross_source_mean = float((values[2] + values[3]) / 2.0)
    return {
        "same_source_mean": same_source_mean,
        "cross_source_mean": cross_source_mean,
        "beats_random": bool(same_source_mean > values[4]),
        "cross_source_within_95_percent": bool(
            cross_source_mean >= 0.95 * same_source_mean
        ),
    }


def _primary_ndcg_at_10(summary: pd.DataFrame, direction: Direction) -> float:
    selected = summary.loc[
        summary["protocol"].eq("primary")
        & summary["metric"].eq("ndcg")
        & summary["k"].eq(10)
        & summary["aggregation"].eq("query_mean")
        & summary["query_source"].eq(direction[0])
        & summary["gallery_source"].eq(direction[1])
    ]
    if len(selected) != 1:
        raise ValueError(f"baseline summary has malformed nDCG@10 for {direction}")
    value = float(selected["value"].item())
    if not np.isfinite(value):
        raise ValueError(f"baseline summary has non-finite nDCG@10 for {direction}")
    return value


def _headline_rows(
    *,
    headline: Mapping[str, float | bool],
    random_ndcg: float,
    fold: int,
    size: str,
) -> pd.DataFrame:
    same_source_mean = float(headline["same_source_mean"])
    cross_source_mean = float(headline["cross_source_mean"])
    ratio = (
        cross_source_mean / same_source_mean if same_source_mean > 0 else np.nan
    )
    common: dict[str, object] = {
        "method": "headline",
        "scope": "development",
        "fold": fold,
        "size": size,
        "query_source": "all",
        "gallery_source": "all",
        "protocol": "primary",
        "k": 10,
        "aggregation": "query_mean",
        "query_count": pd.NA,
        "class_count": pd.NA,
    }
    return pd.DataFrame.from_records(
        [
            {
                **common,
                "metric": "same_source_mean",
                "value": same_source_mean,
                "passed": pd.NA,
            },
            {
                **common,
                "metric": "cross_source_mean",
                "value": cross_source_mean,
                "passed": pd.NA,
            },
            {
                **common,
                "metric": "beats_random",
                "value": same_source_mean - random_ndcg,
                "passed": bool(headline["beats_random"]),
            },
            {
                **common,
                "metric": "cross_source_within_95_percent",
                "value": ratio,
                "passed": bool(headline["cross_source_within_95_percent"]),
            },
        ]
    )


def build_baseline_summary(
    pair_evaluations: Mapping[Direction, PairEvaluation],
    random_rankings: pd.DataFrame,
    primary_views: RetrievalViews,
) -> pd.DataFrame:
    """Combine four directions, the random floor, and headline checks."""
    _require_directions(pair_evaluations)
    pair_frames: list[pd.DataFrame] = []
    contexts: set[tuple[int, str]] = set()
    for direction in _DIRECTIONS:
        evaluation = pair_evaluations[direction]
        fold, size = _pair_context(direction, evaluation)
        contexts.add((fold, size))
        pair_frames.append(
            _prefix(
                evaluation.summary,
                method=PROBE_VERSION,
                fold=fold,
                size=size,
                query_source=direction[0],
                gallery_source=direction[1],
            )
        )
    if len(contexts) != 1:
        raise ValueError("baseline pair summaries must share one fold and size")
    fold, size = contexts.pop()
    pair_summary = pd.concat(pair_frames, ignore_index=True)

    primary_rows = pair_summary.loc[pair_summary["protocol"].eq("primary")]
    k_values = tuple(
        sorted(
            int(value)
            for value in primary_rows.loc[
                primary_rows["metric"].eq("ndcg"), "k"
            ].unique()
        )
    )
    if 10 not in k_values:
        raise ValueError("baseline evidence requires Protocol A nDCG@10")
    _, random_summary = evaluate_primary_rankings(
        random_rankings,
        primary_views,
        k_values=k_values,
    )
    random_summary = _prefix(
        random_summary,
        method="random-seed-2753",
        fold=fold,
        size=size,
        query_source="random",
        gallery_source="random",
        protocol="primary",
    )
    random_ndcg = _primary_ndcg_at_10(
        random_summary,
        ("random", "random"),  # type: ignore[arg-type]
    )
    directional_ndcg = {
        direction: _primary_ndcg_at_10(pair_summary, direction)
        for direction in _DIRECTIONS
    }
    headline = build_headline_summary(
        teacher_ndcg=directional_ndcg[("teacher", "teacher")],
        v1_ndcg=directional_ndcg[("v1", "v1")],
        teacher_to_v1_ndcg=directional_ndcg[("teacher", "v1")],
        v1_to_teacher_ndcg=directional_ndcg[("v1", "teacher")],
        random_ndcg=random_ndcg,
    )
    headline_rows = _headline_rows(
        headline=headline,
        random_ndcg=random_ndcg,
        fold=fold,
        size=size,
    )
    combined = pd.concat(
        [pair_summary, random_summary, headline_rows],
        ignore_index=True,
        sort=False,
    )
    remaining = [column for column in combined if column not in _PREFIX_COLUMNS]
    return combined.loc[:, [*_PREFIX_COLUMNS, *remaining]]


def evaluate_baseline(
    splits: pd.DataFrame,
    indexes: Mapping[SourceName, FeatureIndex],
    *,
    fold: int = 1,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
) -> BaselineEvaluation:
    """Evaluate the frozen probe in all four approved source directions."""
    if fold != 1:
        raise ValueError("the frozen baseline uses validation fold 1")
    if set(indexes) != {"teacher", "v1"}:
        raise ValueError("baseline evaluation requires exactly teacher and v1 indexes")
    for source in ("teacher", "v1"):
        index = indexes[source]
        if index.source != source:
            raise ValueError("feature index source does not match its baseline key")
        if index.contract != _BASELINE_CONTRACT:
            raise ValueError("baseline feature indexes must use the frozen 240x320 contract")

    primary, family = build_development_views(splits, validation_fold=fold)
    pairs = {
        direction: evaluate_source_pair(
            indexes[direction[0]],
            indexes[direction[1]],
            primary_views=primary,
            family_views=family,
            fold=fold,
            k_values=k_values,
            family_k=family_k,
            chunk_size=BASELINE_QUALITY_CHUNK_SIZE,
        )
        for direction in _DIRECTIONS
    }
    random_rankings = build_random_primary_rankings(
        primary,
        seed=2753,
        max_k=max(k_values),
    )
    return BaselineEvaluation(
        summary=build_baseline_summary(pairs, random_rankings, primary),
        query_metrics=build_query_metrics(pairs),
        pair_evaluations=pairs,
        random_rankings=random_rankings,
    )


def _reproduction_values(
    frame: pd.DataFrame,
    *,
    label: str,
    require_method: bool,
) -> pd.Series:
    required = {
        "scope",
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "protocol",
        "metric",
        "k",
        "aggregation",
        "value",
    }
    if require_method:
        required.add("method")
    _require_columns(frame, required, label=label)
    selected = frame.loc[
        frame["scope"].eq("development")
        & frame["fold"].eq(1)
        & frame["size"].eq("240x320")
        & frame["protocol"].eq("primary")
        & frame["metric"].eq("ndcg")
        & frame["k"].eq(10)
        & frame["aggregation"].eq("query_mean")
    ].copy()
    if require_method:
        selected = selected.loc[selected["method"].eq(PROBE_VERSION)]
    selected["direction"] = list(
        zip(
            selected["query_source"],
            selected["gallery_source"],
            strict=True,
        )
    )
    if set(selected["direction"]) != set(_DIRECTIONS) or selected[
        "direction"
    ].duplicated().any():
        raise ValueError(f"{label} has malformed preprocessing probe directions")
    values = pd.to_numeric(
        selected.set_index("direction")["value"],
        errors="coerce",
    ).reindex(_DIRECTIONS)
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{label} has malformed preprocessing probe values")
    return values


def verify_preprocessing_reproduction(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    atol: float = PREPROCESSING_REPRODUCTION_ATOL,
) -> None:
    """Require bounded reproduction of the prior fold-1 240×320 probe scores."""
    if isinstance(atol, bool) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    observed = _reproduction_values(
        summary,
        label="baseline summary",
        require_method=True,
    )
    expected = _reproduction_values(
        comparison,
        label="preprocessing comparison",
        require_method=False,
    )
    observed_values = observed.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    deltas = np.abs(observed_values - expected_values)
    mismatches = [
        (
            direction,
            observed_value,
            expected_value,
            delta,
        )
        for direction, observed_value, expected_value, delta in zip(
            _DIRECTIONS,
            observed_values,
            expected_values,
            deltas,
            strict=True,
        )
        if delta > float(atol)
    ]
    if mismatches:
        details = "\n".join(
            f"- {direction[0]}->{direction[1]}: "
            f"observed={observed_value:.12g}, "
            f"expected={expected_value:.12g}, "
            f"absolute_delta={delta:.12g}"
            for direction, observed_value, expected_value, delta in mismatches
        )
        raise ValueError(
            "baseline does not reproduce the preprocessing probe values:\n"
            f"{details}"
        )
