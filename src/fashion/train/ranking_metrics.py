"""Family-blocked paired bootstrap for per-query ranking scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd

RANKING_BOOTSTRAP_COLUMNS: tuple[str, ...] = (
    "method",
    "replicate",
    "sampled_group_count",
    "sampled_query_count",
    "scored_query_count",
    "mean_score",
    "mean_minus_reference",
)

RANKING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "metric",
    "lower_95",
    "median",
    "upper_95",
    "replicates",
    "random_seed",
    "sampled_group_count",
)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def paired_family_ranking_bootstrap(
    scores: Mapping[str, Sequence[float] | np.ndarray],
    groups: Sequence[object] | np.ndarray,
    *,
    reference: str,
    replicates: int = 10_000,
    random_seed: int = 2753,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Resample whole product families and compare paired per-query ranking means.

    ``scores`` maps a method name to one score per query, aligned to ``groups``.
    ``numpy.nan`` marks a query whose metric is undefined; such queries are excluded
    from the mean rather than counted as zero. One multiplicity draw is shared by every
    method, so paired differences stay paired.
    """
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a non-empty mapping")
    replicate_count = _positive_int(replicates, "replicates")
    draw_batch_size = _positive_int(batch_size, "batch_size")
    seed = _positive_int(random_seed, "random_seed")

    group_values = np.asarray(groups, dtype=object)
    if group_values.ndim != 1 or group_values.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional sequence")
    group_labels, group_indices = np.unique(group_values, return_inverse=True)
    group_count = int(group_labels.size)
    if group_count < 2:
        raise ValueError("groups must contain at least two unique values")

    method_names = sorted(str(name) for name in scores)
    if reference not in method_names:
        raise ValueError("reference method is not present in scores")

    counts = np.zeros((group_count, len(method_names)), dtype=np.int64)
    sums = np.zeros((group_count, len(method_names)), dtype=np.float64)
    for column, name in enumerate(method_names):
        values = np.asarray(scores[name], dtype=np.float64)
        if values.shape != group_values.shape:
            raise ValueError("score arrays must match the group length")
        scorable = np.isfinite(values)
        if not scorable.any():
            raise ValueError("every method needs at least one scorable query")
        np.add.at(counts[:, column], group_indices[scorable], 1)
        np.add.at(sums[:, column], group_indices[scorable], values[scorable])
    rows_per_group = np.zeros(group_count, dtype=np.int64)
    np.add.at(rows_per_group, group_indices, 1)

    reference_column = method_names.index(reference)
    probabilities = np.full(group_count, 1.0 / group_count, dtype=np.float64)
    probabilities[-1] = 1.0 - probabilities[:-1].sum()

    means = np.empty((replicate_count, len(method_names)), dtype=np.float64)
    scored = np.empty((replicate_count, len(method_names)), dtype=np.int64)
    sampled_rows = np.empty(replicate_count, dtype=np.int64)

    generator = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, replicate_count, draw_batch_size):
        stop = min(start + draw_batch_size, replicate_count)
        multiplicities = generator.multinomial(
            group_count, probabilities, size=stop - start
        )
        batch_counts = multiplicities @ counts
        batch_sums = multiplicities @ sums
        if not (batch_counts > 0).all():
            raise ValueError("a resampled draw produced no scorable query for a method")
        means[start:stop, :] = batch_sums / batch_counts
        scored[start:stop, :] = batch_counts
        sampled_rows[start:stop] = multiplicities @ rows_per_group

    replicate_ids = np.arange(1, replicate_count + 1, dtype=np.int64)
    frames: list[pd.DataFrame] = []
    for column, name in enumerate(method_names):
        frames.append(
            pd.DataFrame(
                {
                    "method": name,
                    "replicate": replicate_ids,
                    "sampled_group_count": np.full(replicate_count, group_count, dtype=np.int64),
                    "sampled_query_count": sampled_rows,
                    "scored_query_count": scored[:, column],
                    "mean_score": means[:, column],
                    "mean_minus_reference": means[:, column] - means[:, reference_column],
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.loc[:, list(RANKING_BOOTSTRAP_COLUMNS)]


def summarise_ranking_bootstrap(
    draws: pd.DataFrame,
    *,
    value_column: str,
    random_seed: int = 2753,
) -> pd.DataFrame:
    """Return empirical 2.5th, 50th, and 97.5th percentiles for every method."""
    seed = _positive_int(random_seed, "random_seed")
    missing = {"method", "replicate", "sampled_group_count", value_column}.difference(
        draws.columns
    )
    if missing:
        raise ValueError(f"bootstrap draws are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for method, frame in draws.groupby("method", sort=True):
        quantiles = np.quantile(
            frame[value_column].to_numpy(dtype=np.float64), [0.025, 0.5, 0.975]
        )
        rows.append(
            {
                "metric": f"{method}_{value_column}",
                "lower_95": float(quantiles[0]),
                "median": float(quantiles[1]),
                "upper_95": float(quantiles[2]),
                "replicates": int(frame["replicate"].nunique()),
                "random_seed": seed,
                "sampled_group_count": int(frame["sampled_group_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=list(RANKING_SUMMARY_COLUMNS))


__all__ = [
    "RANKING_BOOTSTRAP_COLUMNS",
    "RANKING_SUMMARY_COLUMNS",
    "paired_family_ranking_bootstrap",
    "summarise_ranking_bootstrap",
]
