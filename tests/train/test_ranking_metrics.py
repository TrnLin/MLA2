from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fashion.train.ranking_metrics import (
    paired_family_ranking_bootstrap,
    summarise_ranking_bootstrap,
)


def _scores() -> dict[str, np.ndarray]:
    return {
        "r5": np.array([0.9, 0.8, 0.7, np.nan, 0.6, 0.5], dtype=float),
        "probe": np.array([0.4, 0.3, 0.2, np.nan, 0.1, 0.0], dtype=float),
    }


def _groups() -> np.ndarray:
    return np.array(["f1", "f1", "f2", "f2", "f3", "f4"], dtype=object)


def test_bootstrap_is_deterministic_for_one_seed() -> None:
    first = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=64, random_seed=2753
    )
    second = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=64, random_seed=2753
    )
    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_is_batch_invariant() -> None:
    small = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=64, random_seed=2753, batch_size=7
    )
    large = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=64, random_seed=2753, batch_size=64
    )
    pd.testing.assert_frame_equal(small, large)


def test_bootstrap_returns_one_row_per_method_and_replicate() -> None:
    draws = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=16, random_seed=2753
    )
    assert list(draws.columns) == [
        "method",
        "replicate",
        "sampled_group_count",
        "sampled_query_count",
        "scored_query_count",
        "mean_score",
        "mean_minus_reference",
    ]
    assert len(draws) == 2 * 16
    assert sorted(draws["method"].unique()) == ["probe", "r5"]
    assert draws["replicate"].min() == 1
    assert draws["replicate"].max() == 16
    assert (draws["sampled_group_count"] == 4).all()


def test_reference_method_has_zero_paired_difference() -> None:
    draws = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=16, random_seed=2753
    )
    reference = draws.loc[draws["method"].eq("probe")]
    assert reference["mean_minus_reference"].abs().max() == 0.0


def test_undefined_queries_are_excluded_not_zero_filled() -> None:
    draws = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=8, random_seed=2753
    )
    assert draws["mean_score"].min() > 0.0
    assert (draws["scored_query_count"] < draws["sampled_query_count"]).any()


def test_whole_groups_are_resampled_together() -> None:
    scores = {
        "a": np.array([1.0, 1.0, 0.0, 0.0], dtype=float),
        "b": np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
    }
    groups = np.array(["hi", "hi", "lo", "lo"], dtype=object)
    draws = paired_family_ranking_bootstrap(
        scores, groups, reference="b", replicates=200, random_seed=2753
    )
    observed = set(np.round(draws.loc[draws["method"].eq("a"), "mean_score"].to_numpy(), 6))
    assert observed <= {0.0, 0.5, 1.0}


def test_summary_reports_percentile_endpoints() -> None:
    draws = paired_family_ranking_bootstrap(
        _scores(), _groups(), reference="probe", replicates=1000, random_seed=2753
    )
    summary = summarise_ranking_bootstrap(draws, value_column="mean_score")
    assert list(summary.columns) == [
        "metric",
        "lower_95",
        "median",
        "upper_95",
        "replicates",
        "random_seed",
        "sampled_group_count",
    ]
    row = summary.loc[summary["metric"].eq("r5_mean_score")].iloc[0]
    expected = np.quantile(
        draws.loc[draws["method"].eq("r5"), "mean_score"].to_numpy(), [0.025, 0.5, 0.975]
    )
    assert row["lower_95"] == pytest.approx(expected[0], abs=1e-12)
    assert row["median"] == pytest.approx(expected[1], abs=1e-12)
    assert row["upper_95"] == pytest.approx(expected[2], abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference": "missing"}, "reference method is not present"),
        ({"reference": "probe", "replicates": 0}, "replicates must be a positive integer"),
        ({"reference": "probe", "batch_size": 0}, "batch_size must be a positive integer"),
    ],
)
def test_bootstrap_rejects_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_family_ranking_bootstrap(_scores(), _groups(), **kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_misaligned_or_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="scores must be a non-empty mapping"):
        paired_family_ranking_bootstrap({}, _groups(), reference="r5")
    with pytest.raises(ValueError, match="score arrays must match the group length"):
        paired_family_ranking_bootstrap(
            {"r5": np.array([0.1, 0.2])}, _groups(), reference="r5"
        )
    with pytest.raises(ValueError, match="groups must contain at least two unique values"):
        paired_family_ranking_bootstrap(
            {"r5": np.array([0.1, 0.2])}, np.array(["f", "f"], dtype=object), reference="r5"
        )
    with pytest.raises(ValueError, match="every method needs at least one scorable query"):
        paired_family_ranking_bootstrap(
            {"r5": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
            _groups(),
            reference="r5",
        )
