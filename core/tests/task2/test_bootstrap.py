from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task2.bootstrap import (
    analyse_paired_bootstrap_packs,
    build_paired_bootstrap_decision,
    load_paired_bootstrap_spec,
    plot_paired_bootstrap_intervals,
)
from fashion.task2.slices import CandidateOOFPack


def _small_spec(**changes):
    return replace(
        load_paired_bootstrap_spec(),
        expected_row_count=8,
        expected_group_count=4,
        replicates=101,
        batch_size=13,
        **changes,
    )


def _development() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(1, 9),
            "season": ["Fall", "Fall", "Spring", "Spring", "Summer", "Summer", "Winter", "Winter"],
            "product_family_group": ["g1", "g2", "g2", "g3", "g3", "g4", "g4", "g4"],
        }
    )


def _packs() -> list[CandidateOOFPack]:
    spec = load_paired_bootstrap_spec()
    true = _development().set_index("id")["season"]
    predictions = {
        ("C2", 2753): ["Fall", "Spring", "Spring", "Fall", "Summer", "Winter", "Winter", "Fall"],
        ("I2", 2753): ["Fall", "Fall", "Spring", "Spring", "Summer", "Winter", "Winter", "Winter"],
        ("C2", 2026): ["Fall", "Fall", "Summer", "Spring", "Summer", "Fall", "Winter", "Fall"],
        ("I2", 2026): ["Fall", "Spring", "Spring", "Spring", "Summer", "Summer", "Winter", "Fall"],
    }
    packs = []
    for pair in spec.pairs:
        for candidate, experiment_id in (
            ("C2", pair.c2_experiment_id),
            ("I2", pair.i2_experiment_id),
        ):
            ids = np.asarray([8, 3, 1, 7, 4, 6, 2, 5])
            predicted = np.asarray(predictions[(candidate, pair.seed)], dtype=object)
            frame = pd.DataFrame(
                {
                    "id": ids,
                    "y_true": true.loc[ids].to_numpy(),
                    "y_pred": predicted[ids - 1],
                }
            )
            packs.append(
                CandidateOOFPack(
                    candidate=candidate,
                    experiment_id=experiment_id,
                    seed=pair.seed,
                    oof=frame,
                    registry=pd.DataFrame(),
                )
            )
    return list(reversed(packs))


def test_repository_paired_bootstrap_spec_is_fully_frozen() -> None:
    spec = load_paired_bootstrap_spec()

    assert spec.expected_row_count == 32_753
    assert spec.expected_group_count == 22_885
    assert spec.replicates == 10_000
    assert spec.bootstrap_seed == 2753
    assert spec.batch_size == 64
    assert spec.confidence_level == 0.95
    assert [pair.role for pair in spec.pairs] == [
        "primary_interval",
        "stability_sensitivity",
    ]
    assert [pair.seed for pair in spec.pairs] == [2753, 2026]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"notes": "drift"}), "fields changed"),
        (
            lambda payload: payload["bootstrap"].update({"replicates_per_seed_pair": 10_000.0}),
            "positive integer",
        ),
        (
            lambda payload: payload["candidate_pairs"][1].update({"c2_experiment_id": "wrong"}),
            "pair identities",
        ),
        (
            lambda payload: payload["warnings"].update({"holdout_is_forbidden": False}),
            "warnings",
        ),
        (
            lambda payload: payload["decision_boundary"].update(
                {"confidence_language_if_interval_contains_zero": "stronger_claim"}
            ),
            "decision safety boundary",
        ),
    ],
)
def test_paired_bootstrap_spec_rejects_protocol_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = json.loads(
        (ROOT / "configs/task2/g6_paired_group_bootstrap.json").read_text(encoding="utf-8")
    )
    mutation(payload)
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_paired_bootstrap_spec(path)


def test_paired_bootstrap_aligns_oof_by_id_and_reuses_draws() -> None:
    spec = _small_spec()

    tables = analyse_paired_bootstrap_packs(_packs(), _development(), spec)

    assert len(tables.observed_metrics) == 2
    assert len(tables.draws) == 2 * spec.replicates
    assert len(tables.interval_summary) == 12
    assert tables.group_audit.loc[0, "unique_group_count"] == 4
    sampled_rows = tables.draws.pivot(
        index="replicate",
        columns="comparison_id",
        values="sampled_row_count",
    )
    assert sampled_rows.nunique(axis=1).eq(1).all()
    assert sampled_rows.iloc[:, 0].nunique() > 1
    assert tables.draws.groupby("comparison_id")["replicate"].nunique().eq(101).all()
    primary = tables.observed_metrics.set_index("comparison_id").loc["primary_interval"]
    assert primary["i2_minus_c2_macro_f1"] > 0


def test_paired_bootstrap_interval_summary_matches_declared_numpy_quantiles() -> None:
    spec = _small_spec()
    tables = analyse_paired_bootstrap_packs(_packs(), _development(), spec)
    row = tables.interval_summary.loc[
        tables.interval_summary["comparison_id"].eq("primary_interval")
        & tables.interval_summary["metric"].eq("macro_f1")
    ].iloc[0]
    values = tables.draws.loc[
        tables.draws["comparison_id"].eq("primary_interval"),
        "i2_minus_c2_macro_f1",
    ].to_numpy()
    expected = np.quantile(values, [0.025, 0.975], method="linear")

    assert row["ci_lower"] == pytest.approx(expected[0])
    assert row["ci_upper"] == pytest.approx(expected[1])
    assert row["replicates"] == 101
    assert row["interval_method"] == "percentile"


@pytest.mark.parametrize(
    "mutation",
    ["truth", "duplicate_id", "fractional_id", "missing_group"],
)
def test_paired_bootstrap_rejects_alignment_and_group_drift(mutation: str) -> None:
    spec = _small_spec()
    packs = _packs()
    development = _development()
    if mutation == "truth":
        packs[0].oof.loc[packs[0].oof.index[0], "y_true"] = "Fall"
    elif mutation == "duplicate_id":
        packs[0].oof.loc[packs[0].oof.index[0], "id"] = packs[0].oof.loc[
            packs[0].oof.index[1], "id"
        ]
    elif mutation == "fractional_id":
        packs[0].oof["id"] = packs[0].oof["id"].astype(object)
        packs[0].oof.loc[packs[0].oof.index[0], "id"] = 8.5
    else:
        development.loc[0, "product_family_group"] = ""

    with pytest.raises(ValueError):
        analyse_paired_bootstrap_packs(packs, development, spec)


def test_paired_bootstrap_decision_keeps_candidate_and_holdout_boundaries() -> None:
    spec = _small_spec()
    tables = analyse_paired_bootstrap_packs(_packs(), _development(), spec)

    decision = build_paired_bootstrap_decision(tables, spec)

    assert decision["current_candidate"] == "I2"
    assert decision["candidate_selection_affected"] is False
    assert decision["new_candidates_allowed"] is False
    assert decision["ultimate_winner_frozen"] is False
    assert decision["holdout_opened"] is False
    assert set(decision["pair_outcomes"]) == {
        "primary_interval",
        "stability_sensitivity",
    }


@pytest.mark.parametrize("mutation", ["reversed_interval", "nan", "wrong_flag"])
def test_paired_bootstrap_decision_rejects_interval_drift(mutation: str) -> None:
    spec = _small_spec()
    tables = analyse_paired_bootstrap_packs(_packs(), _development(), spec)
    mask = tables.interval_summary["comparison_id"].eq(
        "primary_interval"
    ) & tables.interval_summary["metric"].eq("macro_f1")
    if mutation == "reversed_interval":
        tables.interval_summary.loc[mask, ["ci_lower", "ci_upper"]] = [0.2, 0.1]
    elif mutation == "nan":
        tables.interval_summary.loc[mask, "observed_delta"] = np.nan
    else:
        tables.interval_summary["interval_contains_zero"] = tables.interval_summary[
            "interval_contains_zero"
        ].astype(object)
        tables.interval_summary.loc[mask, "interval_contains_zero"] = "false"

    with pytest.raises(ValueError, match="decision"):
        build_paired_bootstrap_decision(tables, spec)


def test_paired_bootstrap_plot_writes_two_panel_png(tmp_path: Path) -> None:
    tables = analyse_paired_bootstrap_packs(_packs(), _development(), _small_spec())
    output = tmp_path / "paired_bootstrap_intervals.png"

    returned = plot_paired_bootstrap_intervals(tables, output)

    assert returned == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000
