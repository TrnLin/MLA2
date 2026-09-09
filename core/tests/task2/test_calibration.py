from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.task2.calibration import (
    CalibrationCandidate,
    CalibrationSpec,
    analyse_calibration_packs,
    analyse_candidate_calibration,
    load_calibration_spec,
    plot_calibration_reliability,
    plot_risk_coverage,
    risk_coverage_curve,
    top_label_reliability_bins,
)
from fashion.task2.slices import CandidateOOFPack
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS, temperature_scale_probabilities


def _spec(*, expected_row_count: int = 40) -> CalibrationSpec:
    return CalibrationSpec(
        analysis_id="g6-cross-fitted-calibration",
        expected_row_count=expected_row_count,
        candidates=(
            CalibrationCandidate("C2", "g3-c2-t0-resnet18", 2753),
            CalibrationCandidate("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
        ),
        folds=tuple(range(5)),
        probability_floor=1e-12,
        temperature_bounds=(0.05, 10.0),
        optimizer_tolerance=1e-8,
        ece_bins=15,
        coverage_start=0.1,
        coverage_stop=1.0,
        coverage_step=0.01,
        review_budgets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )


def _pack(candidate: str, experiment_id: str, *, fewer_errors: bool) -> CandidateOOFPack:
    labels = np.tile(np.asarray(SEASON_LABELS, dtype=object), 10)
    rows = []
    for index, label in enumerate(labels):
        true_index = SEASON_LABELS.index(str(label))
        error = index % (8 if fewer_errors else 4) == 0
        predicted_index = (true_index + 1) % 4 if error else true_index
        probabilities = np.full(4, 0.01, dtype=float)
        probabilities[predicted_index] = 0.94
        probabilities[(predicted_index + 1) % 4] = 0.03
        probabilities /= probabilities.sum()
        rows.append(
            {
                "run_id": f"{experiment_id}-f{index // 8}",
                "experiment_id": experiment_id,
                "id": index + 1,
                "fold": index // 8,
                "seed": 2753,
                "y_true": str(label),
                "y_pred": SEASON_LABELS[predicted_index],
                **{
                    f"prob_{season}": probabilities[class_index]
                    for class_index, season in enumerate(SEASON_LABELS)
                },
            }
        )
    return CandidateOOFPack(
        candidate=candidate,
        experiment_id=experiment_id,
        seed=2753,
        oof=pd.DataFrame(rows),
        registry=pd.DataFrame(),
    )


def test_frozen_calibration_contract_loads_exact_protocol() -> None:
    spec = load_calibration_spec()

    assert spec.expected_row_count == 32_753
    assert [(row.candidate, row.seed) for row in spec.candidates] == [
        ("C2", 2753),
        ("I2", 2753),
    ]
    assert spec.folds == tuple(range(5))
    assert spec.temperature_bounds == (0.05, 10.0)
    assert spec.review_budgets == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def test_calibration_config_rejects_fractional_fold(tmp_path: Path) -> None:
    config_path = Path("configs/task2/g6_cross_fitted_calibration.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cross_fitting"]["folds"][1] = 1.5
    path = tmp_path / "fractional-fold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cross-fitting fold must be an integer"):
        load_calibration_spec(path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("extra_candidate_field", "candidate experiment 0 fields changed"),
        ("duplicate_metric", "calibration metric protocol changed"),
        ("string_warning", "calibration safety warnings changed"),
    ],
)
def test_calibration_config_rejects_safety_drift(tmp_path: Path, mutation: str, error: str) -> None:
    config_path = Path("configs/task2/g6_cross_fitted_calibration.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if mutation == "extra_candidate_field":
        payload["candidate_experiments"][0]["notes"] = "not declared"
    elif mutation == "duplicate_metric":
        payload["calibration_metrics"]["metrics"].append("accuracy")
    else:
        payload["warnings"]["holdout_is_forbidden"] = "false"
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_calibration_spec(path)


def test_reliability_bins_preserve_support_including_empty_bins() -> None:
    pack = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    probabilities = pack.oof.loc[:, [f"prob_{label}" for label in SEASON_LABELS]]

    bins = top_label_reliability_bins(pack.oof["y_true"], probabilities, ece_bins=15)

    assert len(bins) == 15
    assert bins["count"].sum() == 40
    assert bins["count"].eq(0).any()
    assert bins.loc[bins["count"].eq(0), "mean_confidence"].isna().all()


@pytest.mark.parametrize(
    "probabilities, error",
    [
        ([[np.nan, 0.0, 0.0, 1.0]], "finite"),
        ([[0.8, 0.1, 0.1, 0.1]], "sum to one"),
    ],
)
def test_reliability_bins_reject_invalid_probabilities(
    probabilities: list[list[float]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        top_label_reliability_bins(["Fall"], probabilities)


def test_risk_coverage_uses_ceiling_and_full_coverage_matches_error_rate() -> None:
    pack = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    probabilities = pack.oof.loc[:, [f"prob_{label}" for label in SEASON_LABELS]]

    curve = risk_coverage_curve(
        pack.oof["id"],
        pack.oof["y_true"],
        probabilities,
        coverage_start=0.1,
        coverage_stop=1.0,
        coverage_step=0.01,
    )

    assert len(curve) == 91
    assert curve.iloc[0]["retained_count"] == 4
    assert curve.iloc[-1]["retained_count"] == 40
    assert curve.iloc[-1]["selective_risk"] == pytest.approx(0.25)
    assert curve["coverage"].is_monotonic_increasing


def test_risk_coverage_rejects_fractional_ids() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        risk_coverage_curve(
            [1.5],
            ["Fall"],
            [[0.7, 0.1, 0.1, 0.1]],
        )


def test_risk_coverage_preserves_large_int64_ids_and_rejects_overflow() -> None:
    identifiers = [2**53 + 1, 2**53 + 2]
    probabilities = [[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]

    curve = risk_coverage_curve(
        identifiers,
        ["Fall", "Fall"],
        probabilities,
        coverage_start=0.5,
        coverage_stop=1.0,
        coverage_step=0.5,
    )

    assert curve.iloc[0]["accepted_id_sha256"] == canonical_sha256([identifiers[0]])
    with pytest.raises(ValueError, match="fit signed int64"):
        risk_coverage_curve(
            [2**63],
            ["Fall"],
            [[0.7, 0.1, 0.1, 0.1]],
        )


def test_risk_coverage_breaks_confidence_ties_by_ascending_id() -> None:
    probabilities = np.tile(np.asarray([[0.7, 0.1, 0.1, 0.1]]), (4, 1))

    curve = risk_coverage_curve(
        [4, 1, 3, 2],
        ["Fall"] * 4,
        probabilities,
        coverage_start=0.5,
        coverage_stop=0.5,
        coverage_step=0.01,
    )

    assert curve.iloc[0]["accepted_id_sha256"] == canonical_sha256([1, 2])


@pytest.mark.parametrize("bad_fold", [0.5, "0"])
def test_candidate_analysis_rejects_non_integer_fold_values(bad_fold: object) -> None:
    pack = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    pack.oof["fold"] = pack.oof["fold"].astype(object)
    pack.oof.loc[0, "fold"] = bad_fold

    with pytest.raises(ValueError, match="calibration OOF folds must be an integer"):
        analyse_candidate_calibration(pack, _spec())


def test_candidate_analysis_rejects_fractional_ids_before_ranking() -> None:
    pack = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    pack.oof["id"] = pack.oof["id"].astype(object)
    pack.oof.loc[0, "id"] = 1.5

    with pytest.raises(ValueError, match="calibration OOF IDs must be an integer"):
        analyse_candidate_calibration(pack, _spec())


def test_candidate_analysis_cross_fits_and_preserves_discrimination() -> None:
    pack = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)

    tables = analyse_candidate_calibration(pack, _spec())

    assert len(tables.fold_temperatures) == 5
    assert tables.fold_temperatures["fit_fold_count"].eq(4).all()
    assert tables.fold_temperatures["calibration_rows"].eq(32).all()
    assert tables.fold_temperatures["evaluation_rows"].eq(8).all()
    summary = tables.calibration_summary.set_index("calibration_method")
    assert summary.loc["cross_fitted_temperature", "nll"] < summary.loc["uncalibrated", "nll"]
    assert (
        summary.loc["cross_fitted_temperature", "accuracy"]
        == summary.loc["uncalibrated", "accuracy"]
    )
    assert (
        summary.loc["cross_fitted_temperature", "macro_f1"]
        == summary.loc["uncalibrated", "macro_f1"]
    )
    assert len(tables.reliability_bins) == 30
    assert len(tables.risk_coverage) == 182
    assert len(tables.review_budget_summary) == 12
    assert not bool(tables.deployment_temperatures.iloc[0]["evaluation_claim_allowed"])
    assert tables.calibrated_oof["id"].nunique() == 40
    assert tables.calibrated_oof["temperature"].notna().all()
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]
    for fold_row in tables.fold_temperatures.itertuples(index=False):
        mask = pack.oof["fold"].eq(fold_row.evaluation_fold)
        expected = temperature_scale_probabilities(
            pack.oof.loc[mask, probability_columns],
            fold_row.temperature,
            probability_floor=1e-12,
        )
        actual = tables.calibrated_oof.loc[mask, probability_columns].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_pair_analysis_requires_exact_primary_candidates() -> None:
    c2 = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    i2 = _pack("I2", "g4-i2-article-type-lambda-0-3-c1", fewer_errors=True)

    tables = analyse_calibration_packs([c2, i2], _spec())

    assert len(tables.calibration_summary) == 4
    assert len(tables.fold_temperatures) == 10
    assert set(tables.calibration_summary["candidate"]) == {"C2", "I2"}
    with pytest.raises(ValueError, match="candidate pair"):
        analyse_calibration_packs([c2], _spec())


def test_calibration_figures_render_both_candidates(tmp_path: Path) -> None:
    c2 = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    i2 = _pack("I2", "g4-i2-article-type-lambda-0-3-c1", fewer_errors=True)
    tables = analyse_calibration_packs([c2, i2], _spec())
    reliability_path = tmp_path / "reliability.png"
    risk_path = tmp_path / "risk.png"

    assert plot_calibration_reliability(tables.reliability_bins, reliability_path) == (
        reliability_path
    )
    assert plot_risk_coverage(tables.risk_coverage, risk_path) == risk_path
    assert reliability_path.stat().st_size > 10_000
    assert risk_path.stat().st_size > 10_000


def test_calibration_figures_reject_incomplete_panels(tmp_path: Path) -> None:
    c2 = _pack("C2", "g3-c2-t0-resnet18", fewer_errors=False)
    i2 = _pack("I2", "g4-i2-article-type-lambda-0-3-c1", fewer_errors=True)
    tables = analyse_calibration_packs([c2, i2], _spec())

    with pytest.raises(ValueError, match="complete C2/I2"):
        plot_calibration_reliability(
            tables.reliability_bins.loc[tables.reliability_bins["candidate"].eq("C2")],
            tmp_path / "missing-i2-reliability.png",
        )
    with pytest.raises(ValueError, match="both calibration methods"):
        plot_risk_coverage(
            tables.risk_coverage.loc[tables.risk_coverage["calibration_method"].eq("uncalibrated")],
            tmp_path / "missing-method-risk.png",
        )
