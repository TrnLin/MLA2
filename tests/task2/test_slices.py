from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.data.dataset import get_cv_split, get_samples, load_splits
from fashion.task2.slices import (
    CandidateExperiment,
    CandidateOOFPack,
    SliceAnalysisSpec,
    SliceAssignmentBundle,
    analyse_slice_packs,
    assign_file_size_quartiles,
    build_slice_assignments,
    fit_file_size_boundaries,
    load_slice_analysis_spec,
    plot_slice_macro_f1,
    plot_spring_destinations,
)
from fashion.train.metrics import SEASON_LABELS


def _analysis_spec(*, expected_row_count: int = 12) -> SliceAnalysisSpec:
    return SliceAnalysisSpec(
        analysis_id="g6-shortcut-error-slices",
        expected_row_count=expected_row_count,
        candidates=(
            CandidateExperiment("C2", "g3-c2-t0-resnet18", 2753),
            CandidateExperiment("C2", "g5-c2-t0-resnet18-s2026", 2026),
            CandidateExperiment("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
            CandidateExperiment(
                "I2",
                "g5-i2-article-type-lambda-0-3-c1-s2026",
                2026,
            ),
        ),
        high_confidence_threshold=0.8,
        maximum_ranked_confusions=8,
        low_support_threshold=3,
        quantiles=(0.25, 0.5, 0.75),
    )


def _assignment_bundle() -> SliceAssignmentBundle:
    labels = list(SEASON_LABELS) * 3
    assignments = pd.DataFrame(
        {
            "id": range(1, 13),
            "fold": [value % 5 for value in range(12)],
            "season": labels,
            "article_type_shortcut": [
                "aligned",
                "conflict",
                "aligned",
                "conflict",
            ]
            * 3,
            "acquisition_year": [
                "dominant_2011_2012",
                "other_years",
                "dominant_2011_2012",
            ]
            * 4,
            "file_size_quartile": [
                "q1_smallest",
                "q2",
                "q3",
                "q4_largest",
            ]
            * 3,
            "product_family_size": ["singleton", "multirow"] * 6,
            "image_mode": ["rgb", "rgb", "greyscale"] * 4,
        }
    )
    return SliceAssignmentBundle(
        assignments=assignments,
        file_size_boundaries=pd.DataFrame(),
        article_type_mappings=pd.DataFrame(),
        article_type_fold_audit=pd.DataFrame(),
        slice_support=pd.DataFrame(),
        assignment_sha256="a" * 64,
    )


def _oof_pack(
    candidate: str,
    experiment_id: str,
    seed: int,
    *,
    error_ids: set[int],
) -> CandidateOOFPack:
    assignments = _assignment_bundle().assignments
    rows = []
    for row in assignments.itertuples(index=False):
        true_index = SEASON_LABELS.index(row.season)
        predicted_index = (
            (true_index + 1) % len(SEASON_LABELS) if row.id in error_ids else true_index
        )
        probabilities = np.full(len(SEASON_LABELS), 0.1, dtype=float)
        probabilities[predicted_index] = 0.7
        rows.append(
            {
                "run_id": f"{experiment_id}-f{row.fold}",
                "experiment_id": experiment_id,
                "id": row.id,
                "fold": row.fold,
                "seed": seed,
                "y_true": row.season,
                "y_pred": SEASON_LABELS[predicted_index],
                **{
                    f"prob_{label}": probabilities[index]
                    for index, label in enumerate(SEASON_LABELS)
                },
            }
        )
    return CandidateOOFPack(
        candidate=candidate,
        experiment_id=experiment_id,
        seed=seed,
        oof=pd.DataFrame(rows),
        registry=pd.DataFrame(),
    )


def _packs() -> list[CandidateOOFPack]:
    return [
        _oof_pack("C2", "g3-c2-t0-resnet18", 2753, error_ids={2, 4, 6, 8}),
        _oof_pack(
            "C2",
            "g5-c2-t0-resnet18-s2026",
            2026,
            error_ids={2, 4, 6, 10},
        ),
        _oof_pack(
            "I2",
            "g4-i2-article-type-lambda-0-3-c1",
            2753,
            error_ids={2, 6},
        ),
        _oof_pack(
            "I2",
            "g5-i2-article-type-lambda-0-3-c1-s2026",
            2026,
            error_ids={4, 10},
        ),
    ]


def test_frozen_slice_contract_loads_exact_candidate_order() -> None:
    spec = load_slice_analysis_spec()

    assert spec.expected_row_count == 32_753
    assert [row.candidate for row in spec.candidates] == ["C2", "C2", "I2", "I2"]
    assert [row.seed for row in spec.candidates] == [2753, 2026, 2753, 2026]
    assert spec.quantiles == (0.25, 0.5, 0.75)


def test_file_size_quartiles_use_declared_training_boundaries() -> None:
    training = pd.DataFrame({"file_size_bytes": [10, 20, 30, 40, 50, 60, 70, 80]})

    boundaries = fit_file_size_boundaries(training)
    assigned = assign_file_size_quartiles(
        pd.Series([9, boundaries[0], boundaries[1], boundaries[2], 90]),
        boundaries,
    )

    assert boundaries == pytest.approx((27.5, 45.0, 62.5))
    assert assigned.tolist() == [
        "q1_smallest",
        "q2",
        "q3",
        "q4_largest",
        "q4_largest",
    ]


def test_repository_slice_assignments_cover_only_valid_development_rows() -> None:
    splits = load_splits()
    spec = load_slice_analysis_spec()

    bundle = build_slice_assignments(splits, spec)
    expected = get_samples(splits, partition="development", target="season")
    protected_ids = set(splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"])

    assert len(bundle.assignments) == bundle.assignments["id"].nunique() == len(expected)
    assert len(bundle.assignments) == 32_753
    assert not (set(bundle.assignments["id"]) & protected_ids)
    assert set(bundle.assignments["fold"]) == set(range(5))
    assert bundle.assignment_sha256.isalnum() and len(bundle.assignment_sha256) == 64
    pooled = bundle.slice_support.loc[bundle.slice_support["fold"].eq(-1)]
    assert pooled.groupby("slice_family")["support"].sum().eq(32_753).all()


def test_repository_file_size_boundaries_are_fit_without_validation_fold() -> None:
    splits = load_splits()
    bundle = build_slice_assignments(splits, load_slice_analysis_spec())

    for row in bundle.file_size_boundaries.itertuples(index=False):
        training, _ = get_cv_split(splits, int(row.fold))
        expected = fit_file_size_boundaries(get_samples(training, target="season"))
        assert (row.q25_bytes, row.q50_bytes, row.q75_bytes) == pytest.approx(expected)


def test_slice_analysis_compares_same_support_and_keeps_real_error_ids() -> None:
    tables = analyse_slice_packs(_packs(), _assignment_bundle(), _analysis_spec())

    deltas = tables.candidate_slice_deltas
    assert deltas["c2_support"].eq(deltas["i2_support"]).all()
    assert len(tables.spring_metrics) == 4
    destination_totals = tables.spring_destinations.groupby(["candidate", "seed"])[
        "proportion_of_true_spring"
    ].sum()
    assert np.allclose(destination_totals, 1.0, rtol=0.0, atol=1e-12)
    assert set(tables.error_examples["id"]) <= set(range(1, 13))
    assert tables.error_examples.groupby(["candidate", "seed", "confusion_rank"]).size().eq(1).all()
    primary = deltas.loc[
        deltas["seed"].eq(2753)
        & deltas["slice_family"].eq("article_type_shortcut")
        & deltas["slice_name"].eq("conflict")
    ].iloc[0]
    assert primary["i2_minus_c2_accuracy"] > 0


def test_slice_analysis_rejects_oof_fold_drift() -> None:
    packs = _packs()
    changed = packs[0].oof.copy()
    changed.loc[0, "fold"] = 4
    packs[0] = CandidateOOFPack(
        candidate=packs[0].candidate,
        experiment_id=packs[0].experiment_id,
        seed=packs[0].seed,
        oof=changed,
        registry=packs[0].registry,
    )

    with pytest.raises(ValueError, match="fold differs"):
        analyse_slice_packs(packs, _assignment_bundle(), _analysis_spec())


def test_slice_figures_render_all_declared_candidates(tmp_path: Path) -> None:
    tables = analyse_slice_packs(_packs(), _assignment_bundle(), _analysis_spec())
    slice_figure = tmp_path / "slice-macro-f1.png"
    spring_figure = tmp_path / "spring-destinations.png"

    assert plot_slice_macro_f1(tables.slice_metrics, slice_figure) == slice_figure
    assert plot_spring_destinations(tables.spring_destinations, spring_figure) == spring_figure
    assert slice_figure.stat().st_size > 10_000
    assert spring_figure.stat().st_size > 10_000


def test_slice_figures_do_not_require_an_interactive_pyplot_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = analyse_slice_packs(_packs(), _assignment_bundle(), _analysis_spec())

    def reject_interactive_manager(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("slice figures must use the headless Agg canvas")

    monkeypatch.setattr("matplotlib.pyplot.subplots", reject_interactive_manager)

    assert plot_slice_macro_f1(
        tables.slice_metrics,
        tmp_path / "headless-slice-macro-f1.png",
    ).is_file()
    assert plot_spring_destinations(
        tables.spring_destinations,
        tmp_path / "headless-spring-destinations.png",
    ).is_file()
