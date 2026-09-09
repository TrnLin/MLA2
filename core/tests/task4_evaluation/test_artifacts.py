from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fashion.task4_evaluation.score as score_module
from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.task4_evaluation import load_verified_holdout_evaluation
from fashion.task4_evaluation.encoders import (
    R5_METHOD,
    RANDOM_FLOOR_METHOD,
)
from fashion.task4_evaluation.spec import load_holdout_evaluation_spec

EVIDENCE = ROOT / "results/evidence/task4/final_evaluation"
FIGURES = ROOT / "results/figures/task4/final_evaluation"
APPROVAL_REF = "refs/tags/task4-holdout-scoring-approved-v1"
LOCAL_PRIMARY_RANKINGS = EVIDENCE / "holdout_primary_rankings.csv"
LOCAL_PRIMARY_RANKINGS_RECORD = {
    "bytes": 292330199,
    "path": "results/evidence/task4/final_evaluation/holdout_primary_rankings.csv",
    "rows": 3813480,
    "sha256": "ed8e5864417b25450627b426ce3017aa6f1b947db147ca845cb7aadcd12e99f2",
}
REQUIRES_LOCAL_PRIMARY_RANKINGS = pytest.mark.skipif(
    not LOCAL_PRIMARY_RANKINGS.is_file(),
    reason="requires local 292 MB holdout_primary_rankings.csv",
)
BLIND_ARTIFACT_NAMES = {
    "gallery_manifest",
    "holdout_family_rankings",
    "holdout_primary_rankings",
    "holdout_query_manifest",
    "runtime",
}
PROTECTED_OR_SCORED_COLUMNS = {
    "articleType",
    "season",
    "gender",
    "usage",
    "y_true",
    "relevance",
    "grade",
    "ndcg_at_10",
    "recall_at_10",
}
FAILURE_SLICES = {
    "grayscale",
    "rare_article_type",
    "rare_type_colour",
    "unusual_geometry",
    "family_unavailable",
    "weak_family",
}
FIGURE_NAMES = {
    "holdout_bootstrap_intervals.png",
    "holdout_error_examples.png",
    "holdout_scorecard.png",
    "holdout_selective_retrieval.png",
    "holdout_slices_robustness.png",
    "holdout_source_robustness.png",
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _prediction_receipt() -> dict[str, object]:
    return _read_json(EVIDENCE / "prediction_receipt.json")


def _timestamp(value: object) -> datetime:
    assert isinstance(value, str)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_blind_receipt_predates_label_access_and_binds_frozen_r5() -> None:
    receipt = _prediction_receipt()
    attempt = _read_json(EVIDENCE / "unlock_attempt.json")
    unlock = _read_json(EVIDENCE / "unlock_receipt.json")
    spec = load_holdout_evaluation_spec()

    assert receipt["phase"] == "blind_prediction_before_holdout_label_access"
    assert receipt["labels_opened"] is False
    assert receipt["teacher_test_scored"] is False
    assert receipt["model_changed"] is False
    assert receipt["retuning_allowed"] is False
    assert _timestamp(receipt["created_at_utc"]) < _timestamp(attempt["created_at_utc"])
    assert _timestamp(receipt["created_at_utc"]) < _timestamp(unlock["opened_at_utc"])
    assert receipt["git"]["commit"] == unlock["blind_source_commit"]
    assert receipt["model"] == {
        "checkpoint_sha256": spec.model_checkpoint_sha256,
        "development_winner_score": spec.development_winner_score,
        "run_id": spec.model_run_id,
        "scratch": True,
    }


def test_local_primary_rankings_receipt_preserves_identity_contract() -> None:
    record = _prediction_receipt()["artifacts"]["holdout_primary_rankings"]
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")

    assert record == LOCAL_PRIMARY_RANKINGS_RECORD
    assert "3,813,480" in readme
    assert "292,330,199" in readme
    assert LOCAL_PRIMARY_RANKINGS_RECORD["sha256"] in readme


@REQUIRES_LOCAL_PRIMARY_RANKINGS
def test_blind_receipt_hashes_and_counts_every_output() -> None:
    artifacts = _prediction_receipt()["artifacts"]

    assert set(artifacts) == BLIND_ARTIFACT_NAMES
    for name, record in artifacts.items():
        path = ROOT / record["path"]
        assert path.is_file(), name
        assert path.stat().st_size == record["bytes"], name
        assert compute_sha256(path) == record["sha256"], name


@REQUIRES_LOCAL_PRIMARY_RANKINGS
def test_blind_artifacts_have_no_protected_or_scored_labels() -> None:
    artifacts = _prediction_receipt()["artifacts"]

    for name, record in artifacts.items():
        path = ROOT / record["path"]
        if path.suffix != ".csv":
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns.intersection(PROTECTED_OR_SCORED_COLUMNS), name

    queries = pd.read_csv(
        EVIDENCE / "holdout_query_manifest.csv",
        usecols=["articleType_redacted"],
        keep_default_na=False,
    )
    assert queries["articleType_redacted"].astype(str).str.strip().eq("").all()


def test_unlock_records_approved_commit_and_forbids_retuning() -> None:
    attempt = _read_json(EVIDENCE / "unlock_attempt.json")
    unlock = _read_json(EVIDENCE / "unlock_receipt.json")
    approved_commit = subprocess.run(
        ["git", "rev-parse", f"{APPROVAL_REF}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert attempt["state"] == "label_access_started"
    assert attempt["scoring_approval_ref"] == APPROVAL_REF
    assert attempt["scoring_approval_commit"] == approved_commit
    assert attempt["git_commit"] == approved_commit
    assert unlock["scoring_approval_ref"] == APPROVAL_REF
    assert unlock["scoring_approval_commit"] == approved_commit
    assert unlock["holdout_opened"] is True
    assert unlock["retuning_allowed"] is False
    for flag in (
        "conditions_changed",
        "metric_changed",
        "model_retrained",
        "winner_changed",
    ):
        assert unlock[flag] is False


@REQUIRES_LOCAL_PRIMARY_RANKINGS
def test_complete_manifest_hash_verifies_all_twenty_scored_artifacts() -> None:
    manifest = load_verified_holdout_evaluation()
    artifacts = manifest["artifacts"]
    figure_records = {
        name: record
        for name, record in artifacts.items()
        if name.startswith("figure_")
    }

    assert manifest["status"] == "complete"
    assert manifest["coverage"]["scored_artifacts"] == 20
    assert len(artifacts) == 20
    assert len(figure_records) == 6
    assert len(artifacts) - len(figure_records) == 14


def test_scorecard_and_bootstrap_support_honest_judgement() -> None:
    scorecard = pd.read_csv(EVIDENCE / "holdout_scorecard.csv").set_index(
        ["method", "direction"]
    )
    intervals = pd.read_csv(EVIDENCE / "holdout_bootstrap_intervals.csv").set_index(
        "metric"
    )
    r5_score = scorecard.at[(R5_METHOD, "teacher"), "ndcg_at_10_query_mean"]
    random_score = scorecard.at[
        (RANDOM_FLOOR_METHOD, "teacher"), "ndcg_at_10_query_mean"
    ]
    random_interval = intervals.loc["r5_minus_random_floor_ndcg_at_10"]
    hog_interval = intervals.loc["r5_minus_hog_hsv_edge_fusion_ndcg_at_10"]

    assert r5_score == pytest.approx(0.5162, abs=5e-5)
    assert random_score == pytest.approx(0.0343, abs=5e-5)
    assert r5_score > random_score
    assert random_interval["lower_95"] > 0
    assert random_interval["lower_95"] == pytest.approx(0.4733, abs=5e-5)
    assert random_interval["upper_95"] == pytest.approx(0.4903, abs=5e-5)
    assert hog_interval["lower_95"] < 0 < hog_interval["upper_95"]


def test_saved_per_query_scores_reproduce_family_bootstrap() -> None:
    spec = load_holdout_evaluation_spec()
    per_query = pd.read_csv(EVIDENCE / "holdout_per_query_primary.csv")
    holdout = pd.read_csv(
        EVIDENCE / "holdout_query_manifest.csv",
        usecols=["id", "direction", "product_family_group"],
    )
    holdout = holdout.loc[
        holdout["direction"].eq("teacher"), ["id", "product_family_group"]
    ]
    recomputed = score_module._bootstrap_intervals(per_query, holdout, spec).sort_values(
        "metric"
    )
    recorded = pd.read_csv(EVIDENCE / "holdout_bootstrap_intervals.csv").sort_values(
        "metric"
    )

    assert recomputed["metric"].tolist() == recorded["metric"].tolist()
    assert recomputed[
        ["replicates", "random_seed", "sampled_group_count"]
    ].reset_index(drop=True).equals(
        recorded[
            ["replicates", "random_seed", "sampled_group_count"]
        ].reset_index(drop=True)
    )
    np.testing.assert_allclose(
        recomputed[["lower_95", "median", "upper_95"]].to_numpy(),
        recorded[["lower_95", "median", "upper_95"]].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_declared_slices_directions_conditions_and_sources_appear() -> None:
    spec = load_holdout_evaluation_spec()
    slices = pd.read_csv(EVIDENCE / "holdout_slice_metrics.csv")
    per_query = pd.read_csv(
        EVIDENCE / "holdout_per_query_primary.csv",
        usecols=["method", "direction", "condition"],
    )
    robustness = pd.read_csv(EVIDENCE / "holdout_robustness_metrics.csv")
    source_robustness = pd.read_csv(EVIDENCE / "holdout_source_robustness.csv")
    expected_groups = {
        (method, direction, condition)
        for method in spec.methods
        for direction in spec.directions
        for condition in spec.conditions
        if (
            (method != RANDOM_FLOOR_METHOD and direction == spec.stressed_direction)
            or (
                method != RANDOM_FLOOR_METHOD
                and direction != spec.stressed_direction
                and condition == "clean"
            )
            or (
                method == RANDOM_FLOOR_METHOD
                and direction == spec.stressed_direction
                and condition == "clean"
            )
        )
    }
    expected_robustness = {
        (method, condition)
        for method in spec.methods
        for condition in spec.conditions
        if method != RANDOM_FLOOR_METHOD or condition == "clean"
    }

    assert set(slices["slice"]) == FAILURE_SLICES
    assert set(per_query.itertuples(index=False, name=None)) == expected_groups
    assert set(
        robustness[["method", "condition"]].itertuples(index=False, name=None)
    ) == expected_robustness
    assert set(source_robustness["method"]) == set(spec.methods)
    assert len(source_robustness) == len(spec.methods)


def test_all_six_figures_are_nonempty_and_manifest_hash_tracked() -> None:
    manifest = _read_json(EVIDENCE / "evaluation_manifest.json")
    figure_records = {
        name: record
        for name, record in manifest["artifacts"].items()
        if name.startswith("figure_")
    }
    tracked_names = {Path(record["path"]).name for record in figure_records.values()}

    assert tracked_names == FIGURE_NAMES
    for name, record in figure_records.items():
        path = ROOT / record["path"]
        assert path.parent == FIGURES
        assert path.stat().st_size == record["bytes"] > 10_000, name
        assert compute_sha256(path) == record["sha256"], name
