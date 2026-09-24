"""Boundaries for the separate post-submission full-development RF study."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT, SPLITS_CSV
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.post_submission_rf_evaluation import (
    DEFAULT_EVIDENCE_DIR,
    EVALUATION_MANIFEST,
    PREDICTION_RECEIPT,
    _verify_rf_prediction_receipt,
    load_verified_rf_holdout_evaluation,
)
from fashion.task2.post_submission_rf_refit import (
    DEFAULT_FOREST_PATH,
    DEFAULT_MANIFEST_PATH,
    _record,
    _verify_record,
    load_verified_rf_fit,
)
from fashion.train.artifacts import ArtifactVerificationError, canonical_sha256
from fashion.train.metrics import SEASON_LABELS


def _require_local_forest() -> None:
    if not DEFAULT_FOREST_PATH.is_file() or not DEFAULT_MANIFEST_PATH.is_file():
        pytest.skip("local ignored post-submission RF weight is not installed")


def test_hashed_artifact_rejects_changed_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "forest.joblib"
    artifact.write_bytes(b"original")
    declaration = _record(artifact, tmp_path)
    assert _verify_record(declaration, tmp_path, artifact) == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ArtifactVerificationError, match="SHA-256"):
        _verify_record(declaration, tmp_path, artifact)


def test_full_refit_uses_only_canonical_development_rows() -> None:
    _require_local_forest()
    bundle = load_verified_rf_fit(device="cpu")
    splits = load_splits(SPLITS_CSV)
    development = get_samples(splits, partition="development", target="season")
    protected_ids = set(splits.loc[splits["partition"].ne("development"), "id"])
    training_ids = set(development["id"])
    assert len(development) == len(training_ids) == 32_753
    assert training_ids.isdisjoint(protected_ids)
    assert bundle.manifest["training_id_sha256"] == canonical_sha256(
        sorted(int(identifier) for identifier in training_ids)
    )
    assert bundle.manifest["source_encoder"]["run_id"] == bundle.encoder.run_id
    assert bundle.manifest["source_encoder"]["bundle"]["sha256"] == compute_sha256(
        bundle.encoder.bundle_path
    )
    assert bundle.manifest["role"] == "post_submission_full_development_refit"
    assert bundle.forest.n_features_in_ == 256
    assert np.array_equal(bundle.forest.classes_, np.arange(len(SEASON_LABELS)))


def test_holdout_prediction_receipt_precedes_retrospective_comparison() -> None:
    _require_local_forest()
    if not EVALUATION_MANIFEST.is_file():
        pytest.skip("post-submission holdout evaluation has not run locally")
    manifest = load_verified_rf_holdout_evaluation()
    receipt = json.loads(PREDICTION_RECEIPT.read_text(encoding="utf-8"))
    assert manifest["holdout_previously_opened"] is True
    assert manifest["used_for_training_or_tuning"] is False
    assert receipt["labels_used_in_prediction"] is False
    assert manifest["rows"] == 5_778
    assert manifest["prediction_receipt"]["sha256"] == compute_sha256(
        PREDICTION_RECEIPT
    )
    predictions = pd.read_csv(DEFAULT_EVIDENCE_DIR / "holdout_rf_predictions.csv")
    assert predictions["id"].nunique() == len(predictions) == 5_778
    assert set(predictions["y_pred"]).issubset(SEASON_LABELS)
    assert "y_true" not in predictions and "season" not in predictions
    probabilities = predictions[[f"prob_{label}" for label in SEASON_LABELS]].to_numpy()
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-12)


def test_prediction_receipt_rejects_changed_or_missing_evidence() -> None:
    _require_local_forest()
    if not PREDICTION_RECEIPT.is_file():
        pytest.skip("post-submission RF predictions have not run locally")
    bundle = load_verified_rf_fit(device="cpu")
    receipt = json.loads(PREDICTION_RECEIPT.read_text(encoding="utf-8"))
    mutated = deepcopy(receipt)
    mutated["labels_used_in_prediction"] = True
    with pytest.raises(ValueError, match="receipt contract"):
        _verify_rf_prediction_receipt(
            mutated, bundle=bundle, root=ROOT, evidence=DEFAULT_EVIDENCE_DIR
        )
    mutated = deepcopy(receipt)
    del mutated["artifacts"]["clean"]
    with pytest.raises(ValueError, match="receipt contract"):
        _verify_rf_prediction_receipt(
            mutated, bundle=bundle, root=ROOT, evidence=DEFAULT_EVIDENCE_DIR
        )
    mutated = deepcopy(receipt)
    mutated["image_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactVerificationError, match="SHA-256"):
        _verify_rf_prediction_receipt(
            mutated, bundle=bundle, root=ROOT, evidence=DEFAULT_EVIDENCE_DIR
        )


def test_comparison_has_paired_uncertainty_and_five_fixed_conditions() -> None:
    _require_local_forest()
    if not EVALUATION_MANIFEST.is_file():
        pytest.skip("post-submission holdout evaluation has not run locally")
    scorecard = pd.read_csv(DEFAULT_EVIDENCE_DIR / "scorecard.csv")
    intervals = pd.read_csv(DEFAULT_EVIDENCE_DIR / "paired_bootstrap_intervals.csv")
    robustness = pd.read_csv(DEFAULT_EVIDENCE_DIR / "robustness.csv")
    assert set(scorecard["model"]) == {"I2 frozen", "I2 embedding + RF"}
    assert set(scorecard["n_samples"]) == {5_778}
    assert set(intervals["metric"]) == {"macro_f1", "accuracy", "spring_f1"}
    assert set(intervals["replicates"]) == {10_000}
    assert robustness["condition"].nunique() == 5
    assert len(robustness) == 10
    assert (robustness["rows"] == 5_778).all()
    assert ROOT in DEFAULT_EVIDENCE_DIR.parents
