from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from fashion.train.task3_clean_slate import fixed_feature_vector
from fashion.train.task3_usage_hog_svm import (
    UsageHogSvmConfig,
    WeightedScaledLinearSVC,
    _calibrated_parameter_count,
    check_usage_hog_svm_setup,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_rgb_hog_is_pure_finite_descriptor() -> None:
    image = np.full((80, 60, 3), 250, dtype=np.uint8)
    image[20:65, 18:44] = np.array([20, 80, 180], dtype=np.uint8)
    pure = fixed_feature_vector(image, view="full_rgb_hog")
    composite = fixed_feature_vector(image, view="full")

    assert pure.dtype == np.float32
    assert pure.ndim == 1
    assert np.isfinite(pure).all()
    assert len(pure) < len(composite)
    assert np.array_equal(pure, composite[: len(pure)])


def test_weighted_scaled_svm_accepts_sample_weights() -> None:
    rng = np.random.default_rng(2753)
    features = rng.normal(size=(30, 8))
    labels = np.asarray(["A"] * 20 + ["B"] * 10)
    weights = np.asarray([0.5] * 20 + [2.0] * 10)
    model = WeightedScaledLinearSVC(c=1.0, max_iterations=2000, seed=2753)
    model.fit(features, labels, sample_weight=weights)

    assert model.predict(features).shape == (30,)
    assert model.decision_function(features).shape == (30,)
    assert model.classes_.tolist() == ["A", "B"]


def test_weighted_scaled_svm_calibrates_to_probabilities() -> None:
    rng = np.random.default_rng(2753)
    features = rng.normal(size=(60, 8))
    labels = np.asarray(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
    weights = np.linspace(0.5, 1.5, len(labels))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=2753).split(features, labels))
    model = CalibratedClassifierCV(
        estimator=WeightedScaledLinearSVC(max_iterations=2000),
        method="sigmoid",
        cv=folds,
        ensemble=True,
    )
    model.fit(features, labels, sample_weight=weights)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (60, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert _calibrated_parameter_count(model) > 0


def test_usage_hog_contract_and_zero_fit_preflight(prepared_project) -> None:
    config = UsageHogSvmConfig()
    check = check_usage_hog_svm_setup(root=prepared_project.root)

    assert config.c == pytest.approx(1.0)
    assert config.feature_view == "full_rgb_hog"
    assert config.class_weight_beta == pytest.approx(0.999)
    assert config.class_weight_cap == pytest.approx(5.0)
    assert check["folds"] == [0, 4]
    assert check["execution_device"] == "cpu"
    assert check["model_fits"] == 0
    assert check["optimizer_steps"] == 0


def test_usage_hog_notebook_runs_one_cpu_screen() -> None:
    path = ROOT / "notebooks/04s_task3_usage_v2_u2_full_rgb_hog_svm.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert source.count("run_usage_hog_svm_screen(") == 1
    assert source.count("prepare_usage_hog_features(") == 1
    assert source.count("check_usage_hog_svm_setup(") == 1
    assert "folds=(0, 4)" in code
    assert "one-vs-rest linear SVM" in source
    assert "`C=1`" in source
    assert "pretrained=True" not in source
    assert "google.colab" not in source
    assert "DRIVE_" not in source
    assert not any(
        output.get("output_type") == "error"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
