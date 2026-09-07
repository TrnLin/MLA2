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
    path = ROOT / "notebooks/task3_training/usage_v2_u2_full_rgb_hog_svm.ipynb"
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


def test_inner_weights_do_not_reach_sigmoid_calibration(monkeypatch):
    import sklearn.calibration as calibration

    from fashion.train.task3_experiments import effective_number_class_weights

    rng = np.random.default_rng(123)
    labels = np.array(["A"] * 36 + ["B"] * 18 + ["C"] * 9)
    features = rng.normal(size=(len(labels), 5))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=12).split(features, labels))
    seen_weights = []
    original = calibration._SigmoidCalibration.fit

    def record(self, X, y, sample_weight=None):
        seen_weights.append(sample_weight)
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(calibration._SigmoidCalibration, "fit", record)
    model = CalibratedClassifierCV(
        estimator=WeightedScaledLinearSVC(required_classes=("A", "B", "C")),
        cv=folds,
        ensemble=True,
    ).fit(features, labels)
    assert len(seen_weights) == 9
    assert all(weight is None for weight in seen_weights)
    for fitted, (training, _) in zip(model.calibrated_classifiers_, folds, strict=True):
        base = fitted.estimator
        counts = np.array([(labels[training] == c).sum() for c in base.weight_classes_])
        weights = effective_number_class_weights(counts, beta=0.999, cap=5.0)
        np.testing.assert_array_equal(base.training_counts_, counts)
        np.testing.assert_allclose(base.class_weights_, weights)
        lookup = dict(zip(base.weight_classes_, weights, strict=True))
        np.testing.assert_allclose(
            base.scaler_.mean_,
            np.average(features[training], axis=0, weights=[lookup[c] for c in labels[training]]),
        )


def test_inner_support_excludes_home_and_rejects_missing_calibration_class():
    import pandas as pd

    from fashion.train.task3_usage_hog_decision import SUPPORTED_USAGE_CLASSES
    from fashion.train.task3_usage_hog_svm import _inner_training_scope

    rows = pd.DataFrame(
        [
            {"usage": label, "cv_fold": fold, "product_family_group": f"{fold}-{label}"}
            for fold in (1, 2, 3, 4)
            for label in (*SUPPORTED_USAGE_CLASSES, "Home")
        ]
    )
    fitting, inner = _inner_training_scope(rows, outer_fold=0)
    assert "Home" not in set(fitting.usage)
    assert len(inner) == 4
    missing = rows.loc[~(rows.cv_fold.eq(1) & rows.usage.eq("Party"))]
    with pytest.raises(ValueError, match="calibration lacks"):
        _inner_training_scope(missing, outer_fold=0)


def test_solver_warning_and_iterations_are_retained():
    from sklearn.exceptions import ConvergenceWarning

    rng = np.random.default_rng(99)
    features = rng.normal(size=(150, 40))
    labels = np.array(["A", "B", "C"] * 50)
    model = WeightedScaledLinearSVC(max_iterations=1)
    with pytest.warns(ConvergenceWarning):
        model.fit(features, labels)
    assert model.solver_iterations_ == 1
    assert model.solver_converged_ is False
    assert any(w["category"] == "ConvergenceWarning" for w in model.solver_warnings_)


def test_supervisor_failure_leaves_a_failed_registry_row(prepared_project, monkeypatch):
    import pandas as pd

    import fashion.train.task3_usage_hog_svm as module
    from fashion.train.resource_worker import WorkerResourceError
    from fashion.train.task3_usage_hog_decision import SUPPORTED_USAGE_CLASSES

    # Synthetic scope isolates registration/termination from costly image work.
    rows = pd.DataFrame(
        [
            {
                "id": fold * 100 + i,
                "cv_fold": fold,
                "usage": label,
                "product_family_group": f"{fold}-{i}",
            }
            for fold in range(5)
            for i, label in enumerate((*SUPPORTED_USAGE_CLASSES, "Home"))
        ]
    )
    monkeypatch.setattr(
        module,
        "get_cv_split",
        lambda splits, fold: (rows[rows.cv_fold.ne(fold)], rows[rows.cv_fold.eq(fold)]),
    )
    monkeypatch.setattr(module, "_valid", lambda frame, target: frame)
    monkeypatch.setattr(
        module, "_classes", lambda maps, target: sorted([*SUPPORTED_USAGE_CLASSES, "Home"])
    )

    def timeout(function, kwargs, **limits):
        assert kwargs["run_id"]
        saved = pd.read_csv(prepared_project.root / "runs.csv")
        assert saved.iloc[0]["status"] == "running"
        raise WorkerResourceError("forced wall-time limit")

    monkeypatch.setattr(module, "run_supervised", timeout)
    contract = prepared_project.root / "contract.json"
    contract.write_text("{}")
    with pytest.raises(WorkerResourceError, match="forced wall-time"):
        module._run_fold(
            0,
            splits=rows,
            label_maps={},
            cache={"contract_path": str(contract)},
            audit_hash="test",
            parent_run_ids=[f"parent-{f}" for f in range(5)],
            output_root=prepared_project.root / "outputs",
            registry_path=prepared_project.root / "runs.csv",
            registry_mirrors=(),
            root=prepared_project.root,
            reuse_completed=False,
        )
    saved = pd.read_csv(prepared_project.root / "runs.csv")
    assert len(saved) == 1
    assert saved.iloc[0]["status"] == "failed"
    assert saved.iloc[0]["exception_type"] == "WorkerResourceError"
    assert saved.iloc[0]["last_completed_stage"] == "supervised_fold_incomplete"


def test_supervised_tiny_fold_writes_verified_artifacts_and_reuses(prepared_project, monkeypatch):
    import pandas as pd
    from PIL import Image

    import fashion.train.task3_usage_hog_svm as module
    from fashion.train.task3_usage_hog_decision import SUPPORTED_USAGE_CLASSES

    root = prepared_project.root
    classes = sorted([*SUPPORTED_USAGE_CLASSES, "Home"])
    rng = np.random.default_rng(18)
    records, features = [], []
    for fold in range(5):
        for i, label in enumerate(classes):
            item_id = 1000 + fold * 100 + i
            relative = f"data/raw/teacher/train/images_train/{item_id}.jpg"
            pixels = rng.integers(0, 256, size=(80, 60, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / relative)
            with Image.open(root / relative) as image:
                features.append(fixed_feature_vector(image, view="full_rgb_hog"))
            records.append(
                {
                    "id": item_id,
                    "cv_fold": fold,
                    "usage": label,
                    "path": relative,
                    "product_family_group": f"{fold}-{i}",
                    "partition": "development",
                }
            )
    rows = pd.DataFrame(records)
    monkeypatch.setattr(
        module,
        "get_cv_split",
        lambda splits, fold: (rows[rows.cv_fold.ne(fold)], rows[rows.cv_fold.eq(fold)]),
    )
    monkeypatch.setattr(module, "_valid", lambda frame, target: frame)
    monkeypatch.setattr(module, "_classes", lambda maps, target: classes)
    np.save(root / "features.npy", np.stack(features))
    rows[["id"]].to_csv(root / "ids.csv", index=False)
    (root / "contract.json").write_text("{}")
    args = dict(
        splits=rows,
        label_maps={},
        audit_hash="synthetic-only",
        cache={
            "matrix_path": str(root / "features.npy"),
            "ids_path": str(root / "ids.csv"),
            "contract_path": str(root / "contract.json"),
        },
        parent_run_ids=[f"parent-{f}" for f in range(5)],
        output_root=root / "outputs",
        registry_path=root / "runs.csv",
        registry_mirrors=(),
        root=root,
    )
    result = module._run_fold(0, **args, reuse_completed=False)
    metrics = result["metrics"]
    assert metrics["fold_wall_seconds"] >= metrics["train_seconds"]
    assert metrics["hard_address_space_limit_bytes"] == 7 * 1024**3
    history = pd.read_csv(Path(result["run_dir"]) / "solver_history.csv")
    assert len(history) == 32
    assert (history.solver_iterations > 0).all()
    assert all(isinstance(json.loads(v), list) for v in history.solver_warnings)
    assert len(list((Path(result["run_dir"]) / "corruptions").glob("*.csv"))) == 5
    reused = module._run_fold(0, **args, reuse_completed=True)
    assert reused["run_id"] == result["run_id"]
    registry = pd.read_csv(root / "runs.csv")
    assert len(registry) == 1
    assert registry.iloc[0]["status"] == "complete"
