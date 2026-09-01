import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

from fashion.config import SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.splits import validate_splits
from fashion.task1.classical import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    Task1ClassicalRunConfig,
    Task1HogSpec,
    Task1KNNConfig,
    Task1LinearSVMConfig,
    _stable_softmax,
    extract_task1_hog,
    fit_predict_task1_knn,
    fit_predict_task1_linear_svm,
    knn_probabilities_from_neighbors,
    load_or_build_task1_hog_features,
    query_task1_neighbors,
    run_task1_classical_fold,
)
from fashion.train.registry import RunRegistry


def test_hog_specs_have_fixed_ids_and_feature_lengths() -> None:
    assert TASK1_HOG_COARSE.hog_id == "task1_gray_hog_ppc16_v1"
    assert TASK1_HOG_COARSE.expected_features == 288
    assert TASK1_HOG_COARSE.pixels_per_cell == (16, 16)
    assert TASK1_HOG_FINE.hog_id == "task1_gray_hog_ppc10_v1"
    assert TASK1_HOG_FINE.expected_features == 1260
    assert TASK1_HOG_FINE.pixels_per_cell == (10, 10)
    for spec in (TASK1_HOG_COARSE, TASK1_HOG_FINE):
        assert spec.orientations == 9
        assert spec.cells_per_block == (2, 2)
        assert spec.block_norm == "L2-Hys"
        assert spec.transform_sqrt is True
        assert spec.image_size == (80, 60)


@pytest.mark.parametrize("spec", [TASK1_HOG_COARSE, TASK1_HOG_FINE])
def test_extract_task1_hog_is_float32_deterministic_and_fixed_width(
    tmp_path: Path, spec: Task1HogSpec
) -> None:
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[20:60, 15:45] = (220, 80, 40)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels, mode="RGB").save(path)

    first = extract_task1_hog(path, spec)
    second = extract_task1_hog(path, spec)

    assert first.shape == (spec.expected_features,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_hog_spec_rejects_unapproved_geometry() -> None:
    with pytest.raises(ValueError, match="expected feature count"):
        Task1HogSpec(
            hog_id="bad",
            pixels_per_cell=(8, 8),
            expected_features=1,
        )


def _cache_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12],
            "path": ["images/11.jpg", "images/12.jpg"],
            "sha256": ["a" * 64, "b" * 64],
            "partition": ["development", "development"],
            "articleType": ["class-000", "class-001"],
        }
    )


def _label_map() -> dict[str, object]:
    classes = [f"class-{index:03d}" for index in range(124)]
    return {
        "num_classes": 124,
        "classes": classes,
        "label_to_index": {label: index for index, label in enumerate(classes)},
    }


def _splits_with_images(root: Path) -> pd.DataFrame:
    image_dir = root / "images"
    image_dir.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(40):
        product_id = index + 1
        Image.new(
            "RGB",
            (60, 80),
            color=(20 + index, 60 + index, 100 + index),
        ).save(image_dir / f"{product_id}.png")
        rows.append(
            {
                "id": product_id,
                "path": f"images/{product_id}.png",
                "sha256": f"sha-{product_id}",
                "duplicate_group": f"duplicate-{product_id}",
                "product_name_key": f"name-{product_id}",
                "product_family_group": f"family-{product_id}",
                "partition": "development",
                "cv_fold": index % 5,
                "is_cross_role_exact_duplicate": False,
                "is_cross_role_near_duplicate": False,
                "has_conflicting_target_labels": False,
                "conflicting_targets": "",
                "quarantine_reason": "",
                "articleType": f"class-{index % 2:03d}",
                "season": "Summer",
                "gender": "Unisex",
                "usage": "Casual",
                "has_articleType_label": True,
                "has_season_label": True,
                "has_gender_label": True,
                "has_usage_label": True,
            }
        )
    return pd.DataFrame(rows)


def test_classical_smoke_fold_writes_124_scores_and_registry_row(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")

    result = run_task1_classical_fold(
        splits,
        _label_map(),
        validation_fold=0,
        hog_spec=TASK1_HOG_COARSE,
        model_config=Task1KNNConfig(3, "distance"),
        run_config=Task1ClassicalRunConfig.smoke(),
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        split_path=split_path,
    )

    predictions = pd.read_csv(result.prediction_path)
    assert result.status == "completed"
    assert len([name for name in predictions if name.startswith("prob_")]) == 124
    assert len(predictions) == 8
    row = registry.read().iloc[0]
    assert row["task"] == "task1"
    assert row["model_family"] == "task1_hog_knn_v1"
    assert row["benchmark_only"] == "false"
    assert row["scratch"] == "true"
    assert row["final_eligible"] == "false"
    assert row["primary_metric_name"] == "macro_f1_124"


def test_classical_smoke_fold_resumes_only_verified_completed_artifacts(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")
    kwargs = {
        "validation_fold": 0,
        "hog_spec": TASK1_HOG_COARSE,
        "model_config": Task1KNNConfig(3, "distance"),
        "run_config": Task1ClassicalRunConfig.smoke(),
        "registry": registry,
        "root": tmp_path,
        "result_root": tmp_path / "results",
        "cache_root": tmp_path / "cache",
        "split_path": split_path,
    }

    first = run_task1_classical_fold(splits, _label_map(), **kwargs)
    second = run_task1_classical_fold(splits, _label_map(), **kwargs)

    assert second.run_id == first.run_id
    assert len(registry.read()) == 1


def test_classical_smoke_fold_does_not_resume_a_different_valid_label_order(
    tmp_path: Path,
) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")
    kwargs = {
        "validation_fold": 0,
        "hog_spec": TASK1_HOG_COARSE,
        "model_config": Task1KNNConfig(3, "distance"),
        "run_config": Task1ClassicalRunConfig.smoke(),
        "registry": registry,
        "root": tmp_path,
        "result_root": tmp_path / "results",
        "cache_root": tmp_path / "cache",
        "split_path": split_path,
    }
    first = run_task1_classical_fold(splits, _label_map(), **kwargs)
    changed_label_map = _label_map()
    classes = list(reversed(changed_label_map["classes"]))
    changed_label_map["classes"] = classes
    changed_label_map["label_to_index"] = {
        label: index for index, label in enumerate(classes)
    }

    second = run_task1_classical_fold(splits, changed_label_map, **kwargs)

    assert second.run_id != first.run_id
    assert len(registry.read()) == 2


def test_classical_smoke_fold_passes_configured_batch_size_to_knn_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    received_batch_sizes: list[int] = []
    original_query = query_task1_neighbors

    def recording_query(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        *,
        max_k: int,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        received_batch_sizes.append(batch_size)
        return original_query(
            x_train, y_train, x_validation, max_k=max_k, batch_size=batch_size
        )

    monkeypatch.setattr("fashion.task1.classical.query_task1_neighbors", recording_query)
    run_task1_classical_fold(
        splits,
        _label_map(),
        validation_fold=0,
        hog_spec=TASK1_HOG_COARSE,
        model_config=Task1KNNConfig(3, "distance"),
        run_config=replace(Task1ClassicalRunConfig.smoke(), validation_batch_size=7),
        registry=RunRegistry(tmp_path / "runs.csv"),
        root=tmp_path,
        result_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        split_path=split_path,
    )

    assert received_batch_sizes == [7]


@pytest.mark.parametrize(
    "run_config",
    [
        Task1ClassicalRunConfig(stage="experiment", final_eligible=False),
        Task1ClassicalRunConfig(stage="smoke", final_eligible=True),
        Task1ClassicalRunConfig(stage="invalid", final_eligible=False),  # type: ignore[arg-type]
    ],
)
def test_classical_fold_rejects_any_non_smoke_or_non_final_contract(
    run_config: Task1ClassicalRunConfig,
) -> None:
    with pytest.raises(ValueError, match="classic run config must be either"):
        run_task1_classical_fold(
            pd.DataFrame(),
            {},
            validation_fold=0,
            hog_spec=TASK1_HOG_COARSE,
            model_config=Task1KNNConfig(3, "distance"),
            run_config=run_config,
            split_path=Path("not-used.csv"),
        )


def test_classical_full_fold_rejects_noncanonical_split_path() -> None:
    with pytest.raises(ValueError, match="canonical split path"):
        run_task1_classical_fold(
            pd.DataFrame(),
            {},
            validation_fold=0,
            hog_spec=TASK1_HOG_COARSE,
            model_config=Task1KNNConfig(3, "distance"),
            run_config=Task1ClassicalRunConfig.full(),
            split_path=Path("not-canonical.csv"),
        )


def test_classical_full_fold_rejects_altered_structurally_valid_canonical_splits() -> None:
    changed = load_splits(SPLITS_CSV).copy()
    singleton = changed.loc[
        changed.groupby("sha256")["id"].transform("size").eq(1)
        & changed.groupby("duplicate_group")["id"].transform("size").eq(1)
        & changed.groupby("product_name_key")["id"].transform("size").eq(1)
        & changed.groupby("product_family_group")["id"].transform("size").eq(1)
    ]
    first = singleton.iloc[0]
    second = singleton.loc[singleton["cv_fold"].ne(first["cv_fold"])].iloc[0]
    changed.loc[[first.name, second.name], "cv_fold"] = [second.cv_fold, first.cv_fold]
    validate_splits(changed)

    with pytest.raises(ValueError, match="supplied splits must match the canonical split file"):
        run_task1_classical_fold(
            changed,
            {},
            validation_fold=0,
            hog_spec=TASK1_HOG_COARSE,
            model_config=Task1KNNConfig(3, "distance"),
            run_config=Task1ClassicalRunConfig.full(),
            split_path=SPLITS_CSV,
        )


def test_classical_smoke_fold_finalizes_registry_when_classic_model_explodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")

    def explode(*_: object, **__: object) -> tuple[object, np.ndarray]:
        raise RuntimeError("classic model exploded")

    monkeypatch.setattr("fashion.task1.classical.fit_predict_task1_knn", explode)
    with pytest.raises(RuntimeError, match="classic model exploded"):
        run_task1_classical_fold(
            splits,
            _label_map(),
            validation_fold=0,
            hog_spec=TASK1_HOG_COARSE,
            model_config=Task1KNNConfig(3, "distance"),
            run_config=Task1ClassicalRunConfig.smoke(),
            registry=registry,
            root=tmp_path,
            result_root=tmp_path / "results",
            cache_root=tmp_path / "cache",
            split_path=split_path,
        )

    row = registry.read().iloc[0]
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "classic model exploded"


def test_hog_cache_reuses_valid_ordered_float32_features(tmp_path: Path) -> None:
    calls: list[int] = []

    def extractor(path: Path, spec: Task1HogSpec) -> np.ndarray:
        calls.append(int(path.stem))
        return np.full(spec.expected_features, int(path.stem), dtype=np.float32)

    first = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=extractor,
    )
    second = load_or_build_task1_hog_features(
        _cache_rows(),
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: pytest.fail("valid cache should be reused"),
    )

    assert calls == [11, 12]
    np.testing.assert_array_equal(first.ids, np.array([11, 12]))
    np.testing.assert_array_equal(first.features, second.features)
    assert second.features.dtype == np.float32


def test_hog_cache_rebuilds_when_source_inventory_changes(tmp_path: Path) -> None:
    rows = _cache_rows()
    load_or_build_task1_hog_features(
        rows,
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    changed = rows.copy()
    changed.loc[0, "sha256"] = "d" * 64
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(
        changed,
        TASK1_HOG_COARSE,
        root=tmp_path,
        cache_root=tmp_path / "cache",
        split_sha256="c" * 64,
        extractor=replacement,
    )

    assert calls == 2
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rejects_non_development_or_duplicate_rows(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows.loc[1, "id"] = 11

    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
        )


def test_hog_cache_rejects_holdout_rows(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows.loc[1, "partition"] = "holdout"

    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
        )


def test_hog_cache_rejects_ids_that_duplicate_after_integer_normalization(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows["id"] = ["01", "1"]

    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows,
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
            extractor=lambda *_: np.zeros(288, dtype=np.float32),
        )


def test_hog_cache_leaves_no_final_file_when_atomic_write_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_write(*_: object) -> Path:
        raise RuntimeError("write stopped")

    monkeypatch.setattr("fashion.task1.classical.atomic_write_bytes", interrupted_write)

    with pytest.raises(RuntimeError, match="write stopped"):
        load_or_build_task1_hog_features(
            _cache_rows(),
            TASK1_HOG_COARSE,
            root=tmp_path,
            cache_root=tmp_path / "cache",
            split_sha256="c" * 64,
            extractor=lambda *_: np.zeros(288, dtype=np.float32),
        )

    assert not list((tmp_path / "cache").glob("*.npz"))


def _toy_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array([[0, 0], [0, 1], [4, 4], [4, 5], [8, 8], [8, 9]], dtype=np.float32)
    y_train = np.array([2, 2, 7, 7, 19, 19], dtype=np.int64)
    x_validation = np.array([[0, 0.2], [4, 4.2], [8, 8.2]], dtype=np.float32)
    return x_train, y_train, x_validation


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_scores_expand_missing_classes_and_sum_to_one(weights: str) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1KNNConfig(n_neighbors=3, weights=weights)
    _, probabilities = fit_predict_task1_knn(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_linear_svm_scores_expand_missing_classes_and_are_finite(
    class_weight: str | None,
) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1LinearSVMConfig(C=1.0, class_weight=class_weight)
    _, probabilities = fit_predict_task1_linear_svm(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [3, 5])
def test_reused_neighbour_votes_match_sklearn(weights: str, k: int) -> None:
    x_train, y_train, x_validation = _toy_features()
    distances, indexes = query_task1_neighbors(x_train, y_train, x_validation, max_k=5)
    reused = knn_probabilities_from_neighbors(
        distances, indexes, y_train, n_neighbors=k, weights=weights
    )
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean")
    expected_local = model.fit(x_train, y_train).predict_proba(x_validation)
    expected = np.zeros((len(x_validation), 124))
    expected[:, model.classes_.astype(int)] = expected_local
    np.testing.assert_allclose(reused, expected)


def test_distance_votes_with_exact_matches_do_not_warn_or_divide_by_zero() -> None:
    distances = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    indexes = np.array([[0, 1, 2]], dtype=np.int64)
    labels = np.array([0, 1, 2], dtype=np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = knn_probabilities_from_neighbors(
            distances, indexes, labels, n_neighbors=3, weights="distance"
        )

    np.testing.assert_allclose(probabilities[0, 0], 1.0)


def test_stable_softmax_handles_extreme_finite_logits() -> None:
    logits = np.array([[1_000.0, 0.0, -1_000.0], [-1_000.0, 0.0, 1_000.0]])

    probabilities = _stable_softmax(logits)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
