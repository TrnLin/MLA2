from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import fashion.task1.classical_models as classical_models
from fashion.config import SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.splits import validate_splits
from fashion.task1.classical_features import TASK1_HOG_COARSE, Task1HogSpec
from fashion.task1.classical_models import Task1KNNConfig
from fashion.task1.classical_training import (
    Task1ClassicalRunConfig,
    run_task1_classical_fold,
)
from fashion.train.registry import RunRegistry


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


def test_classical_smoke_fold_does_not_resume_a_stale_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")
    implementation_sha256 = ["a" * 64]
    monkeypatch.setattr(
        "fashion.task1.classical_training._classical_implementation_sha256",
        lambda: implementation_sha256[0],
    )
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
    implementation_sha256[0] = "b" * 64
    second = run_task1_classical_fold(splits, _label_map(), **kwargs)

    assert second.run_id != first.run_id
    assert registry.read()["implementation_sha256"].tolist() == ["a" * 64, "b" * 64]


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
    original_query = classical_models.query_task1_neighbors

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

    monkeypatch.setattr("fashion.task1.classical_models.query_task1_neighbors", recording_query)
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


def test_classical_full_fold_rejects_hog_spec_outside_frozen_candidates() -> None:
    custom_hog = Task1HogSpec(
        hog_id="task1_gray_hog_ppc20_unapproved",
        pixels_per_cell=(20, 20),
        expected_features=216,
    )

    with pytest.raises(ValueError, match="frozen Task 1 HOG specs"):
        run_task1_classical_fold(
            pd.DataFrame(),
            {},
            validation_fold=0,
            hog_spec=custom_hog,
            model_config=Task1KNNConfig(3, "distance"),
            run_config=Task1ClassicalRunConfig.full(),
            split_path=Path("not-used.csv"),
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

    monkeypatch.setattr("fashion.task1.classical_training.fit_predict_task1_knn", explode)
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
