from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from fashion.config import SPLITS_CSV
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.splits import validate_splits
from fashion.task1.candidates import (
    TASK1_GENTLE_WEIGHTED_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
    Task1CnnCandidate,
)
from fashion.task1.losses import TASK1_UNWEIGHTED_LOSS
from fashion.task1.preprocessing import TASK1_CONTROL_PREPROCESSING
from fashion.task1.training import (
    Task1TrainConfig,
    select_training_device,
    train_task1_fold,
)
from fashion.train.registry import RunRegistry


def _label_map() -> dict[str, object]:
    classes = [f"class-{index:03d}" for index in range(124)]
    return {
        "source_scope": "development",
        "label_column": "articleType",
        "num_classes": len(classes),
        "classes": classes,
        "label_to_index": {label: index for index, label in enumerate(classes)},
        "index_to_label": {str(index): label for index, label in enumerate(classes)},
    }


def _splits_with_images(root: Path) -> pd.DataFrame:
    image_dir = root / "images"
    image_dir.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(40):
        product_id = index + 1
        fold = index % 5
        label = f"class-{index % 2:03d}"
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
                "cv_fold": fold,
                "is_cross_role_exact_duplicate": False,
                "is_cross_role_near_duplicate": False,
                "has_conflicting_target_labels": False,
                "conflicting_targets": "",
                "quarantine_reason": "",
                "articleType": label,
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


@pytest.fixture(scope="module")
def canonical_splits() -> pd.DataFrame:
    return load_splits(SPLITS_CSV)


def test_train_configs_expose_smoke_and_full_run_contracts() -> None:
    smoke = Task1TrainConfig.smoke()
    full = Task1TrainConfig.full()

    assert (smoke.stage, smoke.epochs, smoke.batch_size) == ("smoke", 1, 16)
    assert (smoke.max_train_batches, smoke.max_validation_batches) == (2, 2)
    assert smoke.final_eligible is False
    assert (full.stage, full.epochs, full.batch_size) == ("experiment", 20, 128)
    assert full.max_train_batches is None
    assert full.final_eligible is True


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_select_training_device_uses_accelerator_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert select_training_device().type == expected


def test_train_task1_fold_writes_best_checkpoint_and_registered_artifacts(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")

    result = train_task1_fold(
        splits,
        _label_map(),
        validation_fold=0,
        candidate=TASK1_NO_AUG_CANDIDATE,
        config=Task1TrainConfig.smoke(),
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "results",
        device=torch.device("cpu"),
        split_path=split_path,
    )

    assert result.status == "completed"
    assert result.fold == 0
    assert result.preprocessing_id == TASK1_CONTROL_PREPROCESSING.preprocessing_id
    assert len(pd.read_csv(result.history_path)) == 1
    predictions = pd.read_csv(result.prediction_path)
    assert len([column for column in predictions if column.startswith("prob_")]) == 124
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["metrics"] == result.metrics

    rows = registry.read()
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["status"] == "completed"
    assert row["final_eligible"] == "false"
    assert row["checkpoint_sha256"] == compute_sha256(result.checkpoint_path)
    assert row["history_sha256"] == compute_sha256(result.history_path)
    assert row["prediction_sha256"] == compute_sha256(result.prediction_path)


class _ExplodingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("model exploded")


def test_train_task1_fold_finalizes_registry_when_model_raises(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")

    with pytest.raises(RuntimeError, match="model exploded"):
        train_task1_fold(
            splits,
            _label_map(),
            validation_fold=0,
            candidate=TASK1_NO_AUG_CANDIDATE,
            config=Task1TrainConfig.smoke(),
            registry=registry,
            root=tmp_path,
            result_root=tmp_path / "results",
            device=torch.device("cpu"),
            split_path=split_path,
            model_factory=lambda _: _ExplodingModel(),
        )

    row = registry.read().iloc[0]
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "model exploded"


def test_final_run_rejects_a_structurally_valid_split_frame_that_differs_from_canonical(
    canonical_splits: pd.DataFrame,
) -> None:
    changed = canonical_splits.copy()
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
        train_task1_fold(
            changed,
            {},
            validation_fold=0,
            candidate=TASK1_NO_AUG_CANDIDATE,
            config=Task1TrainConfig.full(),
            split_path=SPLITS_CSV,
        )


def test_final_run_rejects_smoke_shaped_configuration() -> None:
    invalid_final_config = Task1TrainConfig(
        stage="smoke",
        epochs=1,
        batch_size=16,
        max_train_batches=2,
        max_validation_batches=2,
        final_eligible=True,
    )

    with pytest.raises(ValueError, match="final-eligible Task 1 runs require"):
        train_task1_fold(
            pd.DataFrame(),
            {},
            validation_fold=0,
            candidate=TASK1_NO_AUG_CANDIDATE,
            config=invalid_final_config,
            split_path=SPLITS_CSV,
        )


def test_final_run_rejects_custom_model_factory() -> None:
    with pytest.raises(ValueError, match="final-eligible Task 1 runs require Task1SmallCNN"):
        train_task1_fold(
            pd.DataFrame(),
            {},
            validation_fold=0,
            candidate=TASK1_NO_AUG_CANDIDATE,
            config=Task1TrainConfig.full(),
            split_path=SPLITS_CSV,
            model_factory=lambda _: _ExplodingModel(),
        )


@pytest.mark.parametrize(
    "field",
    ["seed", "max_lr", "weight_decay", "grad_clip_norm", "epochs", "stage"],
)
def test_final_run_rejects_changed_sealed_training_setting(field: str) -> None:
    full = Task1TrainConfig.full()
    values = {"seed": 999, "max_lr": 2e-3, "weight_decay": 2e-5, "grad_clip_norm": 2.0,
              "epochs": 19, "stage": "smoke"}
    invalid = replace(full, **{field: values[field]})
    with pytest.raises(ValueError, match="final-eligible Task 1 runs require"):
        train_task1_fold(
            pd.DataFrame(), {}, validation_fold=0,
            candidate=TASK1_NO_AUG_CANDIDATE,
            config=invalid, split_path=SPLITS_CSV,
        )


def test_final_run_rejects_unapproved_candidate() -> None:
    from fashion.task1.preprocessing import Task1PreprocessingConfig

    custom = Task1PreprocessingConfig(
        preprocessing_id="custom", horizontal_flip_probability=0.25
    )
    custom_candidate = Task1CnnCandidate("custom", custom, TASK1_UNWEIGHTED_LOSS)
    with pytest.raises(ValueError, match="approved candidate"):
        train_task1_fold(
            pd.DataFrame(), {}, validation_fold=0,
            candidate=custom_candidate, config=Task1TrainConfig.full(), split_path=SPLITS_CSV,
        )


def test_weighted_fold_records_loss_candidate_and_fold_only_weights(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")
    result = train_task1_fold(
        splits,
        _label_map(),
        validation_fold=0,
        candidate=TASK1_GENTLE_WEIGHTED_CANDIDATE,
        config=Task1TrainConfig.smoke(),
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "results",
        device=torch.device("cpu"),
        split_path=split_path,
    )

    row = registry.read().iloc[0]
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert result.candidate_id == "task1_cnn_no_aug_sqrt_weighted_v1"
    assert result.loss_id == "cross_entropy_sqrt_class_weighted_v1"
    assert row["loss_id"] == result.loss_id
    assert row["experiment_id"] == "task1-cnn-task1_cnn_no_aug_sqrt_weighted_v1"
    assert checkpoint["candidate_id"] == result.candidate_id
    assert checkpoint["loss"]["config"]["loss_id"] == result.loss_id
    assert len(checkpoint["loss"]["class_weights"]) == 124
