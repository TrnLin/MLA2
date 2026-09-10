from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from fashion.data.hashing import compute_sha256
from fashion.task1.refit import (
    Task1FixedEpochConfig,
    run_task1_refit,
    train_task1_fixed_epochs,
)
from fashion.task1.registry import Task1RunRegistry


def _label_map() -> dict[str, object]:
    classes = [f"class-{index:03d}" for index in range(124)]
    return {
        "source_scope": "development",
        "label_column": "articleType",
        "num_classes": 124,
        "classes": classes,
        "label_to_index": {label: index for index, label in enumerate(classes)},
        "index_to_label": {str(index): label for index, label in enumerate(classes)},
    }


def _rows(root: Path, *, partition: str = "development") -> pd.DataFrame:
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for index in range(4):
        product_id = index + 1
        Image.new("RGB", (60, 80), color=(30 + index, 60, 90)).save(image_dir / f"{product_id}.png")
        rows.append(
            {
                "id": product_id,
                "path": f"images/{product_id}.png",
                "partition": partition,
                "cv_fold": index % 5,
                "articleType": f"class-{index % 2:03d}",
                "has_articleType_label": True,
            }
        )
    return pd.DataFrame(rows)


class _TinyModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(3, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output(self.pool(images).flatten(1))


def test_fixed_epoch_training_saves_the_last_epoch_without_validation(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    result = train_task1_fixed_epochs(
        rows,
        _label_map()["label_to_index"],
        config=Task1FixedEpochConfig(epochs=2, batch_size=2),
        root=tmp_path,
        device=torch.device("cpu"),
        model_factory=_TinyModel,
    )

    assert result.final_epoch == 2
    assert result.history.epoch.tolist() == [1, 2]
    assert result.history.columns.tolist() == [
        "epoch",
        "train_loss",
        "train_samples",
        "learning_rate",
    ]
    assert result.normalization.fitted_products == 4
    assert result.normalization.fitted_ids_sha256
    assert result.validation_used is False
    assert result.checkpoint_rule == "fixed_last_epoch"


def test_fixed_epoch_training_rejects_non_development_normalization_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development rows only"):
        train_task1_fixed_epochs(
            _rows(tmp_path, partition="holdout"),
            _label_map()["label_to_index"],
            config=Task1FixedEpochConfig(epochs=1, batch_size=2),
            root=tmp_path,
            device=torch.device("cpu"),
            model_factory=_TinyModel,
        )


def _write_contract(root: Path, rows: pd.DataFrame) -> tuple[Path, Path, Path]:
    split_path = root / "splits.csv"
    rows.to_csv(split_path, index=False)
    label_path = root / "label_maps.json"
    label_path.write_text(json.dumps({"articleType": _label_map()}), encoding="utf-8")
    config_path = root / "final_evaluation.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "articleType",
                "candidate_id": "task1_cnn_no_aug_unweighted_v1",
                "preprocessing_id": "task1_rgb_60x80_no_aug_v1",
                "loss_id": "cross_entropy_unweighted_v1",
                "model_family": "task1_small_cnn_v1",
                "scratch": True,
                "seed": 2753,
                "epochs": 20,
                "batch_size": 128,
                "max_lr": 0.001,
                "weight_decay": 0.00001,
                "grad_clip_norm": 1.0,
                "checkpoint_rule": "fixed_last_epoch",
                "validation_used": False,
                "augmentation_used": False,
                "normalization_scope": "development_only",
            }
        ),
        encoding="utf-8",
    )
    return split_path, label_path, config_path


def test_refit_publishes_hash_bound_bundle_and_completed_registry_row(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    split_path, label_path, config_path = _write_contract(tmp_path, rows)
    registry_path = tmp_path / "runs.csv"
    model_path = tmp_path / "models" / "task1_article_type.pt"
    manifest_path = tmp_path / "models" / "task1_article_type.manifest.json"
    history_path = tmp_path / "evidence" / "training_history.csv"

    outcome = run_task1_refit(
        mode="run",
        project_root=tmp_path,
        splits_path=split_path,
        label_map_path=label_path,
        config_path=config_path,
        model_path=model_path,
        manifest_path=manifest_path,
        history_path=history_path,
        registry_path=registry_path,
        trainer=lambda rows, labels, **kwargs: train_task1_fixed_epochs(
            rows,
            labels,
            config=Task1FixedEpochConfig(epochs=1, batch_size=2),
            root=tmp_path,
            device=torch.device("cpu"),
            model_factory=_TinyModel,
        ),
        allow_test_trainer=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert outcome.source == "trained"
    assert manifest["status"] == "test_only"
    assert manifest["final_epoch"] == 1
    assert manifest["training"]["validation_used"] is False
    assert manifest["training"]["checkpoint_rule"] == "fixed_last_epoch"
    assert manifest["canonical_inputs"]["splits"]["sha256"] == compute_sha256(split_path)
    assert manifest["bundle"]["sha256"] == compute_sha256(model_path)
    row = Task1RunRegistry(registry_path).read().iloc[0]
    assert row["status"] == "completed"
    assert row["stage"] == "final_refit"
    assert row["final_eligible"] == "false"
    assert row["best_epoch"] == "1"
    assert row["prediction_path"] == ""

    with pytest.raises(FileExistsError, match="already exist"):
        run_task1_refit(
            mode="run",
            project_root=tmp_path,
            splits_path=split_path,
            label_map_path=label_path,
            config_path=config_path,
            model_path=model_path,
            manifest_path=manifest_path,
            history_path=history_path,
            registry_path=registry_path,
        )


def test_refit_records_failure_without_publishing_outputs(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    split_path, label_path, config_path = _write_contract(tmp_path, rows)
    registry_path = tmp_path / "runs.csv"

    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        run_task1_refit(
            mode="run",
            project_root=tmp_path,
            splits_path=split_path,
            label_map_path=label_path,
            config_path=config_path,
            model_path=tmp_path / "models" / "task1_article_type.pt",
            manifest_path=tmp_path / "models" / "task1_article_type.manifest.json",
            history_path=tmp_path / "evidence" / "training_history.csv",
            registry_path=registry_path,
            trainer=explode,
            allow_test_trainer=True,
        )

    row = Task1RunRegistry(registry_path).read().iloc[0]
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert not (tmp_path / "models" / "task1_article_type.pt").exists()
    attempt = tmp_path / "evidence" / "refit_attempt.json"
    assert attempt.is_file()
    with pytest.raises(RuntimeError, match="failed or partial"):
        run_task1_refit(
            mode="run_or_load",
            project_root=tmp_path,
            splits_path=split_path,
            label_map_path=label_path,
            config_path=config_path,
            model_path=tmp_path / "models" / "task1_article_type.pt",
            manifest_path=tmp_path / "models" / "task1_article_type.manifest.json",
            history_path=tmp_path / "evidence" / "training_history.csv",
            registry_path=registry_path,
            trainer=explode,
            allow_test_trainer=True,
        )
