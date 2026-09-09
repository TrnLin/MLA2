from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

from fashion.data.hashing import compute_sha256
from fashion.task1.final_inference import load_task1_bundle, predict_article_type
from fashion.task1.models import Task1SmallCNN
from fashion.task1.registry import Task1RunRegistry
from fashion.train.artifacts import atomic_write_json
from fashion.train.registry import RunRecord, tracked_run


def _bundle_fixture(root: Path) -> Path:
    classes = [f"class-{index:03d}" for index in range(124)]
    model = Task1SmallCNN(124)
    model_path = root / "models" / "task1_article_type.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "format_version": 1,
            "task": "task1",
            "target": "articleType",
            "candidate_id": "task1_cnn_no_aug_unweighted_v1",
            "model_family": "task1_small_cnn_v1",
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "class_names": classes,
            "label_to_index": {label: index for index, label in enumerate(classes)},
            "preprocessing": {
                "preprocessing_id": "task1_rgb_60x80_no_aug_v1",
                "image_size": [80, 60],
                "pad_color": [255, 255, 255],
                "horizontal_flip_probability": 0.0,
                "max_rotation_degrees": 0.0,
                "max_translation_fraction": 0.0,
                "scale_range": [1.0, 1.0],
                "brightness_range": [1.0, 1.0],
                "contrast_range": [1.0, 1.0],
            },
            "normalization": {
                "mean": [0.4, 0.5, 0.6],
                "std": [0.2, 0.2, 0.2],
                "fitted_products": 32773,
                "fitted_ids_sha256": "d" * 64,
            },
            "training": {
                "seed": 2753,
                "epochs": 20,
                "final_epoch": 20,
                "validation_used": False,
                "checkpoint_rule": "fixed_last_epoch",
            },
        },
        model_path,
    )
    config_path = root / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    split_path = root / "splits.csv"
    split_path.write_text("id\n1\n", encoding="utf-8")
    labels_path = root / "labels.json"
    labels_path.write_text("{}\n", encoding="utf-8")
    history_path = root / "history.csv"
    history_path.write_text("epoch,train_loss,train_samples,learning_rate\n20,1,4,0\n")
    manifest_path = root / "models" / "task1_article_type.manifest.json"
    run_id = "task1-article-type-final-refit-test"
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "task": "task1",
        "target": "articleType",
        "candidate_id": "task1_cnn_no_aug_unweighted_v1",
        "model_family": "task1_small_cnn_v1",
        "scratch": True,
        "final_epoch": 20,
        "development_rows": 32773,
        "canonical_inputs": {
            "splits": {"path": "splits.csv", "sha256": compute_sha256(split_path)},
            "label_map": {"path": "labels.json", "sha256": compute_sha256(labels_path)},
            "config": {"path": "config.json", "sha256": compute_sha256(config_path)},
        },
        "bundle": {
            "path": "models/task1_article_type.pt",
            "sha256": compute_sha256(model_path),
        },
        "history": {"path": "history.csv", "sha256": compute_sha256(history_path)},
        "training": {
            "validation_used": False,
            "checkpoint_rule": "fixed_last_epoch",
            "normalization_scope": "development_only",
        },
    }
    atomic_write_json(manifest_path, manifest)
    registry = Task1RunRegistry(root / "runs.csv")
    record = RunRecord(
        run_id=run_id,
        task="task1",
        stage="final_refit",
        experiment_id="task1-cnn-task1_cnn_no_aug_unweighted_v1-final-refit",
        model_family="task1_small_cnn_v1",
        benchmark_only=False,
        final_eligible=True,
        scratch=True,
        fold=None,
        seed=2753,
        transform_id="task1_rgb_60x80_no_aug_v1",
        loss_id="cross_entropy_unweighted_v1",
        epochs_requested=20,
        primary_metric_name="development_training_loss",
        config_sha256=compute_sha256(config_path),
        split_sha256=compute_sha256(split_path),
        label_map_sha256=compute_sha256(labels_path),
        implementation_sha256="a" * 64,
    )
    with tracked_run(registry, record) as active:
        active.epochs_completed = 20
        active.best_epoch = 20
        active.checkpoint_path = "models/task1_article_type.pt"
        active.checkpoint_sha256 = compute_sha256(model_path)
        active.history_path = "history.csv"
        active.history_sha256 = compute_sha256(history_path)
    return manifest_path


def test_verified_bundle_is_deterministic_and_uses_no_augmentation(tmp_path: Path) -> None:
    manifest_path = _bundle_fixture(tmp_path)
    bundle = load_task1_bundle(
        manifest_path,
        registry_path=tmp_path / "runs.csv",
        project_root=tmp_path,
        device="cpu",
    )
    image_path = tmp_path / "product.png"
    Image.new("RGB", (45, 90), color=(100, 140, 180)).save(image_path)

    first = predict_article_type(bundle, image_path)
    second = predict_article_type(bundle, image_path)

    assert first.predicted_label == second.predicted_label
    assert first.probabilities == second.probabilities
    assert tuple(first.probabilities) == bundle.class_names
    assert sum(first.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert bundle.model.training is False
    assert bundle.transform.training is False
    assert bundle.transform.config.horizontal_flip_probability == 0.0


def test_bundle_rejects_a_changed_checkpoint(tmp_path: Path) -> None:
    manifest_path = _bundle_fixture(tmp_path)
    model_path = tmp_path / "models" / "task1_article_type.pt"
    model_path.write_bytes(model_path.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_task1_bundle(
            manifest_path,
            registry_path=tmp_path / "runs.csv",
            project_root=tmp_path,
            device="cpu",
        )


def test_bundle_rejects_a_changed_config(tmp_path: Path) -> None:
    manifest_path = _bundle_fixture(tmp_path)
    (tmp_path / "config.json").write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_task1_bundle(
            manifest_path,
            registry_path=tmp_path / "runs.csv",
            project_root=tmp_path,
            device="cpu",
        )
