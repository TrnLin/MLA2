from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

import fashion.task4 as task4
from fashion.task4 import training
from fashion.task4.models import build_autoencoder


def _write_source_checkpoint(tmp_path: Path) -> tuple[training.CheckpointRecord, torch.nn.Module]:
    hyperparameters = training.TrainingHyperparameters(
        warmup_epochs=1,
        planned_epochs=2,
        checkpoint_epochs=(1, 2),
    )
    session = training.TrainingSessionConfig(
        run_id="task4-portable-r5-test",
        run_kind="candidate",
        candidate=training.CandidateConfig("R5", "resnet18"),
        hyperparameters=hyperparameters,
        objective="content_mask_mse",
        source_policy=training.SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=training.AugmentationPolicy.NONE,
        validation_fold=1,
        split_fingerprint="b" * 64,
    )
    model = build_autoencoder()
    optimizer = training.build_optimizer(model, hyperparameters)
    scheduler = training.WarmupCosineScheduler(
        optimizer,
        steps_per_epoch=1,
        config=hyperparameters,
    )
    checkpoint = training.save_checkpoint(
        tmp_path / "source.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=training.make_grad_scaler("cpu"),
        epoch=1,
        session=session,
        score=0.5,
    )
    return checkpoint, model


def _write_preprocessing_files(tmp_path: Path) -> tuple[Path, Path]:
    contract = tmp_path / "preprocessing_contract.json"
    contract.write_text(
        json.dumps(
            {
                "input_contract": {
                    "colour_mode": "RGB",
                    "height": 320,
                    "pad_color": [255, 255, 255],
                    "resample": "LANCZOS",
                    "resize": "aspect_preserving_letterbox",
                    "width": 240,
                }
            }
        ),
        encoding="utf-8",
    )
    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "sources": {
                    "teacher": {
                        "mean": [0.84, 0.83, 0.82],
                        "std": [0.27, 0.28, 0.29],
                    },
                    "v1": {
                        "mean": [0.85, 0.84, 0.83],
                        "std": [0.28, 0.29, 0.30],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return contract, normalization


def test_r5_export_is_portable_inference_only_and_loadable(tmp_path: Path) -> None:
    checkpoint, source_model = _write_source_checkpoint(tmp_path)
    contract, normalization = _write_preprocessing_files(tmp_path)
    export = getattr(task4, "export_r5_inference_package", None)
    load = getattr(task4, "load_r5_inference_package", None)
    assert callable(export), "Task 4 has no portable R5 exporter"
    assert callable(load), "Task 4 has no portable R5 loader"

    package_dir, archive_path = export(
        checkpoint.path,
        tmp_path / "task4_r5",
        preprocessing_contract_path=contract,
        normalization_path=normalization,
        archive_path=tmp_path / "task4_r5_portable.zip",
    )

    weights_path = package_dir / "weights.pt"
    manifest_path = package_dir / "manifest.json"
    readme_path = package_dir / "README.md"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(weights) == {"model_state_dict", "schema_version"}
    assert all(tensor.device.type == "cpu" for tensor in weights["model_state_dict"].values())
    assert manifest["method"] == "R5"
    assert manifest["pretrained"] is False
    assert manifest["embedding_dim"] == 128
    assert manifest["weights"]["path"] == "weights.pt"
    assert manifest["weights"]["sha256"] == hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert manifest["source_checkpoint"]["sha256"] == checkpoint.sha256
    assert manifest["input_contract"]["height"] == 320
    assert manifest["normalization"]["teacher"]["mean"] == [0.84, 0.83, 0.82]
    assert readme_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "task4_r5/README.md",
            "task4_r5/manifest.json",
            "task4_r5/weights.pt",
        }

    loaded = load(package_dir)
    assert loaded.training is False
    assert all(
        torch.equal(loaded.state_dict()[name], tensor)
        for name, tensor in source_model.state_dict().items()
    )


def test_r5_loader_rejects_changed_weight_bytes(tmp_path: Path) -> None:
    checkpoint, _ = _write_source_checkpoint(tmp_path)
    contract, normalization = _write_preprocessing_files(tmp_path)
    export = getattr(task4, "export_r5_inference_package", None)
    load = getattr(task4, "load_r5_inference_package", None)
    assert callable(export), "Task 4 has no portable R5 exporter"
    assert callable(load), "Task 4 has no portable R5 loader"
    package_dir, _ = export(
        checkpoint.path,
        tmp_path / "task4_r5",
        preprocessing_contract_path=contract,
        normalization_path=normalization,
        archive_path=tmp_path / "task4_r5_portable.zip",
    )
    with (package_dir / "weights.pt").open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        load(package_dir)
