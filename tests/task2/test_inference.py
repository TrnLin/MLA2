from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import fashion.task2.inference as inference_module
from fashion.models.season import SeasonModelSpec, build_multitask_season_model
from fashion.task2.inference import (
    InvalidSeasonImageError,
    load_season_bundle,
    predict_manifest,
    predict_season,
)


def _verified_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    torch.manual_seed(2753)
    spec = SeasonModelSpec(family="smallcnn", num_classes=4)
    model = build_multitask_season_model(spec, article_type_classes=3)
    manifest_path = tmp_path / "task2_season.manifest.json"
    manifest_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
    bundle_path = tmp_path / "task2_season.pt"
    bundle_path.write_bytes(b"verified-by-fixture")
    manifest: dict[str, object] = {
        "run_id": "task2-season-i2-refit-fall-s2753-test",
        "valid_development_rows": 32_753,
        "loader_audit": {"stats": {"training_id_sha256": "a" * 64}},
        "bundle": {"path": bundle_path.name, "sha256": "b" * 64},
    }
    payload: dict[str, object] = {
        "labels": ["Fall", "Spring", "Summer", "Winter"],
        "model_spec": asdict(spec),
        "model_state_dict": model.state_dict(),
        "auxiliary": {"labels": ["A", "B", "C"]},
        "preprocessing": {
            "image_size": [80, 60],
            "content_pixel_count": 100,
            "mean": [0.4, 0.5, 0.6],
            "std": [0.2, 0.25, 0.3],
        },
        "calibration": {"temperature": 1.5, "review_threshold": None},
    }
    return manifest, manifest_path, payload


def _load_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    verified = _verified_fixture(tmp_path)
    monkeypatch.setattr(
        inference_module,
        "load_verified_development_refit_manifest",
        lambda *args, **kwargs: verified,
    )
    return load_season_bundle(
        verified[1],
        registry_path=tmp_path / "runs.csv",
        project_root=tmp_path,
        device="cpu",
    )


def test_verified_bundle_predicts_calibrated_image_only_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    image_path = tmp_path / "product.png"
    Image.new("RGB", (40, 80), color=(120, 160, 200)).save(image_path)

    prediction = predict_season(bundle, image_path)

    assert prediction.predicted_label in bundle.labels
    assert tuple(prediction.probabilities) == bundle.labels
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert prediction.confidence == prediction.probabilities[prediction.predicted_label]
    assert prediction.review_required is None
    assert prediction.run_id == "task2-season-i2-refit-fall-s2753-test"
    assert prediction.bundle_sha256 == "b" * 64
    assert prediction.latency_ms >= 0
    assert bundle.transform(image_path).shape == (3, 80, 60)
    assert bundle.model.training is False
    assert bundle.model.weights is None


def test_prediction_manifest_preserves_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    Image.new("RGB", (60, 80), color="red").save(paths[0])
    Image.new("RGB", (60, 80), color="blue").save(paths[1])

    predictions = predict_manifest(bundle, paths)

    assert [Path(item.image_path).name for item in predictions] == ["first.png", "second.png"]
    with pytest.raises(ValueError, match="at least one image"):
        predict_manifest(bundle, [])


def test_prediction_rejects_missing_and_corrupted_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        predict_season(bundle, tmp_path / "missing.png")

    corrupted = tmp_path / "corrupted.jpg"
    corrupted.write_bytes(b"not an image")
    with pytest.raises(InvalidSeasonImageError, match="Could not read a valid image"):
        predict_season(bundle, corrupted)


def test_prediction_rejects_decompression_bomb_as_invalid_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    image_path = tmp_path / "oversized.png"
    image_path.write_bytes(b"fixture")

    def reject_oversized_image(path: str | Path) -> torch.Tensor:
        raise Image.DecompressionBombError("image exceeds safe pixel limit")

    with pytest.raises(InvalidSeasonImageError, match="Could not read a valid image"):
        predict_season(replace(bundle, transform=reject_oversized_image), image_path)


class _NonFiniteModel(nn.Module):
    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full((len(images), 4), float("nan"), device=images.device)


class _FixedLogitModel(nn.Module):
    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=images.device)
        return logits.repeat(len(images), 1)


class _WrongShapeModel(nn.Module):
    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros((len(images), 3), device=images.device)


def test_prediction_rejects_non_finite_model_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    bundle = replace(bundle, model=_NonFiniteModel())
    image_path = tmp_path / "product.png"
    Image.new("RGB", (60, 80), color="white").save(image_path)

    with pytest.raises(FloatingPointError, match="non-finite"):
        predict_season(bundle, image_path)


def test_prediction_applies_frozen_temperature_to_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    bundle = replace(bundle, model=_FixedLogitModel())
    image_path = tmp_path / "product.png"
    Image.new("L", (60, 80), color=128).save(image_path)

    prediction = predict_season(bundle, image_path)
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]) / 1.5, dim=0)

    assert list(prediction.probabilities.values()) == pytest.approx(expected.tolist())
    assert prediction.predicted_label == "Winter"


def test_prediction_rejects_wrong_shape_and_missing_image_only_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _load_fixture(tmp_path, monkeypatch)
    image_path = tmp_path / "product.png"
    Image.new("RGBA", (60, 80), color=(1, 2, 3, 128)).save(image_path)

    with pytest.raises(ValueError, match="invalid probability shape"):
        predict_season(replace(bundle, model=_WrongShapeModel()), image_path)
    with pytest.raises(TypeError, match="image-only prediction method"):
        predict_season(replace(bundle, model=nn.Identity()), image_path)


@pytest.mark.parametrize("device", ["tpu", "mps"])
def test_bundle_rejects_unknown_inference_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    verified = _verified_fixture(tmp_path)
    monkeypatch.setattr(
        inference_module,
        "load_verified_development_refit_manifest",
        lambda *args, **kwargs: verified,
    )
    with pytest.raises(ValueError, match="auto, cpu, cuda"):
        load_season_bundle(verified[1], project_root=tmp_path, device=device)


def test_bundle_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_fixture(tmp_path)
    monkeypatch.setattr(
        inference_module,
        "load_verified_development_refit_manifest",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        load_season_bundle(verified[1], project_root=tmp_path, device="cuda")
