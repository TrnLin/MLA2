from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights

import fashion.task4 as task4
from fashion.task4 import models as task4_models
from fashion.task4.models import (
    B1_WEIGHT_ORIGIN,
    ConvolutionalAutoencoder,
    ModelMetadata,
    RetrievalEncoder,
    build_autoencoder,
    build_b1_encoder,
    build_retrieval_encoder,
)


@pytest.mark.parametrize("architecture", ["resnet18", "resnet34"])
def test_retrieval_encoder_returns_unnormalized_128_value_projection(
    architecture: str,
) -> None:
    model = build_retrieval_encoder(architecture)
    model.eval()

    with torch.inference_mode():
        projection = model(torch.randn(2, 3, 64, 64))

    assert projection.shape == (2, 128)
    assert projection.dtype == torch.float32
    assert not torch.allclose(
        torch.linalg.vector_norm(projection, dim=1),
        torch.ones(2),
        atol=1e-4,
    )


def test_projection_head_has_projector_ruled_dimensions_and_no_final_activation() -> None:
    model = build_retrieval_encoder("resnet18")

    layers = list(model.projection)
    linear_shapes = [
        (layer.in_features, layer.out_features)
        for layer in layers
        if isinstance(layer, nn.Linear)
    ]

    assert linear_shapes == [(512, 512), (512, 512), (512, 128)]
    assert [type(layer) for layer in layers] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
    ]


def test_scratch_factory_requests_no_torchvision_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[object] = []

    def fake_resnet18(*, weights: object) -> nn.Module:
        requested.append(weights)
        return _TinyBackbone()

    monkeypatch.setattr("fashion.task4.models.torchvision_models.resnet18", fake_resnet18)

    model = build_retrieval_encoder("resnet18")

    assert requested == [None]
    assert model.metadata == ModelMetadata(
        architecture="resnet18",
        pretrained=False,
        weight_origin="random_initialization",
        deployment_eligible=True,
    )


def test_metadata_is_immutable() -> None:
    metadata = build_retrieval_encoder("resnet18").metadata

    with pytest.raises(FrozenInstanceError):
        metadata.pretrained = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        ("resnet50", False, "random_initialization", True),
        ("resnet18", False, B1_WEIGHT_ORIGIN, True),
        ("resnet18", True, "random_initialization", False),
        ("resnet34", True, B1_WEIGHT_ORIGIN, False),
        ("resnet18", True, B1_WEIGHT_ORIGIN, True),
        ("resnet18", 1, B1_WEIGHT_ORIGIN, False),
        ("resnet18", False, "random_initialization", 1),
    ],
)
def test_metadata_rejects_every_contradictory_combination(
    values: tuple[object, object, object, object],
) -> None:
    with pytest.raises(ValueError, match="metadata"):
        ModelMetadata(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model_type", "metadata"),
    [
        (
            RetrievalEncoder,
            ModelMetadata("resnet18", False, "random_initialization", True),
        ),
        (
            ConvolutionalAutoencoder,
            ModelMetadata("resnet18", False, "random_initialization", True),
        ),
    ],
)
def test_direct_model_constructors_are_rejected(
    model_type: type[nn.Module],
    metadata: ModelMetadata,
) -> None:
    with pytest.raises(RuntimeError, match="factory"):
        model_type(_TinyBackbone(), metadata)


def test_model_metadata_binding_cannot_be_replaced() -> None:
    model = build_retrieval_encoder()
    replacement = ModelMetadata("resnet34", False, "random_initialization", True)

    with pytest.raises(AttributeError, match="metadata"):
        model.metadata = replacement  # type: ignore[misc]


@pytest.mark.parametrize(
    ("architecture", "weights", "deployment_eligible", "accepted"),
    [
        ("resnet18", None, True, True),
        ("resnet18", None, False, True),
        ("resnet34", None, True, True),
        ("resnet34", None, False, True),
        ("resnet18", ResNet18_Weights.IMAGENET1K_V1, False, True),
        ("resnet18", ResNet18_Weights.IMAGENET1K_V1, True, False),
        ("resnet34", ResNet18_Weights.IMAGENET1K_V1, False, False),
        ("resnet34", ResNet34_Weights.IMAGENET1K_V1, False, False),
        ("resnet18", True, False, False),
        ("resnet18", "ResNet18_Weights.IMAGENET1K_V1", False, False),
    ],
)
def test_factory_accepts_only_supported_weight_and_eligibility_combinations(
    architecture: str,
    weights: object,
    deployment_eligible: bool,
    accepted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fashion.task4.models.torchvision_models.resnet18",
        lambda *, weights: _TinyBackbone(),
    )
    monkeypatch.setattr(
        "fashion.task4.models.torchvision_models.resnet34",
        lambda *, weights: _TinyBackbone(),
    )

    if not accepted:
        with pytest.raises(ValueError, match="pretrained|pinned|weights"):
            build_retrieval_encoder(
                architecture,  # type: ignore[arg-type]
                weights=weights,  # type: ignore[arg-type]
                deployment_eligible=deployment_eligible,
            )
        return

    model = build_retrieval_encoder(
        architecture,  # type: ignore[arg-type]
        weights=weights,  # type: ignore[arg-type]
        deployment_eligible=deployment_eligible,
    )
    assert model.metadata.deployment_eligible is deployment_eligible
    assert model.metadata.pretrained is (weights is not None)


def test_factory_rejects_pretrained_deployment_eligibility() -> None:
    with pytest.raises(ValueError, match="pretrained.*eligible"):
        build_retrieval_encoder(
            "resnet18",
            weights=ResNet18_Weights.IMAGENET1K_V1,
            deployment_eligible=True,
        )


def test_b1_factory_uses_only_pinned_weight_and_is_comparison_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[object] = []

    def fake_resnet18(*, weights: object) -> nn.Module:
        requested.append(weights)
        return _TinyBackbone()

    monkeypatch.setattr("fashion.task4.models.torchvision_models.resnet18", fake_resnet18)

    model = build_b1_encoder()

    assert requested == [ResNet18_Weights.IMAGENET1K_V1]
    assert B1_WEIGHT_ORIGIN == "ResNet18_Weights.IMAGENET1K_V1"
    assert model.metadata == ModelMetadata(
        architecture="resnet18",
        pretrained=True,
        weight_origin=B1_WEIGHT_ORIGIN,
        deployment_eligible=False,
    )


def test_encode_returns_finite_float32_unit_embeddings() -> None:
    model = build_retrieval_encoder("resnet18")
    model.eval()

    with torch.inference_mode():
        embeddings = model.encode(torch.randn(2, 3, 64, 64))

    assert embeddings.shape == (2, 128)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2),
        atol=1e-5,
    )


@pytest.mark.parametrize("bad_value", [0.0, float("nan"), float("inf")])
def test_encode_rejects_zero_or_nonfinite_projection(bad_value: float) -> None:
    model = build_retrieval_encoder("resnet18")
    model.eval()
    model.forward = lambda images: torch.full(  # type: ignore[method-assign]
        (images.shape[0], 128),
        bad_value,
        device=images.device,
    )

    with pytest.raises(ValueError, match="finite|non-zero"):
        model.encode(torch.randn(2, 3, 16, 16))


@pytest.mark.parametrize("model_kind", ["encoder", "autoencoder"])
def test_encode_rejects_finite_float32_values_whose_norm_overflows(
    model_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_retrieval_encoder() if model_kind == "encoder" else build_autoencoder()
    model.eval()
    overflow = torch.full((1, 128), torch.finfo(torch.float32).max)
    if model_kind == "encoder":
        monkeypatch.setattr(model, "forward", lambda images: overflow)
    else:
        monkeypatch.setattr(model, "_bottleneck", lambda images: overflow)

    with pytest.raises(ValueError, match="norm.*finite"):
        model.encode(torch.randn(1, 3, 16, 16))


def test_encode_rejects_nonunit_post_normalization_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_retrieval_encoder()
    model.eval()
    monkeypatch.setattr(
        task4_models.F,
        "normalize",
        lambda values, **kwargs: values / 2.0,
    )

    with pytest.raises(ValueError, match="unit norm"):
        model.encode(torch.ones(1, 3, 64, 64))


def test_encoder_encode_supports_batch_one_in_eval_mode() -> None:
    model = build_retrieval_encoder()
    model.eval()

    with torch.inference_mode():
        embedding = model.encode(torch.randn(1, 3, 64, 64))

    assert embedding.shape == (1, 128)


@pytest.mark.parametrize("builder", [build_retrieval_encoder, build_autoencoder])
def test_encode_does_not_change_batchnorm_running_statistics(builder: object) -> None:
    model = builder()  # type: ignore[operator]
    model.eval()
    batchnorms = [
        module for module in model.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    before = [(layer.running_mean.clone(), layer.running_var.clone()) for layer in batchnorms]

    with torch.inference_mode():
        model.encode(torch.randn(1, 3, 64, 64))

    after = [(layer.running_mean, layer.running_var) for layer in batchnorms]
    assert all(
        torch.equal(before_mean, after_mean) and torch.equal(before_var, after_var)
        for (before_mean, before_var), (after_mean, after_var) in zip(before, after, strict=True)
    )


def test_training_forward_updates_batchnorm_running_statistics() -> None:
    model = build_retrieval_encoder()
    model.train()
    batchnorms = [
        module for module in model.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    before = [layer.running_mean.clone() for layer in batchnorms]

    model(torch.randn(4, 3, 64, 64))

    assert any(
        not torch.equal(previous, layer.running_mean)
        for previous, layer in zip(before, batchnorms, strict=True)
    )


@pytest.mark.parametrize("builder", [build_retrieval_encoder, build_autoencoder])
def test_encode_rejects_training_mode_with_clear_error(builder: object) -> None:
    model = builder()  # type: ignore[operator]
    model.train()

    with pytest.raises(RuntimeError, match="eval|retrieval"):
        model.encode(torch.randn(1, 3, 16, 16))


def test_autoencoder_matches_real_input_and_bottleneck_shapes() -> None:
    model = build_autoencoder()
    model.eval()

    with torch.inference_mode():
        reconstruction, bottleneck = model(torch.randn(1, 3, 320, 240))

    assert reconstruction.shape == (1, 3, 320, 240)
    assert bottleneck.shape == (1, 128)
    assert bottleneck.dtype == torch.float32


def test_autoencoder_decoder_receives_only_bottleneck() -> None:
    model = build_autoencoder()
    model.eval()
    decoder_inputs: list[tuple[torch.Tensor, ...]] = []
    handle = model.decoder.register_forward_pre_hook(
        lambda _module, inputs: decoder_inputs.append(inputs)
    )
    try:
        with torch.inference_mode():
            model(torch.randn(1, 3, 64, 64))
    finally:
        handle.remove()

    assert len(decoder_inputs) == 1
    assert len(decoder_inputs[0]) == 1
    assert decoder_inputs[0][0].shape == (1, 128)


def test_autoencoder_encode_normalizes_bottleneck() -> None:
    model = build_autoencoder()
    model.eval()

    with torch.inference_mode():
        embedding = model.encode(torch.randn(1, 3, 64, 64))

    assert embedding.shape == (1, 128)
    assert embedding.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(embedding, dim=1), torch.ones(1))


def test_autoencoder_is_scratch_resnet18(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[object] = []

    def fake_resnet18(*, weights: object) -> nn.Module:
        requested.append(weights)
        return _TinyBackbone()

    monkeypatch.setattr("fashion.task4.models.torchvision_models.resnet18", fake_resnet18)

    model = build_autoencoder()

    assert requested == [None]
    assert isinstance(model, ConvolutionalAutoencoder)
    assert model.metadata.architecture == "resnet18"
    assert model.metadata.pretrained is False
    assert model.metadata.weight_origin == "random_initialization"
    assert model.metadata.deployment_eligible is True


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones((images.shape[0], 512), device=images.device)


def test_task4_package_exports_learned_model_api() -> None:
    assert task4.RetrievalEncoder is RetrievalEncoder
    assert task4.ConvolutionalAutoencoder is ConvolutionalAutoencoder
    assert task4.build_retrieval_encoder is build_retrieval_encoder
    assert task4.build_b1_encoder is build_b1_encoder
    assert task4.build_autoencoder is build_autoencoder
