from __future__ import annotations

import pytest
import torch
from torch import nn

from fashion.models.season import (
    ScratchMobileNetV3Small,
    SeasonModelSpec,
    build_season_model,
)


@pytest.mark.parametrize("shape", [(2, 3, 80, 60), (2, 3, 128, 96)])
def test_mobilenet_v3_small_supports_p0_and_p1(shape: tuple[int, ...]) -> None:
    model = build_season_model(SeasonModelSpec(family="mobilenet_v3_small"))

    logits = model(torch.randn(shape))

    assert logits.shape == (shape[0], 4)
    assert torch.isfinite(logits).all()


def test_mobilenet_is_compact_scratch_and_final_eligible() -> None:
    model = build_season_model(SeasonModelSpec(family="mobilenet_v3_small"))

    assert isinstance(model, ScratchMobileNetV3Small)
    assert model.training_origin == "scratch"
    assert model.weights is None
    assert model.final_eligible
    assert not model.benchmark_only
    assert isinstance(model.gradcam_target_layer, nn.Conv2d)
    assert model.gradcam_target_layer is model.backbone.features[-1][0]
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_mobilenet_build_never_requests_a_weight_download(monkeypatch) -> None:
    def reject_weight_download(*args, **kwargs):
        raise AssertionError("scratch model attempted to download weights")

    monkeypatch.setattr(
        "torchvision.models._api.WeightsEnum.get_state_dict",
        reject_weight_download,
    )

    model = build_season_model(SeasonModelSpec(family="mobilenet_v3_small"))

    assert model.weights is None


def test_mobilenet_backpropagates_to_stem_and_classifier() -> None:
    model = build_season_model(SeasonModelSpec(family="mobilenet_v3_small"))
    loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 80, 60)), torch.tensor([0, 3]))

    loss.backward()

    assert model.backbone.features[0][0].weight.grad is not None
    assert model.backbone.classifier[-1].weight.grad is not None
    assert torch.isfinite(model.backbone.features[0][0].weight.grad).all()


def test_mobilenet_supports_a_declared_non_rgb_input_contract() -> None:
    model = build_season_model(
        SeasonModelSpec(family="mobilenet_v3_small", input_channels=1)
    )

    logits = model(torch.randn(2, 1, 80, 60))

    assert logits.shape == (2, 4)
    assert model.backbone.features[0][0].in_channels == 1


def test_mobilenet_rejects_wrong_channel_count() -> None:
    model = build_season_model(SeasonModelSpec(family="mobilenet_v3_small"))

    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 1, 80, 60))
