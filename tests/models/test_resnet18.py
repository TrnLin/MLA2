from __future__ import annotations

import pytest
import torch
from torch import nn

from fashion.models.season import (
    ScratchSmallStemResNet18,
    SeasonModelSpec,
    build_season_model,
)


@pytest.mark.parametrize("shape", [(2, 3, 80, 60), (2, 3, 128, 96)])
def test_small_stem_resnet18_supports_p0_and_p1(shape: tuple[int, ...]) -> None:
    model = build_season_model(SeasonModelSpec(family="resnet18_small_stem"))

    logits = model(torch.randn(shape))

    assert logits.shape == (shape[0], 4)
    assert torch.isfinite(logits).all()


def test_resnet18_uses_small_image_stem_and_scratch_boundary() -> None:
    model = build_season_model(SeasonModelSpec(family="resnet18_small_stem"))

    assert isinstance(model, ScratchSmallStemResNet18)
    assert model.training_origin == "scratch"
    assert model.weights is None
    assert model.final_eligible
    assert not model.benchmark_only
    assert model.backbone.conv1.kernel_size == (3, 3)
    assert model.backbone.conv1.stride == (1, 1)
    assert model.backbone.conv1.padding == (1, 1)
    assert isinstance(model.backbone.maxpool, nn.Identity)
    assert isinstance(model.gradcam_target_layer, nn.Conv2d)
    assert model.gradcam_target_layer is model.backbone.layer4[-1].conv2


def test_resnet18_build_never_requests_a_weight_download(monkeypatch) -> None:
    def reject_weight_download(*args, **kwargs):
        raise AssertionError("scratch model attempted to download weights")

    monkeypatch.setattr(
        "torchvision.models._api.WeightsEnum.get_state_dict",
        reject_weight_download,
    )

    model = build_season_model(SeasonModelSpec(family="resnet18_small_stem"))

    assert model.weights is None


def test_resnet18_backpropagates_to_stem_and_classifier() -> None:
    model = build_season_model(SeasonModelSpec(family="resnet18_small_stem"))
    loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 80, 60)), torch.tensor([1, 2]))

    loss.backward()

    assert model.backbone.conv1.weight.grad is not None
    assert model.backbone.fc[-1].weight.grad is not None
    assert torch.isfinite(model.backbone.conv1.weight.grad).all()


def test_resnet18_rejects_wrong_channel_count() -> None:
    model = build_season_model(SeasonModelSpec(family="resnet18_small_stem"))

    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 1, 80, 60))
