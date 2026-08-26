from __future__ import annotations

import pytest
import torch
from torch import nn

from fashion.models.season import SeasonModelSpec, SmallSeasonCNN, build_season_model
from fashion.train.reproducibility import seed_everything


@pytest.mark.parametrize("shape", [(2, 3, 80, 60), (2, 3, 128, 96)])
def test_smallcnn_supports_both_predeclared_image_sizes(shape: tuple[int, ...]) -> None:
    model = build_season_model(SeasonModelSpec(family="smallcnn"))

    logits = model(torch.randn(shape))

    assert logits.shape == (shape[0], 4)
    assert torch.isfinite(logits).all()


def test_smallcnn_is_scratch_final_eligible_and_has_gradcam_layer() -> None:
    model = build_season_model(SeasonModelSpec(family="smallcnn"))

    assert isinstance(model, SmallSeasonCNN)
    assert model.training_origin == "scratch"
    assert model.final_eligible
    assert not model.benchmark_only
    assert isinstance(model.gradcam_target_layer, nn.Conv2d)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_171_620


def test_smallcnn_initialization_is_repeatable_after_shared_seed() -> None:
    seed_everything(2753)
    first = build_season_model(SeasonModelSpec(family="smallcnn"))
    seed_everything(2753)
    second = build_season_model(SeasonModelSpec(family="smallcnn"))

    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_smallcnn_backpropagates_to_early_and_final_layers() -> None:
    model = build_season_model(SeasonModelSpec(family="smallcnn"))
    loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 80, 60)), torch.tensor([0, 3]))

    loss.backward()

    assert model.features[0][0].weight.grad is not None
    assert model.classifier[-1].weight.grad is not None
    assert torch.isfinite(model.features[0][0].weight.grad).all()


def test_smallcnn_rejects_wrong_input_contract() -> None:
    model = build_season_model(SeasonModelSpec(family="smallcnn"))

    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 1, 80, 60))


@pytest.mark.parametrize(
    "spec",
    [
        SeasonModelSpec(family="unknown"),
        SeasonModelSpec(family="smallcnn", num_classes=1),
        SeasonModelSpec(family="smallcnn", dropout=1.0),
    ],
)
def test_season_model_spec_rejects_invalid_settings(spec: SeasonModelSpec) -> None:
    with pytest.raises(ValueError):
        build_season_model(spec)
