from __future__ import annotations

import pytest
import torch
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18 as torchvision_resnet18

from fashion.models.season import (
    BenchmarkModelSpec,
    BenchmarkStandardStemResNet18,
    ModelBoundaryError,
    SeasonModelSpec,
    assert_final_model,
    build_benchmark_model,
    build_multitask_season_model,
    build_season_model,
    model_boundary_audit,
)


@pytest.mark.parametrize(
    "family",
    ["smallcnn", "resnet18_small_stem", "mobilenet_v3_small"],
)
def test_every_final_family_passes_scratch_boundary(family: str) -> None:
    model = build_season_model(SeasonModelSpec(family=family))

    audit = assert_final_model(model)

    assert audit["training_origin"] == "scratch"
    assert audit["weights"] is None
    assert not audit["benchmark_only"]
    assert audit["final_eligible"]


def test_standard_stem_scratch_control_is_benchmark_only() -> None:
    model = build_benchmark_model(
        BenchmarkModelSpec(family="resnet18_standard_scratch")
    )

    assert isinstance(model, BenchmarkStandardStemResNet18)
    assert model.training_origin == "scratch"
    assert model.weights is None
    assert model.backbone.conv1.kernel_size == (7, 7)
    assert model.backbone.conv1.stride == (2, 2)
    assert model.benchmark_only
    assert not model.final_eligible
    with pytest.raises(ModelBoundaryError, match="benchmark_only"):
        assert_final_model(model)


def test_pretrained_builder_is_explicit_and_never_final_eligible(monkeypatch) -> None:
    observed = {}

    def local_resnet18(*, weights):
        observed["weights"] = weights
        return torchvision_resnet18(weights=None)

    monkeypatch.setattr("fashion.models.season.resnet18", local_resnet18)

    model = build_benchmark_model(
        BenchmarkModelSpec(family="resnet18_standard_pretrained")
    )

    assert observed["weights"] is ResNet18_Weights.DEFAULT
    assert model.training_origin == "imagenet_pretrained"
    assert model.weights == "ResNet18_Weights.DEFAULT"
    assert model_boundary_audit(model)["benchmark_only"]
    with pytest.raises(ModelBoundaryError, match="training_origin"):
        assert_final_model(model)


def test_final_builder_does_not_accept_benchmark_family_names() -> None:
    with pytest.raises(ValueError, match="unknown Season model family"):
        build_season_model(SeasonModelSpec(family="resnet18_standard_pretrained"))


@pytest.mark.parametrize(
    "family",
    ["smallcnn", "resnet18_small_stem", "mobilenet_v3_small"],
)
def test_multitask_model_has_two_heads_but_image_only_season_inference(family: str) -> None:
    model = build_multitask_season_model(
        SeasonModelSpec(family=family),
        article_type_classes=124,
    )
    images = torch.randn(2, 3, 80, 60)

    outputs = model(images)
    season_only = model.predict_season_logits(images)

    assert outputs["season_logits"].shape == (2, 4)
    assert outputs["article_type_logits"].shape == (2, 124)
    assert season_only.shape == (2, 4)
    assert_final_model(model)


def test_multitask_model_rejects_benchmark_base() -> None:
    benchmark = build_benchmark_model(
        BenchmarkModelSpec(family="resnet18_standard_scratch")
    )

    from fashion.models.season import SeasonArticleTypeMultiTaskModel

    with pytest.raises(ModelBoundaryError):
        SeasonArticleTypeMultiTaskModel(benchmark)
