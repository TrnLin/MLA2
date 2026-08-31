"""Scratch model families for Fashion Intelligence classification tasks."""

from fashion.models.season import (
    BenchmarkModelSpec,
    BenchmarkStandardStemResNet18,
    ModelBoundaryError,
    ScratchMobileNetV3Small,
    ScratchSmallStemResNet18,
    SeasonArticleTypeMultiTaskModel,
    SeasonModelSpec,
    SmallSeasonCNN,
    assert_final_model,
    build_benchmark_model,
    build_multitask_season_model,
    build_season_model,
    model_boundary_audit,
)

__all__ = [
    "BenchmarkModelSpec",
    "BenchmarkStandardStemResNet18",
    "ModelBoundaryError",
    "ScratchMobileNetV3Small",
    "ScratchSmallStemResNet18",
    "SeasonArticleTypeMultiTaskModel",
    "SeasonModelSpec",
    "SmallSeasonCNN",
    "assert_final_model",
    "build_benchmark_model",
    "build_multitask_season_model",
    "build_season_model",
    "model_boundary_audit",
]
