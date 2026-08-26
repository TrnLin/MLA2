"""Scratch model families for Fashion Intelligence classification tasks."""

from fashion.models.season import (
    ScratchMobileNetV3Small,
    ScratchSmallStemResNet18,
    SeasonModelSpec,
    SmallSeasonCNN,
    build_season_model,
)

__all__ = [
    "ScratchMobileNetV3Small",
    "ScratchSmallStemResNet18",
    "SeasonModelSpec",
    "SmallSeasonCNN",
    "build_season_model",
]
