"""Scratch model families for Fashion Intelligence classification tasks."""

from fashion.models.season import (
    ScratchSmallStemResNet18,
    SeasonModelSpec,
    SmallSeasonCNN,
    build_season_model,
)

__all__ = [
    "ScratchSmallStemResNet18",
    "SeasonModelSpec",
    "SmallSeasonCNN",
    "build_season_model",
]
