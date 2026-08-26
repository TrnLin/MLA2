"""Scratch model families for Fashion Intelligence classification tasks."""

from fashion.models.season import SeasonModelSpec, SmallSeasonCNN, build_season_model

__all__ = ["SeasonModelSpec", "SmallSeasonCNN", "build_season_model"]
