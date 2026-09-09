"""Public Task 1 image-only inference API."""

from fashion.task1.final_inference import (
    InvalidTask1ImageError,
    Task1Bundle,
    Task1Prediction,
    load_task1_bundle,
    predict_article_type,
)

__all__ = [
    "InvalidTask1ImageError",
    "Task1Bundle",
    "Task1Prediction",
    "load_task1_bundle",
    "predict_article_type",
]
