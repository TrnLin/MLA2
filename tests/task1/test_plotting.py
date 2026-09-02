from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion.task1.plotting import (
    write_task1_comparison_figure,
    write_task1_confusion_figure,
)


def test_comparison_figure_writes_one_png(tmp_path: Path) -> None:
    """Removing the report figure must make the comparison artifact disappear."""
    metrics = pd.DataFrame(
        {
            "preprocessing_id": ["control"] * 5 + ["mild"] * 5,
            "fold": list(range(5)) * 2,
            "macro_f1": np.linspace(0.1, 0.5, 10),
        }
    )

    output = write_task1_comparison_figure(metrics, output=tmp_path / "comparison.png")

    assert output == tmp_path / "comparison.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_confusion_figure_writes_one_png(tmp_path: Path) -> None:
    """Removing the report figure must make the confusion artifact disappear."""
    predictions = pd.DataFrame({"true_index": [0, 1], "predicted_index": [0, 1]})

    output = write_task1_confusion_figure(
        predictions,
        [f"class-{index}" for index in range(124)],
        output=tmp_path / "confusion.png",
    )

    assert output == tmp_path / "confusion.png"
    assert output.is_file()
    assert output.stat().st_size > 0
