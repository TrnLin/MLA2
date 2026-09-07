"""Read Task 3 evidence without changing literal class labels."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_oof_predictions(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Pool fold predictions while preserving labels such as the Usage class ``NA``."""
    return pd.concat(
        [pd.read_csv(Path(path), keep_default_na=False) for path in paths],
        ignore_index=True,
    )
