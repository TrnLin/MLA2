from __future__ import annotations

import pandas as pd

from fashion.config import ROOT
from fashion.train.registry import TASK2_RUN_COLUMNS


def test_live_registry_contains_exact_frozen_refit_row() -> None:
    snapshot = pd.read_csv(
        ROOT / "results/evidence/task2/final_handoff/registry_snapshot.csv",
        dtype=str,
        keep_default_na=False,
    )
    registry = pd.read_csv(
        ROOT / "results/runs.csv",
        dtype=str,
        keep_default_na=False,
    )
    run_id = snapshot.loc[0, "run_id"]
    live = registry.loc[registry["run_id"].eq(run_id), list(TASK2_RUN_COLUMNS)]

    assert len(snapshot) == 1
    assert len(live) == 1
    assert live.reset_index(drop=True).equals(
        snapshot.loc[:, list(TASK2_RUN_COLUMNS)].reset_index(drop=True)
    )
