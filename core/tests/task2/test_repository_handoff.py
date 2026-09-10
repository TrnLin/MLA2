from __future__ import annotations

import json

import pandas as pd

from fashion.config import ROOT
from fashion.train.artifacts import canonical_sha256
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


def test_shared_registry_preserves_all_recovered_task2_attempts() -> None:
    registry = pd.read_csv(
        ROOT / "results/runs.csv",
        dtype=str,
        keep_default_na=False,
    )
    task2 = registry.loc[registry["task"].eq("task2")]

    assert len(task2) == 152
    assert task2["run_id"].is_unique


def test_recovered_task2_attempts_are_all_terminal() -> None:
    registry = pd.read_csv(
        ROOT / "results/runs.csv",
        dtype=str,
        keep_default_na=False,
    )
    task2 = registry.loc[registry["task"].eq("task2")]

    assert task2["status"].value_counts().to_dict() == {
        "completed": 143,
        "interrupted": 7,
        "failed": 2,
    }


def test_registry_recovery_record_matches_shared_ledger() -> None:
    recovery = json.loads(
        (ROOT / "results/evidence/task2/registry_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    registry = pd.read_csv(
        ROOT / "results/runs.csv",
        dtype=str,
        keep_default_na=False,
    )
    rows = registry.loc[
        registry["task"].eq("task2"),
        list(TASK2_RUN_COLUMNS),
    ].to_dict(orient="records")

    assert recovery["task2_rows_canonical_sha256"] == canonical_sha256(rows)
    assert recovery["final_task2_status_counts"] == {
        "completed": 143,
        "failed": 2,
        "interrupted": 7,
    }
    assert recovery["holdout_opened"] is False
