from __future__ import annotations

import json

import pandas as pd

from fashion.config import ROOT
from fashion.train.artifacts import canonical_sha256
from fashion.train.registry import TASK2_RUN_COLUMNS

POST_SUBMISSION_PREFIX = "postsubmit-"


def _task2_registry_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(
        ROOT / "results/runs.csv",
        dtype=str,
        keep_default_na=False,
    )
    task2 = registry.loc[registry["task"].eq("task2")]
    post_submission = task2.loc[
        task2["experiment_id"].str.startswith(POST_SUBMISSION_PREFIX)
    ]
    frozen = task2.loc[~task2.index.isin(post_submission.index)]
    return frozen, post_submission


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
    frozen, post_submission = _task2_registry_partitions()

    assert len(frozen) == 152
    assert frozen["run_id"].is_unique
    assert post_submission["run_id"].is_unique


def test_recovered_task2_attempts_are_all_terminal() -> None:
    frozen, _ = _task2_registry_partitions()

    assert frozen["status"].value_counts().to_dict() == {
        "completed": 143,
        "interrupted": 7,
        "failed": 2,
    }


def test_post_submission_attempts_are_isolated_from_frozen_handoff() -> None:
    _, post_submission = _task2_registry_partitions()

    assert len(post_submission) >= 14
    assert set(post_submission["status"]) <= {"completed", "failed", "interrupted"}
    assert post_submission["benchmark_only"].str.lower().eq("true").all()
    assert post_submission["final_eligible"].str.lower().eq("false").all()
    assert post_submission["stage"].str.startswith("post_submission_").all()

    completed = post_submission.loc[post_submission["status"].eq("completed")]
    for experiment_id in (
        "postsubmit-i2-embedding-random-forest",
        "postsubmit-i2-embedding-hist-gradient-boosting",
    ):
        folds = set(
            pd.to_numeric(
                completed.loc[completed["experiment_id"].eq(experiment_id), "fold"]
            ).astype(int)
        )
        assert folds == set(range(5))

    leaky = post_submission.loc[
        post_submission["experiment_id"].eq("postsubmit-i2-leaky-relu-0-01")
    ]
    assert {"failed", "interrupted"} <= set(leaky["status"])


def test_registry_recovery_record_matches_shared_ledger() -> None:
    recovery = json.loads(
        (ROOT / "results/evidence/task2/registry_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    frozen, _ = _task2_registry_partitions()
    rows = frozen.loc[:, list(TASK2_RUN_COLUMNS)].to_dict(orient="records")

    assert recovery["task2_rows_canonical_sha256"] == canonical_sha256(rows)
    assert recovery["final_task2_status_counts"] == {
        "completed": 143,
        "failed": 2,
        "interrupted": 7,
    }
    assert recovery["holdout_opened"] is False
