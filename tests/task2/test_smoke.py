from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task2.smoke import (
    G0SmokeConfig,
    load_g0_config,
    run_or_load_g0_smoke,
    select_balanced_smoke_rows,
)
from fashion.train.registry import RunRegistry

LABELS = ("Fall", "Spring", "Summer", "Winter")


def _candidate_frame(rows_per_class: int = 3) -> pd.DataFrame:
    rows = []
    item_id = 1
    for label in LABELS:
        for _ in range(rows_per_class):
            rows.append(
                {
                    "id": item_id,
                    "partition": "development",
                    "season": label,
                }
            )
            item_id += 1
    return pd.DataFrame(rows)


def test_balanced_smoke_selection_is_repeatable_and_class_complete() -> None:
    candidates = _candidate_frame()

    first = select_balanced_smoke_rows(
        candidates,
        labels=LABELS,
        per_class=2,
        seed=2753,
    )
    second = select_balanced_smoke_rows(
        candidates.sample(frac=1.0, random_state=5),
        labels=LABELS,
        per_class=2,
        seed=2753,
    )

    assert first["id"].tolist() == second["id"].tolist()
    assert first["season"].value_counts().to_dict() == {label: 2 for label in LABELS}


def test_g0_config_parser_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "g0.json"
    path.write_text(
        json.dumps({"image_size": [80, 60], "tiny_steps": 3}),
        encoding="utf-8",
    )

    config = load_g0_config(path)

    assert config.image_size == (80, 60)
    assert config.tiny_steps == 3
    with pytest.raises(ValueError, match="unknown G0"):
        G0SmokeConfig.from_dict({"learn_rate": 0.1})
    with pytest.raises(ValueError, match="cannot exceed"):
        G0SmokeConfig(tiny_per_class=3, integration_train_per_class=2).validate()


def _prepare_four_class_smoke_project(prepared_project) -> dict[str, Path]:
    root = prepared_project.root
    frame = pd.read_csv(prepared_project.splits, keep_default_na=False)
    frame = frame.loc[
        frame["path"].map(lambda value: (root / str(value)).is_file())
    ].head(8).copy()
    assert len(frame) == 8
    frame["partition"] = "development"
    frame["cv_fold"] = [0, 0, 0, 0, 1, 2, 3, 4]
    frame["season"] = [*LABELS, *LABELS]
    frame["has_season_label"] = True
    frame["sha256"] = frame["id"].map(lambda value: f"{int(value):064x}")
    frame["duplicate_group"] = frame["id"].map(lambda value: f"duplicate-{int(value)}")
    frame["product_name_key"] = frame["id"].map(lambda value: f"name-{int(value)}")
    frame["product_family_group"] = frame["id"].map(lambda value: f"family-{int(value)}")
    frame["is_cross_role_exact_duplicate"] = False
    frame["is_cross_role_near_duplicate"] = False
    frame["has_conflicting_target_labels"] = False
    frame["conflicting_targets"] = ""
    frame["quarantine_reason"] = ""
    frame.to_csv(prepared_project.splits, index=False)

    mappings = json.loads(prepared_project.label_maps.read_text(encoding="utf-8"))
    mappings["season"].update(
        {
            "num_classes": 4,
            "classes": list(LABELS),
            "label_to_index": {label: index for index, label in enumerate(LABELS)},
            "index_to_label": {str(index): label for index, label in enumerate(LABELS)},
        }
    )
    prepared_project.label_maps.write_text(
        json.dumps(mappings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "data_root": root,
        "source_root": ROOT,
        "splits_path": prepared_project.splits,
        "label_map_path": prepared_project.label_maps,
        "registry_path": root / "results/runs.csv",
        "checkpoint_directory": root / "tmp/task2/checkpoints",
        "run_directory": root / "tmp/task2/runs",
    }


def test_g0_run_registers_checkpoint_and_reuses_verified_cache(prepared_project) -> None:
    paths = _prepare_four_class_smoke_project(prepared_project)
    config = G0SmokeConfig(
        tiny_per_class=1,
        tiny_steps=1,
        minimum_tiny_accuracy=0.0,
        maximum_tiny_loss_ratio=10.0,
        integration_train_per_class=1,
        integration_validation_per_class=1,
        integration_epochs=1,
        batch_size=4,
        validation_batch_size=4,
        effective_batch_size=4,
        num_workers=0,
        use_amp=False,
        device="cpu",
    )

    first = run_or_load_g0_smoke(config, mode="run", **paths)
    second = run_or_load_g0_smoke(config, **paths)

    rows = RunRegistry(paths["registry_path"]).read()
    row = rows.iloc[0]
    assert first.source == "run"
    assert second.source == "cache"
    assert first.run_id == second.run_id == row["run_id"]
    assert first.passed and second.passed
    assert row["stage"] == "g0_smoke"
    assert row["status"] == "completed"
    assert row["final_eligible"] == "false"
    assert len(rows) == 1
    assert first.oof["id"].nunique() == 4
    assert set(first.artifacts) == {"checkpoint", "prediction", "history"}
