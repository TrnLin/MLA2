from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fashion.config import SPLITS_CSV
from fashion.data.hashing import compute_sha256
from fashion.task1.candidates import (
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
)
from fashion.task1.training import Task1FoldResult, Task1TrainConfig
from fashion.task1.weighted_experiments import run_task1_weighted_experiment
from fashion.train.artifacts import canonical_sha256
from fashion.train.registry import RunRecord, RunRegistry


def _label_map() -> dict[str, Any]:
    classes = [f"class-{index:03d}" for index in range(124)]
    return {
        "num_classes": 124,
        "classes": classes,
        "label_to_index": {name: index for index, name in enumerate(classes)},
    }


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 13),
            "partition": ["development"] * 10 + ["holdout", "quarantine"],
            "cv_fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1],
            "articleType": [f"class-{index % 2:03d}" for index in range(1, 13)],
            "has_articleType_label": [True] * 12,
        }
    )


def _fake_fold_runner(
    splits: pd.DataFrame,
    label_map: dict[str, Any],
    *,
    validation_fold: int,
    candidate: Any,
    config: Task1TrainConfig,
    registry: RunRegistry,
    root: str | Path,
    result_root: str | Path,
) -> Task1FoldResult:
    """Replace expensive training, while keeping the real registry and artifacts."""
    run_id = f"{candidate.candidate_id}-{config.stage}-{validation_fold}"
    run_dir = Path(result_root) / run_id
    run_dir.mkdir(parents=True)
    ids = splits.loc[
        splits["partition"].eq("development") & splits["cv_fold"].eq(validation_fold), "id"
    ].tolist()
    data = {"id": ids, "true_index": [value % 2 for value in ids]}
    data["predicted_index"] = data["true_index"]
    data.update(
        {f"prob_{index:03d}": [float(value % 2 == index) for value in ids] for index in range(124)}
    )
    prediction_path = run_dir / "predictions.csv"
    pd.DataFrame(data).to_csv(prediction_path, index=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fake scratch checkpoint")
    history_path = run_dir / "history.csv"
    history_path.write_text("epoch,validation_loss\n1,0.0\n", encoding="utf-8")
    metrics = {
        "macro_f1": 2 / 124,
        "weighted_f1": 1.0,
        "top1_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "validation_loss": 0.0,
    }
    record = RunRecord(
        run_id=run_id,
        experiment_id=f"task1-cnn-{candidate.candidate_id}",
        task="task1",
        stage=config.stage,
        model_family="task1_small_cnn_v1",
        fold=validation_fold,
        seed=2753,
        final_eligible=config.final_eligible,
        config_sha256="a" * 64,
        split_sha256=compute_sha256(SPLITS_CSV),
        label_map_sha256=canonical_sha256(label_map),
        implementation_sha256="b" * 64,
        transform_id=candidate.preprocessing.preprocessing_id,
        loss_id=candidate.loss.loss_id,
    )
    registry.append(record)
    for kind, path in (
        ("checkpoint", checkpoint_path),
        ("history", history_path),
        ("prediction", prediction_path),
    ):
        setattr(record, f"{kind}_path", path.relative_to(root).as_posix())
        setattr(record, f"{kind}_sha256", compute_sha256(path))
    record.status = "completed"
    record.finished_at_utc = "2026-09-05T00:00:00Z"
    record.metrics = metrics
    registry.finalize(record)
    return Task1FoldResult(
        run_id=run_id,
        fold=validation_fold,
        candidate_id=candidate.candidate_id,
        preprocessing_id=candidate.preprocessing.preprocessing_id,
        loss_id=candidate.loss.loss_id,
        status="completed",
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        prediction_path=prediction_path,
    )


@pytest.fixture()
def old_evidence(tmp_path: Path) -> tuple[RunRegistry, Path]:
    registry = RunRegistry(tmp_path / "runs.csv")
    rows = []
    for candidate in (TASK1_NO_AUG_CANDIDATE, TASK1_MILD_AUG_CANDIDATE):
        for fold in range(5):
            result = _fake_fold_runner(
                _splits(),
                _label_map(),
                validation_fold=fold,
                candidate=candidate,
                config=Task1TrainConfig.full(),
                registry=registry,
                root=tmp_path,
                result_root=tmp_path / "old_runs",
            )
            rows.append(
                {
                    "run_id": result.run_id,
                    "fold": result.fold,
                    "preprocessing_id": result.preprocessing_id,
                    **result.metrics,
                }
            )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame(rows).to_csv(evidence / "fold_metrics.csv", index=False)
    for name in ("comparison.csv", "oof_metrics.csv", "per_class_old.csv"):
        (evidence / name).write_text("existing,evidence\n1,2\n", encoding="utf-8")
    return registry, evidence


def _snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _run(tmp_path: Path, registry: RunRegistry, evidence: Path, **kwargs: Any) -> Any:
    return run_task1_weighted_experiment(
        _splits(),
        _label_map(),
        mode="full",
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "new_runs",
        evidence_root=evidence,
        fold_runner=kwargs.pop("fold_runner", _fake_fold_runner),
        **kwargs,
    )


def test_smoke_runs_only_weighted_fold_zero_without_old_evidence(tmp_path: Path) -> None:
    """Smoke must not require or emit report aggregates."""
    registry = RunRegistry(tmp_path / "runs.csv")
    evidence = tmp_path / "evidence"
    smoke = run_task1_weighted_experiment(
        _splits(),
        _label_map(),
        mode="smoke",
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "new_runs",
        evidence_root=evidence,
        fold_runner=_fake_fold_runner,
    )
    assert [(row.fold, row.candidate_id) for row in smoke.fold_results] == [
        (0, "task1_cnn_no_aug_sqrt_weighted_v1")
    ]
    assert smoke.comparison.empty and smoke.oof_metrics.empty
    assert smoke.oof_predictions == {} and smoke.per_class == {}
    assert registry.read()[["stage", "final_eligible"]].values.tolist() == [["smoke", "false"]]
    assert not evidence.exists()


@pytest.mark.parametrize("explicit_identity", [False, True])
def test_full_merges_fifteen_folds_and_preserves_old_artifacts(
    tmp_path: Path, old_evidence: tuple[RunRegistry, Path], explicit_identity: bool
) -> None:
    """Full must train only five weighted folds and preserve all ten existing runs."""
    registry, evidence = old_evidence
    if explicit_identity:
        frame = pd.read_csv(evidence / "fold_metrics.csv")
        frame["candidate_id"] = frame["preprocessing_id"].map(
            {
                "task1_rgb_60x80_no_aug_v1": "task1_cnn_no_aug_unweighted_v1",
                "task1_rgb_60x80_mild_aug_v1": "task1_cnn_mild_aug_unweighted_v1",
            }
        )
        frame["loss_id"] = "cross_entropy_unweighted_v1"
        frame.to_csv(evidence / "fold_metrics.csv", index=False)
    old_files = _snapshot(tmp_path / "old_runs")
    old_registry = registry.read()

    full = _run(tmp_path, registry, evidence)

    assert len(full.fold_results) == 15
    assert len(full.comparison) == len(full.oof_metrics) == 3
    assert set(full.fold_metrics["candidate_id"]) == {
        "task1_cnn_no_aug_unweighted_v1",
        "task1_cnn_mild_aug_unweighted_v1",
        "task1_cnn_no_aug_sqrt_weighted_v1",
    }
    assert full.fold_metrics["candidate_id"].value_counts().tolist() == [5, 5, 5]
    assert full.comparison["top1_accuracy_mean"].tolist() == [1.0] * 3
    assert all(set(frame["id"]) == set(range(1, 11)) for frame in full.oof_predictions.values())
    assert all(len(frame) == 124 for frame in full.per_class.values())
    assert _snapshot(tmp_path / "old_runs") == old_files
    pd.testing.assert_frame_equal(registry.read().iloc[:10], old_registry)
    new_rows = registry.read().iloc[10:]
    assert len(new_rows) == 5
    assert new_rows["fold"].tolist() == ["0", "1", "2", "3", "4"]
    assert set(new_rows["loss_id"]) == {"cross_entropy_sqrt_class_weighted_v1"}
    pd.testing.assert_frame_equal(pd.read_csv(evidence / "fold_metrics.csv"), full.fold_metrics)
    for candidate_id, frame in full.per_class.items():
        pd.testing.assert_frame_equal(
            pd.read_csv(evidence / f"per_class_{candidate_id}.csv"), frame
        )
    assert (evidence / "per_class_old.csv").read_text() == "existing,evidence\n1,2\n"
    assert not list(evidence.glob("*.tmp"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_id", "cross_entropy_sqrt_class_weighted_v1"),
        ("prediction_sha256", "0" * 64),
        ("fold", "4"),
        ("transform_id", "wrong-transform"),
        ("split_sha256", "0" * 64),
        ("label_map_sha256", "0" * 64),
        ("status", "failed"),
        ("final_eligible", "false"),
        ("history_path", ""),
        ("checkpoint_path", ""),
        ("prediction_path", ""),
        ("history_sha256", "0" * 64),
        ("checkpoint_sha256", "0" * 64),
    ],
)
def test_corrupt_old_registry_cannot_replace_shared_evidence(
    tmp_path: Path,
    old_evidence: tuple[RunRegistry, Path],
    field: str,
    value: str,
) -> None:
    """Missing identity/artifact checks must not let bad old evidence be published."""
    registry, evidence = old_evidence
    frame = registry.read()
    frame.loc[0, field] = value
    frame.to_csv(registry.path, index=False)
    before = _snapshot(evidence)
    with pytest.raises(ValueError):
        _run(tmp_path, registry, evidence)
    assert _snapshot(evidence) == before
    assert len(registry.read()) == 10


@pytest.mark.parametrize("corruption", ["id", "swapped_folds", "probability", "true_index"])
def test_corrupt_old_oof_cannot_replace_shared_evidence(
    tmp_path: Path,
    old_evidence: tuple[RunRegistry, Path],
    corruption: str,
) -> None:
    """Valid digests alone cannot prove correct OOF coverage or prediction contents."""
    registry, evidence = old_evidence
    rows = registry.read()
    path = tmp_path / rows.loc[0, "prediction_path"]
    frame = pd.read_csv(path)
    if corruption == "swapped_folds":
        # Pooled IDs still cover development, but the fold artifacts are exchanged.
        for field in ("prediction_path", "prediction_sha256"):
            rows.loc[0, field], rows.loc[1, field] = rows.loc[1, field], rows.loc[0, field]
    else:
        field, value = {
            "id": ("id", 11),
            "probability": ("prob_000", float("nan")),
            "true_index": ("true_index", 999),
        }[corruption]
        frame.loc[0, field] = value
        frame.to_csv(path, index=False)
        rows.loc[0, "prediction_sha256"] = compute_sha256(path)
    rows.to_csv(registry.path, index=False)
    before = _snapshot(evidence)
    with pytest.raises(ValueError):
        _run(tmp_path, registry, evidence)
    assert _snapshot(evidence) == before


@pytest.mark.parametrize(
    "corruption", ["fold", "duplicate_run", "missing_fold", "partial_identity", "metric"]
)
def test_corrupt_old_fold_table_cannot_replace_shared_evidence(
    tmp_path: Path,
    old_evidence: tuple[RunRegistry, Path],
    corruption: str,
) -> None:
    """The merge boundary must reject incomplete or ambiguous old candidates."""
    registry, evidence = old_evidence
    path = evidence / "fold_metrics.csv"
    frame = pd.read_csv(path)
    if corruption == "fold":
        frame["fold"] = frame["fold"].astype(float)
        frame.loc[0, "fold"] = 0.5
    elif corruption == "duplicate_run":
        frame.loc[1, "run_id"] = frame.loc[0, "run_id"]
    elif corruption == "missing_fold":
        frame = frame.iloc[1:]
    elif corruption == "partial_identity":
        frame["loss_id"] = "cross_entropy_unweighted_v1"
    else:
        frame.loc[0, "validation_loss"] = float("nan")
    frame.to_csv(path, index=False)
    before = _snapshot(evidence)
    with pytest.raises(ValueError):
        _run(tmp_path, registry, evidence)
    assert _snapshot(evidence) == before


@pytest.mark.parametrize("corruption", ["exception", "candidate", "artifact", "oof"])
def test_failed_new_fold_leaves_shared_evidence_untouched(
    tmp_path: Path,
    old_evidence: tuple[RunRegistry, Path],
    corruption: str,
) -> None:
    """New fold failures must not leave partial shared aggregates."""
    registry, evidence = old_evidence
    before = _snapshot(evidence)

    def broken_runner(*args: Any, **kwargs: Any) -> Task1FoldResult:
        result = _fake_fold_runner(*args, **kwargs)
        if result.fold != 4:
            return result
        if corruption == "exception":
            raise RuntimeError("fold failed")
        if corruption == "candidate":
            return replace(result, candidate_id=TASK1_NO_AUG_CANDIDATE.candidate_id)
        if corruption == "artifact":
            result.checkpoint_path.write_bytes(b"changed")
        else:
            predictions = pd.read_csv(result.prediction_path)
            predictions.loc[0, "id"] = 12
            predictions.to_csv(result.prediction_path, index=False)
            rows = registry.read()
            rows.loc[rows["run_id"].eq(result.run_id), "prediction_sha256"] = compute_sha256(
                result.prediction_path
            )
            rows.to_csv(registry.path, index=False)
        return result

    with pytest.raises((ValueError, RuntimeError)):
        _run(tmp_path, registry, evidence, fold_runner=broken_runner)
    assert _snapshot(evidence) == before


def test_all_tables_stage_before_shared_files_are_replaced(
    tmp_path: Path,
    old_evidence: tuple[RunRegistry, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late serialization failure must not replace even the first shared table."""
    registry, evidence = old_evidence
    before = _snapshot(evidence)
    original = pd.DataFrame.to_csv

    def fail_per_class(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if "class_name" in frame:
            raise OSError("disk full during staging")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_per_class)
    with pytest.raises(OSError, match="disk full"):
        _run(tmp_path, registry, evidence)
    assert _snapshot(evidence) == before
