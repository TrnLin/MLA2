"""Strict one-factor child experiments for the Task 3 scratch CNN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from fashion.config import ROOT, RUNS_CSV
from fashion.train.config import Task3Target

Task3ChildName = Literal["gender_brightness", "usage_class_balanced"]


@dataclass(frozen=True)
class Task3ChildSpec:
    """A predeclared child that may differ from the baseline in one factor only."""

    name: Task3ChildName
    target: Task3Target
    experiment_id: str
    hypothesis_id: str
    artifact_dir: str
    run_prefix: str
    changed_factor: str
    training_augmentation: str
    loss_name: str
    parent_run_ids: tuple[str, str, str, str, str]
    class_weight_beta: float | None = None
    class_weight_cap: float | None = None

    def __post_init__(self) -> None:
        if len(set(self.parent_run_ids)) != 5 or any(not run_id for run_id in self.parent_run_ids):
            raise ValueError("a Task 3 child requires five distinct completed parent run IDs")
        expected = {
            "gender_brightness": {
                "target": "gender",
                "experiment_id": "t3_gender_brightness_smallcnn",
                "hypothesis_id": "t3_gender_e2_brightness",
                "artifact_dir": "experiments/t3_gender_e2_brightness",
                "run_prefix": "t3_gender_e2_brightness",
                "changed_factor": "brightness_augmentation",
                "training_augmentation": "brightness_uniform_085_115",
                "loss_name": "cross_entropy",
                "class_weight_beta": None,
                "class_weight_cap": None,
            },
            "usage_class_balanced": {
                "target": "usage",
                "experiment_id": "t3_usage_class_balanced_smallcnn",
                "hypothesis_id": "t3_usage_e2_class_balanced_ce",
                "artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
                "run_prefix": "t3_usage_e2_class_balanced_ce",
                "changed_factor": "class_balanced_loss",
                "training_augmentation": "none",
                "loss_name": "effective_number_cross_entropy",
                "class_weight_beta": 0.999,
                "class_weight_cap": 5.0,
            },
        }[self.name]
        actual = asdict(self)
        actual.pop("name")
        actual.pop("parent_run_ids")
        mismatches = {
            field: {"expected": value, "actual": actual[field]}
            for field, value in expected.items()
            if actual[field] != value
        }
        if mismatches:
            raise ValueError(
                "Task 3 child changes more than its predeclared factor: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parent_run_ids"] = list(self.parent_run_ids)
        return payload


def gender_brightness_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="gender_brightness",
        target="gender",
        experiment_id="t3_gender_brightness_smallcnn",
        hypothesis_id="t3_gender_e2_brightness",
        artifact_dir="experiments/t3_gender_e2_brightness",
        run_prefix="t3_gender_e2_brightness",
        changed_factor="brightness_augmentation",
        training_augmentation="brightness_uniform_085_115",
        loss_name="cross_entropy",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
    )


def usage_class_balanced_spec(parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    return Task3ChildSpec(
        name="usage_class_balanced",
        target="usage",
        experiment_id="t3_usage_class_balanced_smallcnn",
        hypothesis_id="t3_usage_e2_class_balanced_ce",
        artifact_dir="experiments/t3_usage_e2_class_balanced_ce",
        run_prefix="t3_usage_e2_class_balanced_ce",
        changed_factor="class_balanced_loss",
        training_augmentation="none",
        loss_name="effective_number_cross_entropy",
        parent_run_ids=tuple(parent_run_ids),  # type: ignore[arg-type]
        class_weight_beta=0.999,
        class_weight_cap=5.0,
    )


def effective_number_class_weights(
    counts: Sequence[int], *, beta: float = 0.999, cap: float = 5.0
) -> np.ndarray:
    """Return fold-only mean-one effective-number weights with a fixed upper cap."""
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("class counts must be a non-empty one-dimensional sequence")
    if (values < 0).any():
        raise ValueError("class counts cannot be negative")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be between zero and one")
    if cap <= 0:
        raise ValueError("class-weight cap must be positive")
    weights = np.zeros(len(values), dtype=np.float64)
    present = values > 0
    if not present.any():
        raise ValueError("at least one class must be present")
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, values[present]))
    weights[present] /= weights[present].mean()
    weights[present] = np.minimum(weights[present], cap)
    return weights


def latest_completed_baseline_parent_run_ids(
    target: Task3Target, *, output_root: str | Path
) -> tuple[str, str, str, str, str]:
    """Find the newest complete baseline checkpoint for each canonical fold."""
    target_dir = Path(output_root) / "baseline" / target
    latest: dict[int, str] = {}
    for run_dir in sorted(target_dir.glob(f"t3_baseline_{target}_smallcnn_f*")):
        required = (
            run_dir / "config.json",
            run_dir / "final_epoch.pt",
            run_dir / "metrics.json",
            run_dir / "oof_predictions.csv",
        )
        if not all(path.is_file() for path in required):
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        fold = int(metrics["validation_fold"])
        run_id = str(metrics["run_id"])
        if run_id != run_dir.name:
            raise ValueError(f"baseline run folder and metrics disagree: {run_dir}")
        latest[fold] = run_id
    if set(latest) != set(range(5)):
        raise FileNotFoundError(
            f"expected completed {target} baseline folds 0-4; found {sorted(latest)}"
        )
    return tuple(latest[fold] for fold in range(5))  # type: ignore[return-value]


def _spec(name: Task3ChildName, parent_run_ids: Sequence[str]) -> Task3ChildSpec:
    if name == "gender_brightness":
        return gender_brightness_spec(parent_run_ids)
    return usage_class_balanced_spec(parent_run_ids)


def check_task3_child_setup(
    name: Task3ChildName,
    *,
    parent_run_ids: Sequence[str],
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Check a locked child and its parent chain without an optimiser step."""
    from fashion.train.task3_baseline import check_task3_baseline_setup

    spec = _spec(name, parent_run_ids)
    baseline = check_task3_baseline_setup(spec.target, root=root, device_name=device_name)
    return {
        **baseline,
        "child": spec.to_dict(),
        "changed_factor": spec.changed_factor,
        "optimizer_steps": 0,
        "ready": True,
    }


def run_task3_child_cv(
    name: Task3ChildName,
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Sequence[int] = range(5),
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Run one exact five-fold child while preserving its baseline controls."""
    from fashion.train.task3_baseline import run_task3_baseline_cv

    spec = _spec(name, parent_run_ids)
    return run_task3_baseline_cv(
        spec.target,
        folds=folds,
        output_root=output_root,
        registry_path=registry_path,
        registry_mirrors=registry_mirrors,
        root=root,
        device_name=device_name,
        child_spec=spec,
    )
