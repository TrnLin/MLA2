from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import fashion.task1.cnn_experiments as cnn_experiments
from fashion.task1.cnn_experiments import run_task1_experiment
from fashion.task1.training import Task1FoldResult


@dataclass(frozen=True)
class _FoldCall:
    fold: int
    transform_id: str
    stage: str


def _label_map() -> dict[str, object]:
    classes = [f"class-{index:03d}" for index in range(124)]
    return {
        "num_classes": len(classes),
        "classes": classes,
        "label_to_index": {name: index for index, name in enumerate(classes)},
    }


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "partition": ["development"] * 10,
            "cv_fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "has_articleType_label": [True] * 10,
        }
    )


def _prediction_frame(ids: list[int]) -> pd.DataFrame:
    data: dict[str, list[int] | list[float]] = {
        "id": ids,
        "true_index": [index % 2 for index in ids],
        "predicted_index": [index % 2 for index in ids],
    }
    data.update(
        {
            f"prob_{index:03d}": [1.0 if value % 2 == index else 0.0 for value in ids]
            for index in range(124)
        }
    )
    return pd.DataFrame(data)


def _fake_fold_runner(
    splits: pd.DataFrame,
    _: dict[str, object],
    *,
    validation_fold: int,
    preprocessing: Any,
    config: Any,
    result_root: str | Path,
    **__: object,
) -> Task1FoldResult:
    """Return a complete, fixed fold artifact without running a CNN."""
    _fake_fold_runner.calls.append(
        _FoldCall(validation_fold, preprocessing.preprocessing_id, config.stage)
    )
    prediction_path = Path(result_root) / f"{preprocessing.preprocessing_id}-{validation_fold}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame(splits.loc[splits["cv_fold"].eq(validation_fold), "id"].tolist()).to_csv(
        prediction_path,
        index=False,
    )
    return Task1FoldResult(
        run_id=f"{preprocessing.preprocessing_id}-{validation_fold}",
        fold=validation_fold,
        preprocessing_id=preprocessing.preprocessing_id,
        status="completed",
        metrics={
            "macro_f1": 0.5,
            "weighted_f1": 0.5,
            "top1_accuracy": 0.5,
            "top5_accuracy": 0.5,
        },
        checkpoint_path=prediction_path.with_suffix(".pt"),
        history_path=prediction_path.with_name("history.csv"),
        prediction_path=prediction_path,
    )


_fake_fold_runner.calls = []  # type: ignore[attr-defined]


def test_runner_uses_one_control_fold_for_smoke_and_ten_fixed_full_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the run schedule must fail this test before physical runs are launched."""
    monkeypatch.setattr(cnn_experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
    splits = _splits()
    _fake_fold_runner.calls.clear()  # type: ignore[attr-defined]

    smoke = run_task1_experiment(
        splits, _label_map(), mode="smoke", fold_runner=_fake_fold_runner, result_root=tmp_path
    )

    calls = _fake_fold_runner.calls  # type: ignore[attr-defined]
    assert [(call.fold, call.transform_id) for call in calls] == [
        (0, "task1_rgb_60x80_no_aug_v1")
    ]
    assert [call.stage for call in calls] == ["smoke"]
    assert smoke.mode == "smoke"
    assert len(smoke.fold_metrics) == 1
    assert smoke.comparison.empty

    calls.clear()
    full = run_task1_experiment(
        splits, _label_map(), mode="full", fold_runner=_fake_fold_runner, result_root=tmp_path
    )

    assert len(calls) == 10
    assert {call.fold for call in calls} == set(range(5))
    assert [call.transform_id for call in calls] == [
        "task1_rgb_60x80_no_aug_v1"
    ] * 5 + ["task1_rgb_60x80_mild_aug_v1"] * 5
    assert len(full.fold_metrics) == 10
    assert len(full.comparison) == 2
    assert list(full.oof_metrics.columns) == [
        "preprocessing_id", "macro_f1_124", "weighted_f1", "top1_accuracy", "top5_accuracy"
    ]
    assert len(full.oof_metrics) == 2
    assert set(full.oof_predictions) == {
        "task1_rgb_60x80_no_aug_v1",
        "task1_rgb_60x80_mild_aug_v1",
    }
    assert all(len(frame) == 124 for frame in full.per_class.values())
    assert (tmp_path / "evidence" / "fold_metrics.csv").is_file()
    assert (tmp_path / "evidence" / "oof_metrics.csv").is_file()


def test_runner_propagates_fold_exception_without_writing_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fold must stop before incomplete report evidence is emitted."""

    def exploding_runner(*_: object, **__: object) -> Task1FoldResult:
        raise RuntimeError("fold exploded")

    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(cnn_experiments, "TASK1_EVIDENCE_DIR", evidence_root)
    with pytest.raises(RuntimeError, match="fold exploded"):
        run_task1_experiment(
            _splits(),
            _label_map(),
            mode="full",
            fold_runner=exploding_runner,
            result_root=tmp_path / "runs",
        )

    assert not list(evidence_root.glob("*.csv"))


@pytest.mark.parametrize(
    ("mutate_predictions", "message"),
    [
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (lambda frame: frame.iloc[1:].copy(), "expected development IDs"),
    ],
)
def test_full_runner_rejects_invalid_oof_coverage(
    tmp_path: Path,
    mutate_predictions: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broken fold artifacts must not be summarised as cross-validation evidence."""

    def invalid_artifact_runner(*args: object, **kwargs: object) -> Task1FoldResult:
        result = _fake_fold_runner(*args, **kwargs)
        frame = pd.read_csv(result.prediction_path)
        mutate_predictions(frame).to_csv(result.prediction_path, index=False)
        return result

    monkeypatch.setattr(cnn_experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
    with pytest.raises(ValueError, match=message):
        run_task1_experiment(
            _splits(),
            _label_map(),
            mode="full",
            fold_runner=invalid_artifact_runner,
            result_root=tmp_path / "runs",
        )


def test_full_runner_requires_five_fold_metrics_per_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete candidate cannot be compared with a five-fold candidate."""

    def wrong_preprocessing_runner(*args: object, **kwargs: object) -> Task1FoldResult:
        result = _fake_fold_runner(*args, **kwargs)
        if result.fold == 4:
            return Task1FoldResult(
                **{**result.__dict__, "preprocessing_id": "unexpected-preprocessing"}
            )
        return result

    monkeypatch.setattr(cnn_experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
    with pytest.raises(ValueError, match="exactly five folds"):
        run_task1_experiment(
            _splits(),
            _label_map(),
            mode="full",
            fold_runner=wrong_preprocessing_runner,
            result_root=tmp_path / "runs",
        )
