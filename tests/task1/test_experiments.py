from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import fashion.task1.experiments as experiments
from fashion.task1.classical import Task1ClassicalFoldResult
from fashion.task1.experiments import (
    _rank_classical_candidates,
    _split_sha256,
    run_task1_classical_experiment,
    run_task1_experiment,
    write_task1_comparison_figure,
    write_task1_confusion_figure,
)
from fashion.task1.training import Task1FoldResult


@dataclass(frozen=True)
class _FoldCall:
    fold: int
    transform_id: str
    stage: str


@dataclass(frozen=True)
class _ClassicalFoldCall:
    fold: int
    hog_id: str
    model_id: str
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
    _fake_fold_runner.calls.append(
        _FoldCall(validation_fold, preprocessing.preprocessing_id, config.stage)
    )
    prediction_path = Path(result_root) / f"{preprocessing.preprocessing_id}-{validation_fold}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame(splits.loc[splits["cv_fold"].eq(validation_fold), "id"].tolist()).to_csv(
        prediction_path, index=False
    )
    return Task1FoldResult(
        run_id=f"{preprocessing.preprocessing_id}-{validation_fold}",
        fold=validation_fold,
        preprocessing_id=preprocessing.preprocessing_id,
        status="completed",
        metrics={"macro_f1": 0.1 + validation_fold / 100, "top1_accuracy": 0.2},
        checkpoint_path=prediction_path.with_suffix(".pt"),
        history_path=prediction_path.with_name("history.csv"),
        prediction_path=prediction_path,
    )


_fake_fold_runner.calls = []  # type: ignore[attr-defined]


_classic_calls: list[_ClassicalFoldCall] = []


def _fake_classical_fold_runner(
    splits: pd.DataFrame,
    _: dict[str, object],
    *,
    validation_fold: int,
    hog_spec: Any,
    model_config: Any,
    run_config: Any,
    result_root: str | Path,
    **__: object,
) -> Task1ClassicalFoldResult:
    """Return fixed fold-0 metrics so controller choices are independently known."""
    model_id = model_config.config_id
    _classic_calls.append(
        _ClassicalFoldCall(validation_fold, hog_spec.hog_id, model_id, run_config.stage)
    )
    prediction_path = Path(result_root) / f"{hog_spec.hog_id}-{model_id}-{validation_fold}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame(splits.loc[splits["cv_fold"].eq(validation_fold), "id"].tolist()).to_csv(
        prediction_path, index=False
    )
    # Fine HOG wins the shared default comparison.  Among fine HOG model settings,
    # the two KNN k=3 options are an exact metric tie, so canonical ID chooses distance.
    default_macro = 0.60 if hog_spec.hog_id == "task1_gray_hog_ppc10_v1" else 0.40
    if model_id in {"knn-k5-distance", "linear-svm-c1-normal"}:
        metrics = {
            "macro_f1": default_macro,
            "weighted_f1": default_macro,
            "top1_accuracy": default_macro,
            "top5_accuracy": default_macro,
        }
    elif model_id in {"knn-k3-uniform", "knn-k3-distance"}:
        metrics = {
            "macro_f1": 0.80,
            "weighted_f1": 0.70,
            "top1_accuracy": 0.60,
            "top5_accuracy": 0.90,
        }
    elif model_id == "linear-svm-c10-balanced":
        metrics = {
            "macro_f1": 0.79,
            "weighted_f1": 0.82,
            "top1_accuracy": 0.83,
            "top5_accuracy": 0.90,
        }
    else:
        metrics = {
            "macro_f1": 0.50,
            "weighted_f1": 0.50,
            "top1_accuracy": 0.50,
            "top5_accuracy": 0.50,
        }
    return Task1ClassicalFoldResult(
        run_id=f"{hog_spec.hog_id}-{model_id}-{validation_fold}",
        fold=validation_fold,
        candidate_id=f"{hog_spec.hog_id}-{model_id}",
        hog_id=hog_spec.hog_id,
        model_family="fake-classical",
        status="completed",
        metrics=metrics,
        model_path=prediction_path.with_suffix(".pkl"),
        prediction_path=prediction_path,
    )


def test_classical_smoke_runs_default_knn_and_svm_on_coarse_hog(tmp_path: Path) -> None:
    """A changed smoke schedule must not launch a different classic candidate."""
    _classic_calls.clear()

    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="smoke",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=tmp_path / "evidence",
    )

    assert [(call.fold, call.hog_id, call.model_id) for call in _classic_calls] == [
        (0, "task1_gray_hog_ppc16_v1", "knn-k5-distance"),
        (0, "task1_gray_hog_ppc16_v1", "linear-svm-c1-normal"),
    ]
    assert result.stage == "smoke"
    assert result.selection is None


def test_classical_tune_selects_shared_hog_then_one_model_per_family(tmp_path: Path) -> None:
    """Wrong HOG selection, deduplication, or ranking must change the frozen selection."""
    _classic_calls.clear()

    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=tmp_path / "evidence",
    )

    assert len(_classic_calls) == 14
    assert [(call.fold, call.hog_id, call.model_id) for call in _classic_calls[:4]] == [
        (0, "task1_gray_hog_ppc16_v1", "knn-k5-distance"),
        (0, "task1_gray_hog_ppc16_v1", "linear-svm-c1-normal"),
        (0, "task1_gray_hog_ppc10_v1", "knn-k5-distance"),
        (0, "task1_gray_hog_ppc10_v1", "linear-svm-c1-normal"),
    ]
    assert len({(call.fold, call.hog_id, call.model_id) for call in _classic_calls}) == 14
    assert len({item.candidate_id for item in result.fold_results}) == 14
    assert result.selection is not None
    assert result.selection.hog_id == "task1_gray_hog_ppc10_v1"
    assert result.selection.knn_config_id == "knn-k3-distance"
    assert result.selection.svm_config_id == "linear-svm-c10-balanced"
    tuning_path = tmp_path / "evidence" / "classical_tuning.csv"
    selection_path = tmp_path / "evidence" / "classical_selection.json"
    assert tuning_path.is_file()
    assert selection_path.is_file()
    tuning = pd.read_csv(tuning_path)
    assert len(tuning) == 14
    assert tuning["candidate_id"].nunique() == 14
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_keys = {"hog_id", "knn_config_id", "svm_config_id"}
    provenance_keys = {
        "tuning_sha256",
        "split_sha256",
        "label_map_sha256",
        "implementation_sha256",
    }
    assert set(selection_payload).difference(provenance_keys) == selection_keys
    assert set(selection_payload).difference(selection_keys) == provenance_keys


def test_classical_tune_uses_coarse_hog_for_an_exact_default_macro_f1_tie(
    tmp_path: Path,
) -> None:
    """An exact default macro-F1 tie must use the frozen coarse-HOG tie-break."""

    def tied_default_runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id.endswith(("-knn-k5-distance", "-linear-svm-c1-normal")):
            return replace(
                result,
                metrics={
                    "macro_f1": 0.60,
                    "weighted_f1": 0.60,
                    "top1_accuracy": 0.60,
                    "top5_accuracy": 0.60,
                },
            )
        return result

    _classic_calls.clear()
    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=tied_default_runner,
        result_root=tmp_path / "runs",
        evidence_root=tmp_path / "evidence",
    )

    assert result.selection is not None
    assert result.selection.hog_id == "task1_gray_hog_ppc16_v1"


def test_classical_tune_ranks_model_configs_only_on_the_selected_hog(tmp_path: Path) -> None:
    """A strong default from the rejected HOG must not select its model configuration."""

    def runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id == "task1_gray_hog_ppc16_v1-knn-k5-distance":
            return replace(
                result,
                metrics={
                    "macro_f1": 0.95,
                    "weighted_f1": 0.95,
                    "top1_accuracy": 0.95,
                    "top5_accuracy": 0.95,
                },
            )
        if result.candidate_id == "task1_gray_hog_ppc16_v1-linear-svm-c1-normal":
            return replace(
                result,
                metrics={
                    "macro_f1": 0.10,
                    "weighted_f1": 0.10,
                    "top1_accuracy": 0.10,
                    "top5_accuracy": 0.10,
                },
            )
        return result

    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=runner,
        result_root=tmp_path / "runs",
        evidence_root=tmp_path / "evidence",
    )

    assert result.selection is not None
    assert result.selection.hog_id == "task1_gray_hog_ppc10_v1"
    assert result.selection.knn_config_id == "knn-k3-distance"


def test_classical_final_is_guarded_until_final_evidence_exists(tmp_path: Path) -> None:
    """A final request must not silently run the fold-0 tuning schedule instead."""
    with pytest.raises(ValueError, match="not implemented until Task 6"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            fold_runner=_fake_classical_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=tmp_path / "evidence",
        )


def test_classical_ranking_uses_every_metric_then_canonical_candidate_id() -> None:
    """Changing a ranking tie-breaker must pick the candidate with its better value."""
    frame = pd.DataFrame(
        {
            "candidate_id": ["z", "y", "x", "w", "v", "a"],
            "macro_f1": [0.7, 0.8, 0.7, 0.7, 0.7, 0.7],
            "weighted_f1": [0.7, 0.1, 0.8, 0.7, 0.7, 0.7],
            "top1_accuracy": [0.7, 0.1, 0.1, 0.8, 0.7, 0.7],
            "top5_accuracy": [0.7, 0.1, 0.1, 0.1, 0.8, 0.7],
        }
    )

    assert _rank_classical_candidates(frame)["candidate_id"].tolist() == [
        "y",
        "x",
        "w",
        "v",
        "a",
        "z",
    ]


def test_classical_split_provenance_hash_handles_empty_split_cells() -> None:
    """An empty non-ID split cell must not stop a completed tune from being sealed."""
    splits = _splits()
    splits.loc[0, "cv_fold"] = np.nan

    assert len(_split_sha256(splits)) == 64


def test_runner_uses_one_control_fold_for_smoke_and_ten_fixed_full_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the run schedule must fail this test before physical runs are launched."""
    monkeypatch.setattr(experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
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
    monkeypatch.setattr(experiments, "TASK1_EVIDENCE_DIR", evidence_root)
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

    monkeypatch.setattr(experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
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

    monkeypatch.setattr(experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
    with pytest.raises(ValueError, match="exactly five folds"):
        run_task1_experiment(
            _splits(),
            _label_map(),
            mode="full",
            fold_runner=wrong_preprocessing_runner,
            result_root=tmp_path / "runs",
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


def test_confusion_figure_writes_one_png(tmp_path: Path) -> None:
    predictions = pd.DataFrame({"true_index": [0, 1], "predicted_index": [0, 1]})
    output = write_task1_confusion_figure(
        predictions, [f"class-{index}" for index in range(124)], output=tmp_path / "confusion.png"
    )
    assert output.is_file()
    assert output.stat().st_size > 0
