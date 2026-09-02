from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import fashion.task1.experiments as experiments
from fashion.task1.classical import Task1ClassicalFoldResult
from fashion.task1.experiments import (
    Task1ClassicalSelection,
    _rank_classical_candidates,
    _split_sha256,
    run_task1_classical_experiment,
)


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
    expected_candidate_ids = {
        "task1_gray_hog_ppc16_v1-knn-k5-distance",
        "task1_gray_hog_ppc16_v1-linear-svm-c1-normal",
        "task1_gray_hog_ppc10_v1-knn-k5-distance",
        "task1_gray_hog_ppc10_v1-linear-svm-c1-normal",
        "task1_gray_hog_ppc10_v1-knn-k3-uniform",
        "task1_gray_hog_ppc10_v1-knn-k3-distance",
        "task1_gray_hog_ppc10_v1-knn-k5-uniform",
        "task1_gray_hog_ppc10_v1-knn-k5-distance",
        "task1_gray_hog_ppc10_v1-knn-k11-uniform",
        "task1_gray_hog_ppc10_v1-knn-k11-distance",
        "task1_gray_hog_ppc10_v1-linear-svm-c0.1-balanced",
        "task1_gray_hog_ppc10_v1-linear-svm-c1-balanced",
        "task1_gray_hog_ppc10_v1-linear-svm-c10-normal",
        "task1_gray_hog_ppc10_v1-linear-svm-c10-balanced",
        "task1_gray_hog_ppc10_v1-linear-svm-c0.1-normal",
    }
    assert set(tuning["candidate_id"]) == expected_candidate_ids
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
    assert selection_payload["hog_id"] == "task1_gray_hog_ppc10_v1"
    assert selection_payload["knn_config_id"] == "knn-k3-distance"
    assert selection_payload["svm_config_id"] == "linear-svm-c10-balanced"


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


def test_classical_tune_excludes_nondefault_svm_convergence_failure_and_continues(
    tmp_path: Path,
) -> None:
    """One failed optional SVM must not stop later candidates or enter selection."""
    failed_config_id = "linear-svm-c0.1-normal"

    def runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id.endswith(f"-{failed_config_id}"):
            raise ConvergenceWarning("optional SVM did not converge")
        return result

    _classic_calls.clear()
    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=runner,
        result_root=tmp_path / "runs",
        evidence_root=tmp_path / "evidence",
    )

    assert failed_config_id in {call.model_id for call in _classic_calls}
    assert "linear-svm-c10-balanced" in {call.model_id for call in _classic_calls}
    assert not result.tuning["candidate_id"].str.endswith(f"-{failed_config_id}").any()
    assert result.selection is not None
    assert result.selection.svm_config_id != failed_config_id
    assert len(result.fold_results) == 13


def test_classical_tune_propagates_required_default_svm_convergence_failure(
    tmp_path: Path,
) -> None:
    """The default SVM is required to choose HOG, so its failure must stop tuning."""

    def runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id.endswith("-linear-svm-c1-normal"):
            raise ConvergenceWarning("required default SVM did not converge")
        return result

    evidence_root = tmp_path / "evidence"
    with pytest.raises(ConvergenceWarning, match="required default SVM"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="tune",
            fold_runner=runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    assert not (evidence_root / "classical_selection.json").exists()


def _final_selection() -> Task1ClassicalSelection:
    return Task1ClassicalSelection(
        hog_id="task1_gray_hog_ppc10_v1",
        knn_config_id="knn-k5-distance",
        svm_config_id="linear-svm-c1-balanced",
    )


def _assert_no_classical_aggregate_csv(evidence_root: Path) -> None:
    assert not {
        "classical_fold_metrics.csv",
        "classical_comparison.csv",
        "classical_oof_metrics.csv",
    }.intersection(path.name for path in evidence_root.glob("*.csv"))


def test_classical_final_runs_two_selected_candidates_on_exactly_five_folds(
    tmp_path: Path,
) -> None:
    """The final stage must only report the two frozen candidates across five folds."""
    _classic_calls.clear()
    evidence_root = tmp_path / "evidence"

    result = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="final",
        selection=_final_selection(),
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=evidence_root,
    )

    assert len(result.fold_results) == 10
    assert {item.fold for item in result.fold_results} == set(range(5))
    assert [(call.fold, call.model_id) for call in _classic_calls] == [
        *[(fold, "knn-k5-distance") for fold in range(5)],
        *[(fold, "linear-svm-c1-balanced") for fold in range(5)],
    ]
    assert len(result.comparison) == 2
    assert set(result.oof_predictions) == {
        "task1_gray_hog_ppc10_v1-knn-k5-distance",
        "task1_gray_hog_ppc10_v1-linear-svm-c1-balanced",
    }
    assert all(len(frame) == 124 for frame in result.per_class.values())
    assert {
        "classical_fold_metrics.csv",
        "classical_comparison.csv",
        "classical_oof_metrics.csv",
    }.issubset(path.name for path in evidence_root.glob("*.csv"))
    assert {
        "per_class_classical_task1_gray_hog_ppc10_v1-knn-k5-distance.csv",
        "per_class_classical_task1_gray_hog_ppc10_v1-linear-svm-c1-balanced.csv",
    }.issubset(path.name for path in evidence_root.glob("*.csv"))

    fold_metrics = pd.read_csv(evidence_root / "classical_fold_metrics.csv")
    assert list(fold_metrics.columns) == [
        "run_id",
        "fold",
        "candidate_id",
        "hog_id",
        "model_family",
        "macro_f1",
        "weighted_f1",
        "top1_accuracy",
        "top5_accuracy",
    ]
    assert len(fold_metrics) == 10
    assert set(fold_metrics["candidate_id"]) == set(result.oof_predictions)
    assert all(
        set(group["fold"]) == set(range(5))
        for _, group in fold_metrics.groupby("candidate_id")
    )

    comparison = pd.read_csv(evidence_root / "classical_comparison.csv")
    assert list(comparison.columns) == [
        "candidate_id",
        "hog_id",
        "model_family",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "top1_accuracy_mean",
        "top1_accuracy_std",
        "top5_accuracy_mean",
        "top5_accuracy_std",
    ]
    assert len(comparison) == 2
    assert set(comparison["candidate_id"]) == set(result.comparison["candidate_id"])
    assert comparison.filter(regex="_(mean|std)$").notna().all().all()

    oof_metrics = pd.read_csv(evidence_root / "classical_oof_metrics.csv")
    assert list(oof_metrics.columns) == [
        "candidate_id",
        "macro_f1",
        "weighted_f1",
        "top1_accuracy",
        "top5_accuracy",
    ]
    assert len(oof_metrics) == 2
    assert set(oof_metrics["candidate_id"]) == set(result.oof_predictions)
    assert oof_metrics.drop(columns="candidate_id").notna().all().all()

    per_class_columns = [
        "class_index",
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    for candidate_id in result.oof_predictions:
        per_class = pd.read_csv(
            evidence_root / f"per_class_classical_{candidate_id}.csv"
        )
        assert list(per_class.columns) == per_class_columns
        assert len(per_class) == 124
        assert per_class["class_index"].tolist() == list(range(124))


def test_classical_final_requires_a_selection_before_writing_aggregate_evidence(
    tmp_path: Path,
) -> None:
    """A missing frozen selection must fail before creating report aggregate files."""
    evidence_root = tmp_path / "evidence"

    with pytest.raises(ValueError, match="classical_selection.json"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            fold_runner=_fake_classical_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    _assert_no_classical_aggregate_csv(evidence_root)


def test_classical_final_rejects_unknown_selected_config_before_writing_aggregate_evidence(
    tmp_path: Path,
) -> None:
    """Final evidence cannot be built with an ID outside the frozen config grids."""
    evidence_root = tmp_path / "evidence"
    bad_selection = Task1ClassicalSelection(
        hog_id="task1_gray_hog_ppc10_v1",
        knn_config_id="knn-k999-distance",
        svm_config_id="linear-svm-c1-balanced",
    )

    with pytest.raises(ValueError, match="unknown KNN config"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            selection=bad_selection,
            fold_runner=_fake_classical_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    _assert_no_classical_aggregate_csv(evidence_root)


def test_classical_final_rejects_duplicate_fold_before_writing_aggregate_evidence(
    tmp_path: Path,
) -> None:
    """Each selected candidate needs exactly one result for every fold."""

    def duplicate_fold_runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id.endswith("linear-svm-c1-balanced") and result.fold == 4:
            return replace(result, fold=0)
        return result

    evidence_root = tmp_path / "evidence"
    with pytest.raises(ValueError, match="exactly five folds"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            selection=_final_selection(),
            fold_runner=duplicate_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    _assert_no_classical_aggregate_csv(evidence_root)


def test_classical_final_rejects_missing_oof_product_before_writing_aggregate_evidence(
    tmp_path: Path,
) -> None:
    """A missing fold-artifact ID must stop incomplete OOF evidence from being written."""

    def missing_oof_runner(*args: object, **kwargs: object) -> Task1ClassicalFoldResult:
        result = _fake_classical_fold_runner(*args, **kwargs)
        if result.candidate_id.endswith("linear-svm-c1-balanced") and result.fold == 4:
            pd.read_csv(result.prediction_path).iloc[1:].to_csv(result.prediction_path, index=False)
        return result

    evidence_root = tmp_path / "evidence"
    with pytest.raises(ValueError, match="expected development IDs"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            selection=_final_selection(),
            fold_runner=missing_oof_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    _assert_no_classical_aggregate_csv(evidence_root)


def test_classical_final_propagates_fold_failure_before_writing_aggregate_evidence(
    tmp_path: Path,
) -> None:
    """A physical fold failure must not leave a partial final report behind."""
    evidence_root = tmp_path / "evidence"

    def exploding_runner(*_: object, **__: object) -> Task1ClassicalFoldResult:
        raise RuntimeError("classic fold exploded")

    with pytest.raises(RuntimeError, match="classic fold exploded"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            selection=_final_selection(),
            fold_runner=exploding_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )

    _assert_no_classical_aggregate_csv(evidence_root)


def test_classical_final_loads_and_validates_tuned_selection_file(tmp_path: Path) -> None:
    """A final run without a supplied selection must verify the frozen tune outputs."""
    evidence_root = tmp_path / "evidence"
    tuned = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=evidence_root,
    )
    assert tuned.selection is not None
    _classic_calls.clear()

    final = run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="final",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=evidence_root,
    )

    assert final.selection == tuned.selection
    assert len(final.fold_results) == 10


@pytest.mark.parametrize("changed_key", ["split_sha256", "label_map_sha256", "tuning_sha256"])
def test_classical_final_rejects_stale_selection_file(
    tmp_path: Path, changed_key: str
) -> None:
    """Selection provenance must be sealed before a final comparison begins."""
    evidence_root = tmp_path / "evidence"
    run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=evidence_root,
    )
    selection_path = evidence_root / "classical_selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload[changed_key] = "0" * 64
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selection"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            fold_runner=_fake_classical_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
        )


def test_classical_final_rejects_selection_from_stale_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final evidence must use a selection produced by the current controller code."""
    implementation_sha256 = ["a" * 64]
    monkeypatch.setattr(
        experiments,
        "_classical_selection_implementation_sha256",
        lambda: implementation_sha256[0],
        raising=False,
    )
    evidence_root = tmp_path / "evidence"
    run_task1_classical_experiment(
        _splits(),
        _label_map(),
        stage="tune",
        fold_runner=_fake_classical_fold_runner,
        result_root=tmp_path / "runs",
        evidence_root=evidence_root,
    )
    implementation_sha256[0] = "b" * 64

    with pytest.raises(ValueError, match="implementation provenance is stale"):
        run_task1_classical_experiment(
            _splits(),
            _label_map(),
            stage="final",
            fold_runner=_fake_classical_fold_runner,
            result_root=tmp_path / "runs",
            evidence_root=evidence_root,
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
