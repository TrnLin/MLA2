from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    EXPERIMENT_REGISTRY_COLUMNS,
    I1_EXPERIMENT_ID,
    I1_REFERENCE_EXPERIMENT_ID,
    build_i1_class_balance_evidence,
)
from fashion.task2.experiments import load_experiment_config
from fashion.train.artifacts import canonical_sha256
from fashion.train.losses import (
    EFFECTIVE_NUMBER_BETA,
    EFFECTIVE_NUMBER_LOSS_ID,
    effective_number_class_weights,
)
from fashion.train.metrics import SEASON_LABELS


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _config_payload(experiment_id: str) -> dict:
    is_i1 = experiment_id == I1_EXPERIMENT_ID
    return {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "method": "deep",
        "model_family": "smallcnn",
        "stage": "g4_i1_class_balanced" if is_i1 else "g3_full_budget",
        "target": "season",
        "folds": [0, 1, 2, 3, 4],
        "seeds": [2753],
        "loss_id": EFFECTIVE_NUMBER_LOSS_ID if is_i1 else "cross_entropy",
        "data": {
            "image_size": [80, 60],
            "augmentation": "a0",
            "batch_size": 32,
            "validation_batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
        "optimisation": {
            "epochs": 30,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "effective_batch_size": 128,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 1.0,
            "patience": 5,
            "min_delta": 1e-4,
            "use_amp": True,
            "device": "auto",
        },
    }


def _class_metrics(*, i1: bool) -> dict[str, dict[str, float | int]]:
    scores = {
        "Fall": 0.685 if i1 else 0.690,
        "Spring": 0.760 if i1 else 0.745,
        "Summer": 0.783,
        "Winter": 0.729 if i1 else 0.730,
    }
    supports = {"Fall": 8928, "Spring": 1329, "Summer": 16235, "Winter": 6261}
    return {
        label: {
            "precision": score,
            "recall": score,
            "f1": score,
            "support": supports[label],
        }
        for label, score in scores.items()
    }


def _class_balance_payload(fold: int) -> dict:
    counts = (7000 + fold, 1000 + fold, 13000 + fold, 5000 + fold)
    weights = effective_number_class_weights(counts, beta=EFFECTIVE_NUMBER_BETA)
    return {
        "schema_version": "1.0.0",
        "labels": list(SEASON_LABELS),
        "class_counts": dict(zip(SEASON_LABELS, counts, strict=True)),
        "class_weights": dict(zip(SEASON_LABELS, weights, strict=True)),
        "beta": EFFECTIVE_NUMBER_BETA,
        "training_product_count": sum(counts),
        "training_id_sha256": f"{fold + 1:064x}",
        "loss_id": EFFECTIVE_NUMBER_LOSS_ID,
    }


def _write_experiment(
    root: Path,
    experiment_id: str,
) -> tuple[Path, Path, list[Path]]:
    is_i1 = experiment_id == I1_EXPERIMENT_ID
    config_payload = _config_payload(experiment_id)
    config_path = _write_json(root / "configs/task2" / f"{experiment_id}.json", config_payload)
    canonical_config = load_experiment_config(config_path).to_dict()
    config_sha256 = canonical_sha256(canonical_config)
    per_class = _class_metrics(i1=is_i1)
    pooled_macro_f1 = sum(float(row["f1"]) for row in per_class.values()) / len(per_class)
    fold_offsets = (-0.004, -0.002, 0.0, 0.002, 0.004)
    fold_scores = [pooled_macro_f1 + offset for offset in fold_offsets]
    run_ids = [f"{experiment_id}-f{fold}-synthetic" for fold in range(5)]
    histories: list[Path] = []
    registry_rows = []
    fold_rows = []
    for fold, (run_id, score) in enumerate(zip(run_ids, fold_scores, strict=True)):
        epoch_history = [
            {
                "epoch": epoch,
                "learning_rate": 1e-3,
                "train_loss": 1.1 - 0.08 * epoch,
                "validation_loss": 1.0 - 0.05 * epoch,
                "validation_accuracy": score + 0.01 - 0.01 * (3 - epoch),
                "validation_macro_f1": score - 0.01 * (3 - epoch),
            }
            for epoch in range(1, 4)
        ]
        history_payload = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "fold": fold,
            "seed": 2753,
            "config": canonical_config,
            "epoch_history": epoch_history,
        }
        if is_i1:
            history_payload["class_balance"] = _class_balance_payload(fold)
        history_path = _write_json(
            root / "tmp/task2/runs" / run_id / "history.json",
            history_payload,
        )
        histories.append(history_path)
        row = {column: "" for column in EXPERIMENT_REGISTRY_COLUMNS}
        row.update(
            {
                "run_id": run_id,
                "stage": config_payload["stage"],
                "experiment_id": experiment_id,
                "model_family": "smallcnn",
                "benchmark_only": False,
                "final_eligible": True,
                "scratch": True,
                "fold": fold,
                "seed": 2753,
                "git_commit": "a" * 40,
                "git_dirty": False,
                "config_sha256": config_sha256,
                "split_sha256": "b" * 64,
                "label_map_sha256": "c" * 64,
                "implementation_sha256": ("e" if is_i1 else "d") * 64,
                "transform_id": "a0-synthetic",
                "loss_id": config_payload["loss_id"],
                "epochs_requested": 30,
                "epochs_completed": 3,
                "best_epoch": 3,
                "primary_metric_name": "macro_f1",
                "primary_metric_value": score,
                "runtime_seconds": 60.0 + fold,
                "peak_vram_mb": 192.0,
                "parameter_count": 1_174_244,
                "checkpoint_path": f"tmp/task2/checkpoints/{run_id}.pt",
                "checkpoint_sha256": "f" * 64,
                "prediction_path": f"tmp/task2/runs/{run_id}/oof.csv",
                "prediction_sha256": "1" * 64,
                "history_path": history_path.relative_to(root).as_posix(),
                "history_sha256": compute_sha256(history_path),
                "status": "completed",
            }
        )
        registry_rows.append(row)
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "fold": fold,
                "seed": 2753,
                "source": "run",
                "macro_f1": score,
            }
        )

    evidence = root / "results/evidence/task2" / experiment_id.replace("-", "_")
    pooled_path = _write_json(
        evidence / "pooled_metrics.json",
        {"macro_f1": pooled_macro_f1, "per_class": per_class},
    )
    fold_summary_path = evidence / "fold_summary.csv"
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": sum(fold_scores) / len(fold_scores),
                "fold_sd": pd.Series(fold_scores).std(ddof=1),
                "fold_min": min(fold_scores),
                "fold_max": max(fold_scores),
                "pooled_value": pooled_macro_f1,
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    fold_metrics_path = evidence / "fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)
    registry_path = evidence / "registry_snapshot.csv"
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)
    artifacts = {
        "pooled_metrics": pooled_path,
        "fold_summary": fold_summary_path,
        "fold_metrics": fold_metrics_path,
        "registry_snapshot": registry_path,
    }
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "seed": 2753,
        "folds": [0, 1, 2, 3, 4],
        "run_ids": run_ids,
        "coverage": {
            "row_count": 32753,
            "unique_id_count": 32753,
            "expected_row_count": 32753,
            "protected_id_count": 0,
            "id_set_sha256": "2" * 64,
            "labels": list(SEASON_LABELS),
        },
        "pooled_macro_f1": pooled_macro_f1,
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": compute_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    manifest_path = _write_json(evidence / "manifest.json", manifest)
    return manifest_path, config_path, histories


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, list[Path]]:
    reference_manifest, reference_config, _ = _write_experiment(root, I1_REFERENCE_EXPERIMENT_ID)
    i1_manifest, i1_config, i1_histories = _write_experiment(root, I1_EXPERIMENT_ID)
    return (
        reference_manifest,
        i1_manifest,
        reference_config,
        i1_config,
        i1_histories,
    )


def _build(root: Path, **kwargs):
    reference_manifest, i1_manifest, reference_config, i1_config, _ = _inputs(root)
    return build_i1_class_balance_evidence(
        reference_manifest_path=reference_manifest,
        i1_manifest_path=i1_manifest,
        reference_config_path=reference_config,
        i1_config_path=i1_config,
        project_root=root,
        evidence_directory=root / "results/evidence/task2/i1_class_balance",
        figure_directory=root / "results/figures/task2",
        **kwargs,
    )


def test_i1_evidence_applies_frozen_rule_and_hashes_outputs(tmp_path: Path) -> None:
    manifest = _build(tmp_path)

    assert manifest["gate"] == "G4-I1"
    assert manifest["keep_i1"] is True
    assert manifest["selected_experiment_id"] == I1_EXPERIMENT_ID
    assert manifest["ultimate_winner_frozen"] is False
    assert set(manifest["artifacts"]) == {
        "comparison",
        "paired_fold_metrics",
        "per_class_comparison",
        "class_weights_by_fold",
        "registry_snapshot",
        "learning_curves_by_fold",
        "learning_curve_summary",
        "decision",
        "learning_curves",
        "per_class_f1_delta",
    }
    for artifact in manifest["artifacts"].values():
        path = tmp_path / "results" / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size > 0
        assert compute_sha256(path) == artifact["sha256"]

    evidence = tmp_path / "results/evidence/task2/i1_class_balance"
    comparison = pd.read_csv(evidence / "comparison.csv")
    per_class = pd.read_csv(evidence / "per_class_comparison.csv")
    weights = pd.read_csv(evidence / "class_weights_by_fold.csv")
    curves = pd.read_csv(evidence / "learning_curve_summary.csv")
    decision = json.loads((evidence / "decision.json").read_text(encoding="utf-8"))
    assert len(comparison) == 2
    assert len(per_class) == len(SEASON_LABELS)
    assert len(weights) == 5 * len(SEASON_LABELS)
    assert curves["fold_count"].eq(5).all()
    assert decision["criteria"]["spring_f1_gain"]["passed"] is True
    assert decision["loss_values_comparable_to_reference"] is False


def test_i1_evidence_rejects_weights_that_disagree_with_counts(tmp_path: Path) -> None:
    (
        reference_manifest,
        i1_manifest,
        reference_config,
        i1_config,
        histories,
    ) = _inputs(tmp_path)
    history_path = histories[0]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["class_balance"]["class_weights"]["Spring"] += 0.1
    _write_json(history_path, history)

    manifest = json.loads(i1_manifest.read_text(encoding="utf-8"))
    registry_path = tmp_path / manifest["artifacts"]["registry_snapshot"]["path"]
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    selected = registry["history_path"].eq(history_path.relative_to(tmp_path).as_posix())
    registry.loc[selected, "history_sha256"] = compute_sha256(history_path)
    registry.to_csv(registry_path, index=False)
    manifest["artifacts"]["registry_snapshot"]["sha256"] = compute_sha256(registry_path)
    _write_json(i1_manifest, manifest)

    with pytest.raises(ValueError, match="weights do not match their counts"):
        build_i1_class_balance_evidence(
            reference_manifest_path=reference_manifest,
            i1_manifest_path=i1_manifest,
            reference_config_path=reference_config,
            i1_config_path=i1_config,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/i1_class_balance",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_i1_evidence_rejects_protocol_change_beyond_loss(tmp_path: Path) -> None:
    reference_manifest, i1_manifest, reference_config, i1_config, _ = _inputs(tmp_path)
    config = json.loads(i1_config.read_text(encoding="utf-8"))
    config["data"]["batch_size"] = 16
    _write_json(i1_config, config)

    with pytest.raises(ValueError, match="frozen protocol"):
        build_i1_class_balance_evidence(
            reference_manifest_path=reference_manifest,
            i1_manifest_path=i1_manifest,
            reference_config_path=reference_config,
            i1_config_path=i1_config,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/i1_class_balance",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_i1_evidence_rejects_invalid_decision_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        _build(tmp_path, spring_minimum_gain=-0.01)
