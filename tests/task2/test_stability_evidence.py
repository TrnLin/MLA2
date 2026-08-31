from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import EXPERIMENT_REGISTRY_COLUMNS
from fashion.task2.experiments import load_experiment_config
from fashion.task2.multitask import load_i2_config
from fashion.task2.stability import (
    C2_PRIMARY_EXPERIMENT_ID,
    C2_STABILITY_EXPERIMENT_ID,
    I2_PRIMARY_EXPERIMENT_ID,
    I2_STABILITY_EXPERIMENT_ID,
    load_stability_i2_config,
)
from fashion.task2.stability_evidence import build_seed_stability_evidence
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS

CASES = {
    C2_PRIMARY_EXPERIMENT_ID: {
        "filename": "g3_c2_t0_resnet18.json",
        "candidate": "C2",
        "seed": 2753,
        "stage": "g3_full_budget",
        "family": "resnet18_small_stem",
        "loss_id": "cross_entropy",
        "score": 0.73,
    },
    C2_STABILITY_EXPERIMENT_ID: {
        "filename": "g5_c2_t0_resnet18_seed_2026.json",
        "candidate": "C2",
        "seed": 2026,
        "stage": "g5_seed_stability",
        "family": "resnet18_small_stem",
        "loss_id": "cross_entropy",
        "score": 0.72,
    },
    I2_PRIMARY_EXPERIMENT_ID: {
        "filename": "g4_i2_article_type_lambda_0_3_c1.json",
        "candidate": "I2",
        "seed": 2753,
        "stage": "g4_i2_multitask",
        "family": "smallcnn",
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_3",
        "score": 0.75,
    },
    I2_STABILITY_EXPERIMENT_ID: {
        "filename": "g5_i2_article_type_lambda_0_3_c1_seed_2026.json",
        "candidate": "I2",
        "seed": 2026,
        "stage": "g5_seed_stability",
        "family": "smallcnn",
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_3",
        "score": 0.74,
    },
}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _copy_config(root: Path, experiment_id: str) -> Path:
    filename = CASES[experiment_id]["filename"]
    source = ROOT / "configs/task2" / filename
    target = root / "configs/task2" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _load_config(path: Path, experiment_id: str):
    if experiment_id in {
        C2_PRIMARY_EXPERIMENT_ID,
        C2_STABILITY_EXPERIMENT_ID,
    }:
        return load_experiment_config(path)
    if experiment_id == I2_PRIMARY_EXPERIMENT_ID:
        return load_i2_config(path)
    return load_stability_i2_config(path)


def _write_pack(
    root: Path,
    experiment_id: str,
    *,
    stability_i2_score: float | None = None,
) -> tuple[Path, Path, list[Path]]:
    case = CASES[experiment_id]
    config_path = _copy_config(root, experiment_id)
    config = _load_config(config_path, experiment_id)
    score = float(
        stability_i2_score
        if experiment_id == I2_STABILITY_EXPERIMENT_ID and stability_i2_score is not None
        else case["score"]
    )
    evidence = root / "results/evidence/task2" / experiment_id.replace("-", "_")
    run_ids = [f"{experiment_id}-f{fold}-s{case['seed']}" for fold in range(5)]
    fold_scores = [score + offset for offset in (-0.004, -0.002, 0, 0.002, 0.004)]
    registry_rows = []
    history_paths = []
    fold_rows = []
    for fold, (run_id, fold_score) in enumerate(zip(run_ids, fold_scores, strict=True)):
        history_path = root / "tmp/task2/runs" / run_id / "history.json"
        epochs = []
        for epoch in range(1, 4):
            epoch_row = {
                "epoch": epoch,
                "learning_rate": 1e-3,
                "train_loss": 1.2 - epoch * 0.1 - fold * 0.002,
                "validation_loss": 1.1 - epoch * 0.07 + fold * 0.002,
                "validation_accuracy": fold_score + epoch * 0.01,
                "validation_macro_f1": fold_score + epoch * 0.008,
            }
            if case["candidate"] == "I2":
                epoch_row["train_total_loss"] = epoch_row["train_loss"]
                epoch_row["validation_total_loss"] = epoch_row["validation_loss"]
            epochs.append(epoch_row)
        history = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "fold": fold,
            "seed": case["seed"],
            "config": config.to_dict(),
            "model_boundary": {
                "class": (
                    "SeasonArticleTypeMultiTaskModel"
                    if case["candidate"] == "I2"
                    else "ScratchSmallStemResNet18"
                ),
                "benchmark_only": False,
                "final_eligible": True,
                "training_origin": "scratch",
                "weights": None,
            },
            "epoch_history": epochs,
        }
        _write_json(history_path, history)
        history_paths.append(history_path)
        implementation_hash = "c" * 64 if case["candidate"] == "C2" else "d" * 64
        registry_rows.append(
            {
                "run_id": run_id,
                "stage": case["stage"],
                "experiment_id": experiment_id,
                "model_family": case["family"],
                "benchmark_only": False,
                "final_eligible": True,
                "scratch": True,
                "fold": fold,
                "seed": case["seed"],
                "git_commit": "e" * 40,
                "git_dirty": False,
                "config_sha256": canonical_sha256(config.to_dict()),
                "split_sha256": "a" * 64,
                "label_map_sha256": "b" * 64,
                "implementation_sha256": implementation_hash,
                "transform_id": "resize_pad_80x60_a0",
                "loss_id": case["loss_id"],
                "epochs_requested": 30,
                "epochs_completed": 3,
                "best_epoch": 3,
                "primary_metric_name": "macro_f1",
                "primary_metric_value": fold_score,
                "runtime_seconds": 60 + fold,
                "peak_vram_mb": 512,
                "parameter_count": (1_174_244 if case["candidate"] == "I2" else 11_170_884),
                "checkpoint_path": f"tmp/task2/checkpoints/{run_id}.pt",
                "checkpoint_sha256": "f" * 64,
                "prediction_path": f"tmp/task2/runs/{run_id}/oof.csv",
                "prediction_sha256": "1" * 64,
                "history_path": history_path.relative_to(root).as_posix(),
                "history_sha256": compute_sha256(history_path),
                "status": "completed",
            }
        )
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "fold": fold,
                "seed": case["seed"],
                "source": "cache",
                "macro_f1": fold_score,
                "accuracy": fold_score + 0.02,
            }
        )

    evidence.mkdir(parents=True, exist_ok=True)
    registry_path = evidence / "registry_snapshot.csv"
    pd.DataFrame(registry_rows).loc[:, EXPERIMENT_REGISTRY_COLUMNS].to_csv(
        registry_path,
        index=False,
    )
    fold_metrics_path = evidence / "fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)
    fold_summary_path = evidence / "fold_summary.csv"
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": pd.Series(fold_scores).mean(),
                "fold_sd": pd.Series(fold_scores).std(ddof=1),
                "pooled_value": score,
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    pooled_path = _write_json(
        evidence / "pooled_metrics.json",
        {
            "n_samples": 20,
            "accuracy": score + 0.02,
            "balanced_accuracy": score + 0.01,
            "macro_f1": score,
            "per_class": {
                label: {
                    "precision": score + offset,
                    "recall": score + offset,
                    "f1": score + offset,
                    "support": 5,
                }
                for label, offset in zip(
                    SEASON_LABELS,
                    (-0.02, 0.01, 0.02, -0.01),
                    strict=True,
                )
            },
        },
    )
    artifacts = {
        "registry_snapshot": registry_path,
        "fold_metrics": fold_metrics_path,
        "fold_summary": fold_summary_path,
        "pooled_metrics": pooled_path,
    }
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "seed": case["seed"],
        "folds": list(range(5)),
        "run_ids": run_ids,
        "coverage": {
            "row_count": 20,
            "unique_id_count": 20,
            "expected_row_count": 20,
            "id_set_sha256": "9" * 64,
            "labels": list(SEASON_LABELS),
            "protected_id_count": 0,
        },
        "pooled_macro_f1": score,
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": compute_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    return (
        _write_json(evidence / "manifest.json", manifest),
        config_path,
        history_paths,
    )


def _inputs(
    root: Path,
    *,
    stability_i2_score: float | None = None,
):
    packs = [
        _write_pack(
            root,
            experiment_id,
            stability_i2_score=stability_i2_score,
        )
        for experiment_id in CASES
    ]
    return (
        [pack[0] for pack in packs],
        [pack[1] for pack in packs],
        [pack[2] for pack in packs],
    )


def _build(
    root: Path,
    *,
    stability_i2_score: float | None = None,
):
    manifests, configs, histories = _inputs(
        root,
        stability_i2_score=stability_i2_score,
    )
    result = build_seed_stability_evidence(
        experiment_manifest_paths=manifests,
        experiment_config_paths=configs,
        project_root=root,
        expected_row_count=20,
        evidence_directory=root / "results/evidence/task2/seed_stability",
        figure_directory=root / "results/figures/task2",
    )
    return result, manifests, histories


def test_stability_evidence_preserves_positive_ordering_without_freezing(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _build(tmp_path)
    evidence = tmp_path / "results/evidence/task2/seed_stability"
    decision = json.loads((evidence / "decision.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(evidence / "seed_stability.csv")
    paired = pd.read_csv(evidence / "paired_fold_metrics.csv")
    learning = pd.read_csv(evidence / "learning_curve_summary.csv")

    assert manifest["ordering_stable"] is True
    assert manifest["candidate_selection_affected"] is False
    assert manifest["ultimate_winner_frozen"] is False
    assert decision["i2_minus_c2_macro_f1_seed_2753"] == pytest.approx(0.02)
    assert decision["i2_minus_c2_macro_f1_seed_2026"] == pytest.approx(0.02)
    assert decision["current_candidate"] == "I2"
    assert len(summary) == 4
    assert len(paired) == 5
    assert learning["fold_count"].eq(5).all()
    assert set(learning["seed"]) == {2753, 2026}
    for name in (
        "seed_stability_learning_curves.png",
        "seed_stability_comparison.png",
    ):
        figure = tmp_path / "results/figures/task2" / name
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert figure.stat().st_size > 10_000


def test_stability_evidence_records_reversal_without_post_hoc_selection(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _build(tmp_path, stability_i2_score=0.70)
    decision = json.loads(
        (tmp_path / "results/evidence/task2/seed_stability/decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["ordering_stable"] is False
    assert manifest["candidate_selection_affected"] is True
    assert decision["candidate_status"] == "ordering_reversed_unresolved"
    assert decision["current_candidate"] is None
    assert decision["ultimate_winner_frozen"] is False


def test_stability_evidence_rejects_candidate_implementation_drift(
    tmp_path: Path,
) -> None:
    manifests, configs, _ = _inputs(tmp_path)
    target_manifest_path = manifests[-1]
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    registry_declaration = target_manifest["artifacts"]["registry_snapshot"]
    registry_path = tmp_path / registry_declaration["path"]
    registry = pd.read_csv(registry_path)
    registry["implementation_sha256"] = "0" * 64
    registry.to_csv(registry_path, index=False)
    registry_declaration["sha256"] = compute_sha256(registry_path)
    _write_json(target_manifest_path, target_manifest)

    with pytest.raises(ValueError, match="I2 changed implementation_sha256"):
        build_seed_stability_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            project_root=tmp_path,
            expected_row_count=20,
            evidence_directory=(tmp_path / "results/evidence/task2/seed_stability"),
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_stability_evidence_rejects_tampered_history(tmp_path: Path) -> None:
    manifests, configs, histories = _inputs(tmp_path)
    target = histories[-1][0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["epoch_history"][0]["validation_macro_f1"] = 0.01
    _write_json(target, payload)

    with pytest.raises(ValueError, match="history hash mismatch"):
        build_seed_stability_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            project_root=tmp_path,
            expected_row_count=20,
            evidence_directory=(tmp_path / "results/evidence/task2/seed_stability"),
            figure_directory=tmp_path / "results/figures/task2",
        )
