from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import build_g3_full_budget_evidence
from fashion.task2.experiments import load_experiment_config
from fashion.train.artifacts import canonical_sha256

G3_CASES = {
    "g3-c1-t1-smallcnn": {
        "family": "C1",
        "tuning_id": "T1",
        "model_family": "smallcnn",
        "screen_id": "g2-t1-c1-smallcnn",
        "learning_rate": 1e-3,
        "screen_score": 0.708,
        "parameter_count": 1_174_244,
        "epochs": (18, 21, 25, 30, 20),
    },
    "g3-c2-t0-resnet18": {
        "family": "C2",
        "tuning_id": "T0",
        "model_family": "resnet18_small_stem",
        "screen_id": "g1-c2-resnet18",
        "learning_rate": 3e-4,
        "screen_score": 0.707,
        "parameter_count": 11_170_884,
        "epochs": (19, 22, 24, 27, 30),
    },
}


def _config_payload(
    experiment_id: str,
    *,
    stage: str,
    epochs: int,
    patience: int,
    batch_size: int = 32,
) -> dict:
    spec = G3_CASES[experiment_id]
    return {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "method": "deep",
        "model_family": spec["model_family"],
        "stage": stage,
        "folds": [0, 1, 2, 3, 4],
        "seeds": [2753],
        "loss_id": "cross_entropy",
        "data": {
            "image_size": [80, 60],
            "augmentation": "a0",
            "batch_size": batch_size,
            "validation_batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
        "optimisation": {
            "epochs": epochs,
            "learning_rate": spec["learning_rate"],
            "weight_decay": 1e-4,
            "effective_batch_size": 128,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 1.0,
            "patience": patience,
            "min_delta": 1e-4,
            "use_amp": True,
            "device": "auto",
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_config(
    root: Path,
    experiment_id: str,
    *,
    screen: bool,
    batch_size: int = 32,
) -> Path:
    spec = G3_CASES[experiment_id]
    declared_id = spec["screen_id"] if screen else experiment_id
    payload = _config_payload(
        experiment_id,
        stage="g2_compact_tuning" if screen else "g3_full_budget",
        epochs=8 if screen else 30,
        patience=8 if screen else 5,
        batch_size=batch_size,
    )
    payload["experiment_id"] = declared_id
    path = root / "configs/task2" / f"{declared_id}.json"
    return _write_json(path, payload)


def _write_tuning_manifest(
    root: Path,
    screen_configs: dict[str, Path],
) -> Path:
    evidence = root / "results/evidence/task2/g2_compact_tuning"
    decision_path = _write_json(evidence / "decision.json", {"gate": "G2-T"})
    leaderboard_path = evidence / "leaderboard.csv"
    pd.DataFrame(
        [
            {
                "family": spec["family"],
                "tuning_id": spec["tuning_id"],
                "experiment_id": spec["screen_id"],
                "pooled_macro_f1": spec["screen_score"],
                "selected": True,
            }
            for spec in G3_CASES.values()
        ]
    ).to_csv(leaderboard_path, index=False)
    families = {
        spec["family"]: {
            "selected_experiment_id": spec["screen_id"],
            "selected_tuning_id": spec["tuning_id"],
            "selected_learning_rate": spec["learning_rate"],
            "selected_weight_decay": 1e-4,
            "selected_macro_f1": spec["screen_score"],
        }
        for spec in G3_CASES.values()
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G2-T",
        "decision_status": "closed",
        "families": families,
        "input_configs": {
            G3_CASES[experiment_id]["screen_id"]: {
                "path": path.relative_to(root).as_posix(),
                "sha256": compute_sha256(path),
            }
            for experiment_id, path in screen_configs.items()
        },
        "artifacts": {
            "decision": {
                "path": decision_path.relative_to(root).as_posix(),
                "sha256": compute_sha256(decision_path),
            },
            "leaderboard": {
                "path": leaderboard_path.relative_to(root).as_posix(),
                "sha256": compute_sha256(leaderboard_path),
            },
        },
    }
    return _write_json(evidence / "manifest.json", manifest)


def _write_experiment_evidence(
    root: Path,
    experiment_id: str,
    config_path: Path,
    *,
    pooled_macro_f1: float,
) -> tuple[Path, list[Path]]:
    spec = G3_CASES[experiment_id]
    config = load_experiment_config(config_path)
    config_payload = config.to_dict()
    config_sha256 = canonical_sha256(config_payload)
    evidence = root / "results/evidence/task2" / experiment_id
    evidence.mkdir(parents=True, exist_ok=True)
    offsets = (-0.004, -0.002, 0.0, 0.002, 0.004)
    fold_scores = [pooled_macro_f1 + offset for offset in offsets]
    run_ids = [f"{experiment_id}-f{fold}" for fold in range(5)]
    history_paths: list[Path] = []
    registry_rows: list[dict] = []
    for fold, (run_id, score, epoch_count) in enumerate(
        zip(run_ids, fold_scores, spec["epochs"], strict=True)
    ):
        history_path = root / "tmp/task2/runs" / run_id / "history.json"
        history = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "fold": fold,
            "seed": 2753,
            "config": config_payload,
            "epoch_history": [
                {
                    "epoch": epoch,
                    "learning_rate": spec["learning_rate"],
                    "train_loss": 1.2 - 0.02 * epoch,
                    "validation_loss": 1.15 - 0.015 * epoch,
                    "validation_accuracy": score + 0.02 - 0.001 * (epoch_count - epoch),
                    "validation_macro_f1": score - 0.001 * (epoch_count - epoch),
                }
                for epoch in range(1, epoch_count + 1)
            ],
        }
        _write_json(history_path, history)
        history_paths.append(history_path)
        registry_rows.append(
            {
                "run_id": run_id,
                "stage": "g3_full_budget",
                "experiment_id": experiment_id,
                "model_family": spec["model_family"],
                "benchmark_only": False,
                "final_eligible": True,
                "scratch": True,
                "fold": fold,
                "seed": 2753,
                "git_dirty": False,
                "config_sha256": config_sha256,
                "split_sha256": "a" * 64,
                "label_map_sha256": "b" * 64,
                "implementation_sha256": "c" * 64,
                "transform_id": "a0-transform",
                "loss_id": "cross_entropy",
                "epochs_completed": epoch_count,
                "best_epoch": epoch_count,
                "primary_metric_value": score,
                "runtime_seconds": (45.0 if spec["family"] == "C1" else 180.0)
                + fold,
                "peak_vram_mb": 200.0 if spec["family"] == "C1" else 600.0,
                "parameter_count": spec["parameter_count"],
                "history_path": history_path.relative_to(root).as_posix(),
                "history_sha256": compute_sha256(history_path),
                "status": "completed",
            }
        )

    pooled_path = _write_json(
        evidence / "pooled_metrics.json",
        {
            "macro_f1": pooled_macro_f1,
            "per_class": {
                label: {
                    "precision": pooled_macro_f1 + offset,
                    "recall": pooled_macro_f1 + offset,
                    "f1": pooled_macro_f1 + offset,
                    "support": support,
                }
                for label, offset, support in (
                    ("Fall", -0.02, 6),
                    ("Spring", 0.01, 2),
                    ("Summer", 0.02, 8),
                    ("Winter", -0.01, 4),
                )
            },
        },
    )
    fold_summary_path = evidence / "fold_summary.csv"
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": pd.Series(fold_scores).mean(),
                "fold_sd": pd.Series(fold_scores).std(ddof=1),
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    fold_metrics_path = evidence / "fold_metrics.csv"
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "fold": fold,
                "macro_f1": fold_scores[fold],
            }
            for fold, run_id in enumerate(run_ids)
        ]
    ).to_csv(fold_metrics_path, index=False)
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
        "folds": list(range(5)),
        "run_ids": run_ids,
        "coverage": {
            "row_count": 20,
            "unique_id_count": 20,
            "expected_row_count": 20,
            "id_set_sha256": "d" * 64,
            "labels": ["Fall", "Spring", "Summer", "Winter"],
            "protected_id_count": 0,
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
    return _write_json(evidence / "manifest.json", manifest), history_paths


def _inputs(
    root: Path,
    *,
    c1_score: float = 0.737,
    c2_score: float = 0.735,
    full_batch_overrides: dict[str, int] | None = None,
) -> tuple[list[Path], list[Path], Path, dict[str, list[Path]]]:
    manifests: list[Path] = []
    full_configs: list[Path] = []
    screen_configs: dict[str, Path] = {}
    histories: dict[str, list[Path]] = {}
    scores = {
        "g3-c1-t1-smallcnn": c1_score,
        "g3-c2-t0-resnet18": c2_score,
    }
    for experiment_id in G3_CASES:
        screen_configs[experiment_id] = _write_config(
            root, experiment_id, screen=True
        )
        full_config = _write_config(
            root,
            experiment_id,
            screen=False,
            batch_size=(full_batch_overrides or {}).get(experiment_id, 32),
        )
        full_configs.append(full_config)
        manifest, history_paths = _write_experiment_evidence(
            root,
            experiment_id,
            full_config,
            pooled_macro_f1=scores[experiment_id],
        )
        manifests.append(manifest)
        histories[experiment_id] = history_paths
    tuning_manifest = _write_tuning_manifest(root, screen_configs)
    return manifests, full_configs, tuning_manifest, histories


def _build(root: Path, **input_overrides) -> tuple[dict, dict[str, list[Path]]]:
    manifests, configs, tuning, histories = _inputs(root, **input_overrides)
    manifest = build_g3_full_budget_evidence(
        experiment_manifest_paths=manifests,
        experiment_config_paths=configs,
        tuning_manifest_path=tuning,
        project_root=root,
        evidence_directory=root / "results/evidence/task2/g3_full_budget",
        figure_directory=root / "results/figures/task2",
    )
    return manifest, histories


def _temporary_outputs(root: Path) -> dict[str, Path]:
    return {
        "evidence_directory": root / "results/evidence/task2/g3_full_budget",
        "figure_directory": root / "results/figures/task2",
    }


def test_g3_records_near_tie_without_freezing_winner(tmp_path: Path) -> None:
    manifest, _ = _build(tmp_path)
    evidence = tmp_path / "results/evidence/task2/g3_full_budget"
    decision = json.loads((evidence / "decision.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(evidence / "leaderboard.csv")
    paired = pd.read_csv(evidence / "paired_fold_metrics.csv")
    classes = pd.read_csv(evidence / "per_class_comparison.csv")
    curves = pd.read_csv(evidence / "learning_curve_summary.csv")

    assert decision["near_tie"] is True
    assert decision["observed_c1_minus_c2_macro_f1"] == pytest.approx(0.002)
    assert decision["provisional_reference_family"] == "C1"
    assert decision["ultimate_winner_frozen"] is False
    assert manifest["ultimate_winner_frozen"] is False
    assert len(leaderboard) == 2
    assert len(paired) == 5
    assert len(classes) == 4
    assert curves["fold_count"].eq(5).all()
    assert curves.groupby("family")["epoch"].max().to_dict() == {"C1": 18, "C2": 19}
    for name in ("g3_c1_t1_learning_curves.png", "g3_c2_t0_learning_curves.png"):
        assert (tmp_path / "results/figures/task2" / name).read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )


def test_g3_rejects_change_outside_declared_budget(tmp_path: Path) -> None:
    manifests, configs, tuning, _ = _inputs(
        tmp_path,
        full_batch_overrides={"g3-c2-t0-resnet18": 16},
    )

    with pytest.raises(ValueError, match="outside budget"):
        build_g3_full_budget_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            tuning_manifest_path=tuning,
            project_root=tmp_path,
            **_temporary_outputs(tmp_path),
        )


def test_g3_rejects_tampered_history(tmp_path: Path) -> None:
    manifests, configs, tuning, histories = _inputs(tmp_path)
    history_path = histories["g3-c1-t1-smallcnn"][0]
    history_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="history artifact hash"):
        build_g3_full_budget_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            tuning_manifest_path=tuning,
            project_root=tmp_path,
            **_temporary_outputs(tmp_path),
        )


def test_g3_rejects_duplicate_run_id_in_experiment_manifest(tmp_path: Path) -> None:
    manifests, configs, tuning, _ = _inputs(tmp_path)
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_ids"].append(manifest["run_ids"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly five unique run IDs"):
        build_g3_full_budget_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            tuning_manifest_path=tuning,
            project_root=tmp_path,
            **_temporary_outputs(tmp_path),
        )


def test_g3_rejects_screen_score_not_backed_by_verified_leaderboard(
    tmp_path: Path,
) -> None:
    manifests, configs, tuning, _ = _inputs(tmp_path)
    tuning_manifest = json.loads(tuning.read_text(encoding="utf-8"))
    tuning_manifest["families"]["C1"]["selected_macro_f1"] = 0.999
    tuning.write_text(json.dumps(tuning_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="screen score disagrees with G2 leaderboard"):
        build_g3_full_budget_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            tuning_manifest_path=tuning,
            project_root=tmp_path,
            **_temporary_outputs(tmp_path),
        )


def test_g3_exact_tie_uses_cost_tiebreak_not_manifest_order(tmp_path: Path) -> None:
    manifests, configs, tuning, _ = _inputs(
        tmp_path,
        c1_score=0.735,
        c2_score=0.735,
    )
    outputs = _temporary_outputs(tmp_path)
    manifest = build_g3_full_budget_evidence(
        experiment_manifest_paths=list(reversed(manifests)),
        experiment_config_paths=configs,
        tuning_manifest_path=tuning,
        project_root=tmp_path,
        **outputs,
    )
    leaderboard = pd.read_csv(outputs["evidence_directory"] / "leaderboard.csv")

    assert leaderboard.set_index("family")["rank"].to_dict() == {"C1": 1, "C2": 2}
    assert manifest["provisional_reference_experiment_id"] == "g3-c1-t1-smallcnn"
