from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.axes import Axes

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import G2_TUNING_SPECS, build_g2_tuning_evidence
from fashion.task2.experiments import load_experiment_config
from fashion.train.artifacts import canonical_sha256


def _write_config(
    root: Path,
    experiment_id: str,
    *,
    batch_size: int = 32,
) -> Path:
    spec = G2_TUNING_SPECS[experiment_id]
    path = root / "configs/task2" / f"{experiment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "experiment_id": experiment_id,
                "method": "deep",
                "model_family": spec["model_family"],
                "stage": spec["stage"],
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
                    "epochs": 8,
                    "learning_rate": spec["learning_rate"],
                    "weight_decay": spec["weight_decay"],
                    "effective_batch_size": 128,
                    "gradient_clip_norm": 1.0,
                    "warmup_epochs": 1.0,
                    "patience": 8,
                    "min_delta": 1e-4,
                    "use_amp": True,
                    "device": "auto",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_experiment_evidence(
    root: Path,
    experiment_id: str,
    *,
    config_path: Path,
    pooled_macro_f1: float,
) -> tuple[Path, list[Path]]:
    spec = G2_TUNING_SPECS[experiment_id]
    evidence = root / "results/evidence/task2" / experiment_id
    run_root = root / "tmp/task2/runs"
    evidence.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(config_path)
    config_payload = config.to_dict()
    config_sha256 = canonical_sha256(config_payload)
    run_ids = [f"{experiment_id}-f{fold}" for fold in range(5)]
    fold_scores = [pooled_macro_f1 - 0.002 + 0.001 * fold for fold in range(5)]
    history_paths: list[Path] = []
    registry_rows = []
    for fold, (run_id, fold_score) in enumerate(zip(run_ids, fold_scores, strict=True)):
        history_path = run_root / run_id / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
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
                    "learning_rate": float(spec["learning_rate"]),
                    "train_loss": 1.3 - 0.08 * epoch,
                    "validation_loss": 1.25 - 0.07 * epoch,
                    "validation_accuracy": fold_score - 0.08 + 0.01 * epoch,
                    "validation_macro_f1": fold_score - 0.08 + 0.01 * epoch,
                }
                for epoch in range(1, 9)
            ],
        }
        history_path.write_text(json.dumps(history), encoding="utf-8")
        history_paths.append(history_path)
        registry_rows.append(
            {
                "run_id": run_id,
                "stage": spec["stage"],
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
                "epochs_completed": 8,
                "best_epoch": 8,
                "primary_metric_value": fold_score,
                "runtime_seconds": 60.0 + fold,
                "peak_vram_mb": 600.0,
                "parameter_count": (
                    1_174_244 if spec["family"] == "C1" else 11_170_884
                ),
                "history_path": history_path.relative_to(root).as_posix(),
                "history_sha256": compute_sha256(history_path),
                "status": "completed",
            }
        )

    pooled_path = evidence / "pooled_metrics.json"
    fold_summary_path = evidence / "fold_summary.csv"
    fold_metrics_path = evidence / "fold_metrics.csv"
    registry_path = evidence / "registry_snapshot.csv"
    class_offsets = {"Fall": -0.02, "Spring": 0.02, "Summer": 0.01, "Winter": -0.01}
    supports = {"Fall": 6, "Spring": 2, "Summer": 8, "Winter": 4}
    pooled_path.write_text(
        json.dumps(
            {
                "macro_f1": pooled_macro_f1,
                "per_class": {
                    label: {
                        "precision": pooled_macro_f1 + offset,
                        "recall": pooled_macro_f1 + offset,
                        "f1": pooled_macro_f1 + offset,
                        "support": supports[label],
                    }
                    for label, offset in class_offsets.items()
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": sum(fold_scores) / len(fold_scores),
                "fold_sd": pd.Series(fold_scores).std(ddof=1),
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "run_id": run_ids[fold],
                "fold": fold,
                "seed": 2753,
                "source": "cache",
                "macro_f1": fold_scores[fold],
            }
            for fold in range(5)
        ]
    ).to_csv(fold_metrics_path, index=False)
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
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, history_paths


def _inputs(
    root: Path,
    *,
    score_overrides: dict[str, float] | None = None,
    batch_overrides: dict[str, int] | None = None,
) -> tuple[list[Path], list[Path], dict[str, list[Path]], Path]:
    scores = {
        "g1-c1-smallcnn": 0.700,
        "g2-t1-c1-smallcnn": 0.708,
        "g2-t2-c1-smallcnn": 0.701,
        "g1-c2-resnet18": 0.710,
        "g2-t1-c2-resnet18": 0.700,
        "g2-t2-c2-resnet18": 0.712,
    }
    scores.update(score_overrides or {})
    batches = batch_overrides or {}
    config_paths: list[Path] = []
    manifest_paths: list[Path] = []
    histories: dict[str, list[Path]] = {}
    for experiment_id in G2_TUNING_SPECS:
        config_path = _write_config(
            root,
            experiment_id,
            batch_size=batches.get(experiment_id, 32),
        )
        manifest_path, history_paths = _write_experiment_evidence(
            root,
            experiment_id,
            config_path=config_path,
            pooled_macro_f1=scores[experiment_id],
        )
        config_paths.append(config_path)
        manifest_paths.append(manifest_path)
        histories[experiment_id] = history_paths
    decision_path = root / "results/evidence/task2/g2_augmentation_ablation/decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        json.dumps({"decision_status": "closed", "selected_variant": "A0"}),
        encoding="utf-8",
    )
    return manifest_paths, config_paths, histories, decision_path


def _build(root: Path, **input_overrides) -> tuple[dict, dict[str, list[Path]]]:
    manifests, configs, histories, decision = _inputs(root, **input_overrides)
    manifest = build_g2_tuning_evidence(
        experiment_manifest_paths=manifests,
        experiment_config_paths=configs,
        augmentation_decision_path=decision,
        project_root=root,
        evidence_directory=root / "results/evidence/task2/g2_compact_tuning",
        figure_directory=root / "results/figures/task2",
    )
    return manifest, histories


def test_g2_tuning_selects_only_meaningful_gain_and_writes_learning_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plot_calls: list[tuple] = []
    bar_calls: list[tuple] = []
    original_plot = Axes.plot
    original_bar = Axes.bar

    def capture_plot(self, *args, **kwargs):
        plot_calls.append(args)
        return original_plot(self, *args, **kwargs)

    def capture_bar(self, *args, **kwargs):
        bar_calls.append(args)
        return original_bar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    monkeypatch.setattr(Axes, "bar", capture_bar)
    manifest, _ = _build(tmp_path)
    result_root = tmp_path / "results"
    decision = json.loads(
        (result_root / manifest["artifacts"]["decision"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    leaderboard = pd.read_csv(
        result_root / manifest["artifacts"]["leaderboard"]["path"]
    )
    histories = pd.read_csv(
        result_root / manifest["artifacts"]["learning_curves_by_fold"]["path"]
    )
    summary = pd.read_csv(
        result_root / manifest["artifacts"]["learning_curve_summary"]["path"]
    )

    assert decision["families"]["C1"]["selected_tuning_id"] == "T1"
    assert decision["families"]["C2"]["selected_tuning_id"] == "T0"
    assert leaderboard.loc[leaderboard["selected"], "tuning_id"].tolist() == ["T1", "T0"]
    assert len(histories) == 6 * 5 * 8
    assert len(summary) == 6 * 8
    assert len(plot_calls) == 8
    assert not bar_calls
    for name in ("c1_learning_curves", "c2_learning_curves"):
        figure = result_root / manifest["artifacts"][name]["path"]
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_g2_tuning_accepts_an_exact_minimum_gain(tmp_path: Path) -> None:
    manifest, _ = _build(
        tmp_path,
        score_overrides={"g2-t2-c2-resnet18": 0.713},
    )
    decision_path = tmp_path / "results" / manifest["artifacts"]["decision"]["path"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["families"]["C2"]["best_observed_gain_over_t0"] == pytest.approx(
        0.003
    )
    assert decision["families"]["C2"]["gain_gate_passed"] is True
    assert decision["families"]["C2"]["selected_tuning_id"] == "T2"


def test_g2_tuning_rejects_a_hidden_batch_size_change(tmp_path: Path) -> None:
    manifests, configs, _, decision = _inputs(
        tmp_path,
        batch_overrides={"g2-t2-c2-resnet18": 16},
    )

    with pytest.raises(ValueError, match="differ outside"):
        build_g2_tuning_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            augmentation_decision_path=decision,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/g2_compact_tuning",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_g2_tuning_rejects_a_tampered_history(tmp_path: Path) -> None:
    manifests, configs, histories, decision = _inputs(tmp_path)
    history = histories["g2-t1-c1-smallcnn"][0]
    history.write_text(history.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="history artifact hash"):
        build_g2_tuning_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            augmentation_decision_path=decision,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/g2_compact_tuning",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_g2_tuning_requires_the_closed_a0_decision(tmp_path: Path) -> None:
    manifests, configs, _, decision = _inputs(tmp_path)
    decision.write_text(
        json.dumps({"decision_status": "pending", "selected_variant": None}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="closed A0"):
        build_g2_tuning_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            augmentation_decision_path=decision,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/g2_compact_tuning",
            figure_directory=tmp_path / "results/figures/task2",
        )
