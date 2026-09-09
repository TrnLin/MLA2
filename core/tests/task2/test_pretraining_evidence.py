from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from fashion.config import ROOT
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import EXPERIMENT_REGISTRY_COLUMNS
from fashion.task2.experiments import load_experiment_config
from fashion.task2.pretraining import P0S_EXPERIMENT_ID, PSTAR_EXPERIMENT_ID
from fashion.task2.pretraining_evidence import (
    build_pretraining_benchmark_evidence,
    plot_pretraining_learning_curves,
)
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS

CASES = {
    P0S_EXPERIMENT_ID: {
        "filename": "g4_p0s_resnet18_standard_scratch.json",
        "family": "resnet18_standard_scratch",
        "scratch": True,
        "origin": "scratch",
        "weights": None,
        "score": 0.70,
    },
    PSTAR_EXPERIMENT_ID: {
        "filename": "g4_pstar_resnet18_standard_pretrained.json",
        "family": "resnet18_standard_pretrained",
        "scratch": False,
        "origin": "imagenet_pretrained",
        "weights": "ResNet18_Weights.DEFAULT",
        "score": 0.75,
    },
}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_config(root: Path, experiment_id: str) -> Path:
    filename = CASES[experiment_id]["filename"]
    source = ROOT / "configs/task2" / filename
    target = root / "configs/task2" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _write_pack(root: Path, experiment_id: str) -> tuple[Path, Path, list[Path]]:
    case = CASES[experiment_id]
    config_path = _copy_config(root, experiment_id)
    config = load_experiment_config(config_path)
    evidence = root / "results/evidence/task2" / experiment_id.replace("-", "_")
    run_ids = [f"{experiment_id}-f{fold}" for fold in range(5)]
    fold_scores = [case["score"] + offset for offset in (-0.004, -0.002, 0, 0.002, 0.004)]
    registry_rows = []
    history_paths = []
    fold_rows = []
    for fold, (run_id, score) in enumerate(zip(run_ids, fold_scores, strict=True)):
        history_path = root / "tmp/task2/runs" / run_id / "history.json"
        history = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "fold": fold,
            "seed": 2753,
            "config": config.to_dict(),
            "model_boundary": {
                "class": "BenchmarkStandardStemResNet18",
                "benchmark_only": True,
                "final_eligible": False,
                "training_origin": case["origin"],
                "weights": case["weights"],
            },
            "epoch_history": [
                {
                    "epoch": epoch,
                    "learning_rate": 3e-4,
                    "train_loss": 1.2 - epoch * 0.1 - fold * 0.002,
                    "validation_loss": 1.1 - epoch * 0.07 + fold * 0.002,
                    "validation_accuracy": score + epoch * 0.01,
                    "validation_macro_f1": score + epoch * 0.008,
                }
                for epoch in range(1, 4)
            ],
        }
        _write_json(history_path, history)
        history_paths.append(history_path)
        registry_rows.append(
            {
                "run_id": run_id,
                "stage": "g4_pretraining_benchmark",
                "experiment_id": experiment_id,
                "model_family": case["family"],
                "benchmark_only": True,
                "final_eligible": False,
                "scratch": case["scratch"],
                "fold": fold,
                "seed": 2753,
                "git_commit": "e" * 40,
                "git_dirty": False,
                "config_sha256": canonical_sha256(config.to_dict()),
                "split_sha256": "a" * 64,
                "label_map_sha256": "b" * 64,
                "implementation_sha256": "c" * 64,
                "transform_id": "resize_pad_80x60_a0",
                "loss_id": "cross_entropy",
                "epochs_requested": 30,
                "epochs_completed": 3,
                "best_epoch": 3,
                "primary_metric_name": "macro_f1",
                "primary_metric_value": score,
                "runtime_seconds": 60 + fold,
                "peak_vram_mb": 512,
                "parameter_count": 11_178_564,
                "checkpoint_path": f"tmp/task2/checkpoints/{run_id}.pt",
                "checkpoint_sha256": "d" * 64,
                "prediction_path": f"tmp/task2/runs/{run_id}/oof.csv",
                "prediction_sha256": "f" * 64,
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
                "seed": 2753,
                "source": "cache",
                "macro_f1": score,
                "accuracy": score + 0.02,
            }
        )
    registry_path = evidence / "registry_snapshot.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows).loc[:, EXPERIMENT_REGISTRY_COLUMNS].to_csv(
        registry_path, index=False
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
                "pooled_value": case["score"],
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    pooled_path = _write_json(
        evidence / "pooled_metrics.json",
        {
            "macro_f1": case["score"],
            "per_class": {
                label: {
                    "precision": case["score"] + offset,
                    "recall": case["score"] + offset,
                    "f1": case["score"] + offset,
                    "support": 5,
                }
                for label, offset in zip(SEASON_LABELS, (-0.02, 0.01, 0.02, -0.01), strict=True)
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
        "seed": 2753,
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
        "pooled_macro_f1": case["score"],
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": compute_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    return _write_json(evidence / "manifest.json", manifest), config_path, history_paths


def _inputs(root: Path):
    packs = [_write_pack(root, experiment_id) for experiment_id in CASES]
    return (
        [pack[0] for pack in packs],
        [pack[1] for pack in packs],
        [pack[2] for pack in packs],
    )


def _build(root: Path):
    manifests, configs, histories = _inputs(root)
    result = build_pretraining_benchmark_evidence(
        experiment_manifest_paths=manifests,
        experiment_config_paths=configs,
        project_root=root,
        expected_row_count=20,
        evidence_directory=root / "results/evidence/task2/pretraining_benchmark",
        figure_directory=root / "results/figures/task2",
    )
    return result, manifests, histories


def test_pretraining_evidence_quantifies_effect_without_selecting_pstar(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _build(tmp_path)
    evidence = tmp_path / "results/evidence/task2/pretraining_benchmark"
    decision = json.loads((evidence / "decision.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(evidence / "comparison.csv")
    paired = pd.read_csv(evidence / "paired_fold_metrics.csv")
    learning = pd.read_csv(evidence / "learning_curve_summary.csv")

    assert manifest["observed_pstar_minus_p0s_macro_f1"] == pytest.approx(0.05)
    assert decision["candidate_selection_affected"] is False
    assert decision["pstar_final_eligible"] is False
    assert decision["ultimate_winner_frozen"] is False
    assert decision["pretrained_weight_provenance"]["resolved_enum"].startswith(
        "ResNet18_Weights.IMAGENET1K_"
    )
    assert len(comparison) == 2
    assert len(paired) == 5
    assert paired["pstar_minus_p0s_macro_f1"].eq(0.05).all()
    assert learning["fold_count"].eq(5).all()
    assert learning.groupby("experiment_id")["epoch"].max().eq(3).all()
    for name in (
        "pretraining_benchmark_learning_curves.png",
        "pretraining_benchmark_effect.png",
    ):
        assert (
            (tmp_path / "results/figures/task2" / name)
            .read_bytes()
            .startswith(b"\x89PNG\r\n\x1a\n")
        )


def test_pretraining_evidence_rejects_final_eligible_pstar_registry(
    tmp_path: Path,
) -> None:
    manifests, configs, _ = _inputs(tmp_path)
    pstar_manifest_path = manifests[1]
    manifest = json.loads(pstar_manifest_path.read_text(encoding="utf-8"))
    registry_path = tmp_path / manifest["artifacts"]["registry_snapshot"]["path"]
    registry = pd.read_csv(registry_path)
    registry["final_eligible"] = True
    registry.to_csv(registry_path, index=False)
    manifest["artifacts"]["registry_snapshot"]["sha256"] = compute_sha256(registry_path)
    _write_json(pstar_manifest_path, manifest)

    with pytest.raises(ValueError, match="registry final_eligible"):
        build_pretraining_benchmark_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            project_root=tmp_path,
            expected_row_count=20,
            evidence_directory=tmp_path / "results/evidence/task2/pretraining_benchmark",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_pretraining_evidence_rejects_tampered_history(tmp_path: Path) -> None:
    manifests, configs, histories = _inputs(tmp_path)
    histories[0][0].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="history hash mismatch"):
        build_pretraining_benchmark_evidence(
            experiment_manifest_paths=manifests,
            experiment_config_paths=configs,
            project_root=tmp_path,
            expected_row_count=20,
            evidence_directory=tmp_path / "results/evidence/task2/pretraining_benchmark",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_pretraining_learning_curve_panel_titles_do_not_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    variants = {
        P0S_EXPERIMENT_ID: "P0S standard-stem scratch control",
        PSTAR_EXPERIMENT_ID: "P* standard-stem ImageNet benchmark",
    }
    rows = []
    for experiment_id, variant in variants.items():
        for epoch in range(1, 4):
            rows.append(
                {
                    "variant": variant,
                    "experiment_id": experiment_id,
                    "epoch": epoch,
                    "train_loss_mean": 1.0 - epoch * 0.1,
                    "train_loss_sd": 0.01,
                    "validation_loss_mean": 0.9 - epoch * 0.05,
                    "validation_loss_sd": 0.02,
                    "validation_accuracy_mean": 0.6 + epoch * 0.02,
                    "validation_accuracy_sd": 0.01,
                    "validation_macro_f1_mean": 0.58 + epoch * 0.02,
                    "validation_macro_f1_sd": 0.01,
                }
            )
    observed_overlaps = []
    original_savefig = Figure.savefig

    def capture_title_bounds(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes = figure.axes
        for row_index in range(2):
            left = axes[row_index * 2].title.get_window_extent(renderer)
            right = axes[row_index * 2 + 1].title.get_window_extent(renderer)
            observed_overlaps.append(left.overlaps(right))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_title_bounds)

    plot_pretraining_learning_curves(
        pd.DataFrame(rows),
        tmp_path / "learning_curves.png",
    )

    assert observed_overlaps == [False, False]
