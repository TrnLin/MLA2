from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.axes import Axes

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    _plot_g1_family_screen,
    build_g1_family_screen_evidence,
)


def _write_experiment_evidence(
    root: Path,
    *,
    experiment_id: str,
    model_family: str,
    pooled_macro_f1: float,
    parameter_count: int,
) -> Path:
    slug = experiment_id.replace("-", "_")
    evidence = root / "results/evidence/task2" / slug
    evidence.mkdir(parents=True, exist_ok=True)
    pooled_path = evidence / "pooled_metrics.json"
    fold_summary_path = evidence / "fold_summary.csv"
    registry_path = evidence / "registry_snapshot.csv"
    pooled_path.write_text(
        json.dumps(
            {
                "macro_f1": pooled_macro_f1,
                "per_class": {"Spring": {"f1": pooled_macro_f1 - 0.1}},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": pooled_macro_f1 - 0.001,
                "fold_sd": 0.01,
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    run_ids = [f"{experiment_id}-f{fold}" for fold in range(5)]
    pd.DataFrame(
        [
            {
                "run_id": run_ids[fold],
                "experiment_id": experiment_id,
                "model_family": model_family,
                "benchmark_only": False,
                "final_eligible": True,
                "scratch": True,
                "fold": fold,
                "runtime_seconds": 60 + fold,
                "peak_vram_mb": 100 + fold,
                "parameter_count": parameter_count,
                "status": "completed",
            }
            for fold in range(5)
        ]
    ).to_csv(registry_path, index=False)
    artifacts = {
        "pooled_metrics": pooled_path,
        "fold_summary": fold_summary_path,
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
            "id_set_sha256": "a" * 64,
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
    return manifest_path


def _g1_manifests(root: Path) -> list[Path]:
    return [
        _write_experiment_evidence(
            root,
            experiment_id="g1-c1-smallcnn",
            model_family="smallcnn",
            pooled_macro_f1=0.70,
            parameter_count=1_100_000,
        ),
        _write_experiment_evidence(
            root,
            experiment_id="g1-c2-resnet18",
            model_family="resnet18",
            pooled_macro_f1=0.71,
            parameter_count=11_000_000,
        ),
        _write_experiment_evidence(
            root,
            experiment_id="g1-c3-mobilenetv3",
            model_family="mobilenet_v3_small",
            pooled_macro_f1=0.64,
            parameter_count=1_500_000,
        ),
    ]


def test_g1_evidence_ranks_and_shortlists_two_best_families(tmp_path: Path) -> None:
    manifests = _g1_manifests(tmp_path)
    reference = _write_experiment_evidence(
        tmp_path,
        experiment_id="b1-hog-hsv-svm",
        model_family="hog_hsv_svm",
        pooled_macro_f1=0.61,
        parameter_count=8_000,
    )
    manifest = build_g1_family_screen_evidence(
        manifests,
        reference_manifest_path=reference,
        project_root=tmp_path,
        evidence_directory=tmp_path / "results/evidence/task2/g1_family_screen",
        figure_directory=tmp_path / "results/figures/task2",
    )

    result_root = tmp_path / "results"
    leaderboard = pd.read_csv(
        result_root / manifest["artifacts"]["leaderboard"]["path"]
    )
    assert leaderboard["experiment_id"].tolist() == [
        "g1-c2-resnet18",
        "g1-c1-smallcnn",
        "g1-c3-mobilenetv3",
    ]
    assert leaderboard["shortlisted"].tolist() == [True, True, False]
    assert manifest["selected_experiment_ids"] == [
        "g1-c2-resnet18",
        "g1-c1-smallcnn",
    ]
    assert manifest["reference"]["pooled_macro_f1"] == pytest.approx(0.61)
    figure = result_root / manifest["artifacts"]["figure"]["path"]
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_g1_evidence_rejects_tampered_input_artifact(tmp_path: Path) -> None:
    manifests = _g1_manifests(tmp_path)
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    pooled_path = tmp_path / manifest["artifacts"]["pooled_metrics"]["path"]
    pooled_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash does not match"):
        build_g1_family_screen_evidence(
            manifests,
            project_root=tmp_path,
            evidence_directory=tmp_path / "results/evidence/task2/g1_family_screen",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_g1_figure_places_rightmost_label_inside_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_id": "g1-c1-smallcnn",
                "parameter_count": 1_100_000,
                "pooled_macro_f1": 0.70,
                "five_fold_runtime_minutes": 19.0,
                "shortlisted": True,
            },
            {
                "experiment_id": "g1-c2-resnet18",
                "parameter_count": 11_000_000,
                "pooled_macro_f1": 0.71,
                "five_fold_runtime_minutes": 29.0,
                "shortlisted": True,
            },
        ]
    )
    calls: list[dict[str, object]] = []
    original = Axes.annotate

    def capture(self, text, *args, **kwargs):
        calls.append({"text": text, **kwargs})
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", capture)
    _plot_g1_family_screen(
        leaderboard,
        reference_macro_f1=0.61,
        output_path=tmp_path / "g1.png",
    )

    rightmost = next(call for call in calls if "resnet18" in str(call["text"]))
    assert rightmost["ha"] == "right"
    assert rightmost["xytext"][0] < 0
