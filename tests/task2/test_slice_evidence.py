from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import EXPERIMENT_REGISTRY_COLUMNS
from fashion.task2.slice_evidence import (
    build_slice_decision,
    load_candidate_oof_packs,
    load_declared_config_hashes,
    load_verified_stability_manifest,
)
from fashion.task2.slices import (
    CandidateExperiment,
    SliceAnalysisSpec,
    SliceAnalysisTables,
)
from fashion.task2.stability_evidence import EXPECTED_EXPERIMENTS
from fashion.train.artifacts import ArtifactVerificationError
from fashion.train.metrics import SEASON_LABELS


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _declaration(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": compute_sha256(path),
    }


def _stability_manifest(root: Path) -> Path:
    artifacts = {
        name: _declaration(
            root,
            _write_text(root / f"results/{name}.txt", f"{name}\n"),
        )
        for name in ("registry_snapshot", "seed_stability", "decision")
    }
    input_configs = {
        experiment_id: _declaration(
            root,
            _write_text(root / f"configs/{experiment_id}.json", "{}\n"),
        )
        for experiment_id in EXPECTED_EXPERIMENTS
    }
    input_manifests = {
        experiment_id: _declaration(
            root,
            _write_text(root / f"manifests/{experiment_id}.json", "{}\n"),
        )
        for experiment_id in EXPECTED_EXPERIMENTS
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G5-SEED",
        "decision_status": "closed",
        "ordering_stable": True,
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "artifacts": artifacts,
        "input_configs": input_configs,
        "input_manifests": input_manifests,
    }
    path = root / "results/seed-stability-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _spec() -> SliceAnalysisSpec:
    return SliceAnalysisSpec(
        analysis_id="g6-shortcut-error-slices",
        expected_row_count=20,
        candidates=(
            CandidateExperiment("C2", "g3-c2-t0-resnet18", 2753),
            CandidateExperiment("C2", "g5-c2-t0-resnet18-s2026", 2026),
            CandidateExperiment("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
            CandidateExperiment(
                "I2",
                "g5-i2-article-type-lambda-0-3-c1-s2026",
                2026,
            ),
        ),
        high_confidence_threshold=0.8,
        maximum_ranked_confusions=8,
        low_support_threshold=3,
        quantiles=(0.25, 0.5, 0.75),
    )


def _registry_and_oof(root: Path) -> tuple[pd.DataFrame, dict[str, str], dict[int, str]]:
    targets = {identifier: SEASON_LABELS[(identifier - 1) // 5] for identifier in range(1, 21)}
    config_hashes = {
        candidate.experiment_id: str(index) * 64
        for index, candidate in enumerate(_spec().candidates, start=1)
    }
    rows = []
    for candidate in _spec().candidates:
        expected = EXPECTED_EXPERIMENTS[candidate.experiment_id]
        implementation_hash = "a" * 64 if candidate.candidate == "C2" else "b" * 64
        for fold in range(5):
            run_id = f"{candidate.experiment_id}-f{fold}-s{candidate.seed}"
            fold_ids = [identifier for identifier in targets if (identifier - 1) % 5 == fold]
            oof_rows = []
            for identifier in fold_ids:
                truth = targets[identifier]
                probabilities = {
                    f"prob_{label}": 0.7 if label == truth else 0.1 for label in SEASON_LABELS
                }
                oof_rows.append(
                    {
                        "run_id": run_id,
                        "experiment_id": candidate.experiment_id,
                        "id": identifier,
                        "fold": fold,
                        "seed": candidate.seed,
                        "y_true": truth,
                        "y_pred": truth,
                        **probabilities,
                    }
                )
            prediction_path = root / f"tmp/{run_id}/oof.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(oof_rows).to_csv(prediction_path, index=False)
            row = {
                "candidate": candidate.candidate,
                "run_id": run_id,
                "stage": expected["stage"],
                "experiment_id": candidate.experiment_id,
                "model_family": expected["model_family"],
                "benchmark_only": "false",
                "final_eligible": "true",
                "scratch": "true",
                "fold": fold,
                "seed": candidate.seed,
                "git_commit": "c" * 40,
                "git_dirty": "false",
                "config_sha256": config_hashes[candidate.experiment_id],
                "split_sha256": "d" * 64,
                "label_map_sha256": "e" * 64,
                "implementation_sha256": implementation_hash,
                "transform_id": "a0-test",
                "loss_id": expected["loss_id"],
                "epochs_requested": 3,
                "epochs_completed": 3,
                "best_epoch": 2,
                "primary_metric_name": "macro_f1",
                "primary_metric_value": 1.0,
                "runtime_seconds": 1.0,
                "peak_vram_mb": 1.0,
                "parameter_count": 10 if candidate.candidate == "C2" else 5,
                "checkpoint_path": "unused.pt",
                "checkpoint_sha256": "f" * 64,
                "prediction_path": prediction_path.relative_to(root).as_posix(),
                "prediction_sha256": compute_sha256(prediction_path),
                "history_path": "unused.json",
                "history_sha256": "0" * 64,
                "status": "completed",
            }
            rows.append(row)
    return (
        pd.DataFrame(rows).loc[:, ["candidate", *EXPERIMENT_REGISTRY_COLUMNS]],
        config_hashes,
        targets,
    )


def test_stability_manifest_verifies_every_declared_input(tmp_path: Path) -> None:
    path = _stability_manifest(tmp_path)

    manifest, resolved, sections = load_verified_stability_manifest(
        path.relative_to(tmp_path),
        project_root=tmp_path,
    )

    assert resolved == path
    assert manifest["ordering_stable"] is True
    assert set(sections) == {"artifacts", "input_configs", "input_manifests"}
    assert len(sections["input_configs"]) == 4


def test_stability_manifest_rejects_changed_artifact_bytes(tmp_path: Path) -> None:
    path = _stability_manifest(tmp_path)
    _write_text(tmp_path / "results/decision.txt", "changed\n")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_stability_manifest(path, project_root=tmp_path)


def test_repository_config_hashes_match_normalised_registry_contract() -> None:
    _, _, sections = load_verified_stability_manifest()
    registry = pd.read_csv(
        sections["artifacts"]["registry_snapshot"],
        dtype=str,
        keep_default_na=False,
    )

    hashes = load_declared_config_hashes(sections["input_configs"])

    for experiment_id, digest in hashes.items():
        recorded = set(
            registry.loc[
                registry["experiment_id"].eq(experiment_id),
                "config_sha256",
            ]
        )
        assert recorded == {digest}


def test_candidate_loader_verifies_twenty_exactly_once_fold_files(tmp_path: Path) -> None:
    registry, config_hashes, targets = _registry_and_oof(tmp_path)

    packs, coverage, predictions = load_candidate_oof_packs(
        registry,
        _spec(),
        project_root=tmp_path,
        expected_ids=targets,
        expected_targets=targets,
        protected_ids={999},
        split_sha256="d" * 64,
        label_map_sha256="e" * 64,
        config_sha256_by_experiment=config_hashes,
    )

    assert len(packs) == 4
    assert all(len(pack.oof) == 20 for pack in packs)
    assert all(item["row_count"] == 20 for item in coverage.values())
    assert len(predictions) == 20


def test_slice_decision_reports_weaknesses_without_post_hoc_selection() -> None:
    deltas = pd.DataFrame(
        [
            {
                "seed": 2753,
                "slice_family": "image_mode",
                "slice_name": "rgb",
                "i2_minus_c2_macro_f1": 0.02,
            },
            {
                "seed": 2026,
                "slice_family": "image_mode",
                "slice_name": "rgb",
                "i2_minus_c2_macro_f1": -0.01,
            },
        ]
    )
    spring = pd.DataFrame(
        [
            {"candidate": candidate, "seed": seed, "recall": recall, "f1": recall - 0.01}
            for candidate, seed, recall in (
                ("C2", 2753, 0.70),
                ("I2", 2753, 0.72),
                ("C2", 2026, 0.71),
                ("I2", 2026, 0.70),
            )
        ]
    )
    tables = SliceAnalysisTables(
        slice_metrics=pd.DataFrame(),
        candidate_slice_deltas=deltas,
        slice_contrasts=pd.DataFrame(),
        spring_metrics=spring,
        spring_destinations=pd.DataFrame(),
        error_confusions=pd.DataFrame(),
        error_examples=pd.DataFrame(),
    )

    decision = build_slice_decision(tables, stability_decision={"current_candidate": "I2"})

    assert decision["i2_below_c2_slice_count"] == 1
    assert decision["slice_delta_sign_reversals_between_seeds"] == 1
    assert decision["candidate_selection_affected"] is False
    assert decision["ultimate_winner_frozen"] is False
