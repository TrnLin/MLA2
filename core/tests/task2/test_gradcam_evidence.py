from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import fashion.task2.gradcam_evidence as evidence_module
from fashion.data.hashing import compute_sha256
from fashion.data.torch import FoldImageStats
from fashion.task2.bootstrap_evidence import BOOTSTRAP_IMPLEMENTATION_PATHS
from fashion.task2.gradcam import GradCamComputation, load_gradcam_review_spec
from fashion.task2.gradcam_evidence import (
    GRADCAM_IMPLEMENTATION_PATHS,
    build_gradcam_decision,
    build_gradcam_failure_evidence,
    load_verified_bootstrap_manifest,
)
from fashion.train.artifacts import canonical_sha256
from fashion.train.metrics import SEASON_LABELS


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _declaration(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": compute_sha256(path)}


def _bootstrap_manifest(root: Path) -> Path:
    runtime_payload = {"python": "3.12", "torch": "2.13.0"}
    runtime = _write(root / "artifacts/runtime.json", json.dumps(runtime_payload))
    decision = _write(
        root / "artifacts/decision.json",
        json.dumps(
            {
                "gate": "G6-PAIRED-BOOTSTRAP",
                "decision_status": "closed",
                "analysis_role": "development_oof_fitted_pair_uncertainty_only",
                "current_candidate": "I2",
                "candidate_selection_affected": False,
                "new_candidates_allowed": False,
                "ultimate_winner_frozen": False,
                "holdout_opened": False,
                "random_seed_generalisability_claim_allowed": False,
                "pair_outcomes": {
                    "primary_interval": {"interval_contains_zero": False},
                    "stability_sensitivity": {"interval_contains_zero": False},
                },
            }
        ),
    )
    artifacts = {
        "decision": _declaration(decision),
        "runtime": _declaration(runtime),
    }
    for name in (
        "group_audit",
        "interval_summary",
        "observed_metrics",
        "paired_bootstrap_figure",
        "registry_snapshot",
    ):
        artifacts[name] = _declaration(_write(root / f"artifacts/{name}.txt", name))
    predictions = {
        f"run-{index}": _declaration(_write(root / f"predictions/{index}.csv", "id\n1\n"))
        for index in range(20)
    }
    config = _write(root / "config.json", "{}\n")
    calibration = _write(root / "calibration.json", "{}\n")
    slices = _write(root / "slices.json", "{}\n")
    stability = _write(root / "stability.json", "{}\n")
    splits = _write(root / "splits.csv", "id\n1\n")
    labels = _write(root / "labels.json", "{}\n")
    draws = _write(root / "draws.csv", "replicate\n0\n")
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-PAIRED-BOOTSTRAP",
        "decision_status": "closed",
        "analysis_role": "development_oof_fitted_pair_uncertainty_only",
        "candidate_selection_affected": False,
        "new_candidates_allowed": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "random_seed_generalisability_claim_allowed": False,
        "labels": list(SEASON_LABELS),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "analysis_config": _declaration(config),
        "calibration_manifest": _declaration(calibration),
        "slice_manifest": _declaration(slices),
        "stability_manifest": _declaration(stability),
        "stability_coverage_sha256": "b" * 64,
        "slice_assignment_sha256": "c" * 64,
        "canonical_inputs": {
            "splits": _declaration(splits),
            "label_maps": _declaration(labels),
        },
        "implementation_sha256": "d" * 64,
        "implementation_files_at_head": ["src/example.py"],
        "runtime_sha256": canonical_sha256(runtime_payload),
        "input_predictions": predictions,
        "temporary_bootstrap_draws": {
            "path": str(draws),
            "sha256": compute_sha256(draws),
            "row_count": 20_000,
            "rows_per_comparison": {"primary": 10_000, "stability": 10_000},
        },
        "artifacts": artifacts,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_bootstrap_manifest_loader_verifies_complete_boundary(tmp_path: Path) -> None:
    path = _bootstrap_manifest(tmp_path)

    manifest, resolved, sections = load_verified_bootstrap_manifest(
        path,
        project_root=tmp_path,
    )

    assert resolved == path
    assert manifest["gate"] == "G6-PAIRED-BOOTSTRAP"
    assert len(sections["input_predictions"]) == 20
    assert set(sections["artifacts"]) == {
        "decision",
        "group_audit",
        "interval_summary",
        "observed_metrics",
        "paired_bootstrap_figure",
        "registry_snapshot",
        "runtime",
    }


def test_bootstrap_manifest_loader_rejects_changed_decision_bytes(tmp_path: Path) -> None:
    path = _bootstrap_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    decision = Path(manifest["artifacts"]["decision"]["path"])
    decision.write_text("{}\n", encoding="utf-8")

    with pytest.raises(Exception, match="SHA-256 mismatch"):
        load_verified_bootstrap_manifest(path, project_root=tmp_path)


def test_gradcam_implementation_hash_extends_bootstrap_boundary() -> None:
    assert set(BOOTSTRAP_IMPLEMENTATION_PATHS) <= set(GRADCAM_IMPLEMENTATION_PATHS)
    assert {
        "src/fashion/data/images.py",
        "src/fashion/data/torch.py",
        "src/fashion/models/season.py",
        "src/fashion/task2/gradcam.py",
        "src/fashion/task2/gradcam_evidence.py",
    } <= set(GRADCAM_IMPLEMENTATION_PATHS)


def test_selected_image_manifest_rejects_conflicting_provenance() -> None:
    selected = pd.DataFrame(
        {
            "id": [17, 17],
            "path": ["images/17.jpg", "images/moved-17.jpg"],
            "image_sha256": ["a" * 64, "b" * 64],
        }
    )

    with pytest.raises(ValueError, match="provenance conflicts"):
        evidence_module._validated_selected_images(selected)


def _decision_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reviewed_rows = []
    taxonomy_rows = []
    identifier = 1
    for candidate in ("C2", "I2"):
        for label in SEASON_LABELS:
            for group in ("correct", "incorrect"):
                for rank in range(1, 4):
                    reviewed_rows.append(
                        {
                            "candidate": candidate,
                            "true_label": label,
                            "selection_group": group,
                            "selection_rank": rank,
                            "id": identifier,
                            "probability_max_absolute_delta": 1e-6,
                            "attention_review_flag": rank == 1,
                            "zero_heatmap": False,
                        }
                    )
                    if group == "incorrect":
                        taxonomy_rows.append(
                            {
                                "candidate": candidate,
                                "id": identifier,
                                "primary_failure_hypothesis": "article_type_shortcut_conflict",
                                "causal_claim_allowed": False,
                            }
                        )
                    identifier += 1
    taxonomy = pd.DataFrame(taxonomy_rows)
    summary = (
        taxonomy.groupby(["candidate", "primary_failure_hypothesis"], observed=True)
        .size()
        .rename("selected_error_count")
        .reset_index()
    )
    return pd.DataFrame(reviewed_rows), taxonomy, summary


def test_gradcam_decision_closes_review_without_freezing_winner() -> None:
    reviewed, taxonomy, summary = _decision_frames()

    decision = build_gradcam_decision(
        reviewed,
        taxonomy,
        summary,
        load_gradcam_review_spec(),
    )

    assert decision["decision_status"] == "closed"
    assert decision["current_candidate"] == "I2"
    assert decision["ultimate_winner_frozen"] is False
    assert decision["causal_failure_claim_allowed"] is False
    assert len(decision["selected_example_counts"]) == 16
    assert decision["attention_review_flag_count_by_candidate"] == {"C2": 8, "I2": 8}


def test_gradcam_decision_rejects_probability_drift() -> None:
    reviewed, taxonomy, summary = _decision_frames()
    reviewed.loc[0, "probability_max_absolute_delta"] = 0.1

    with pytest.raises(ValueError, match="probability drift"):
        build_gradcam_decision(
            reviewed,
            taxonomy,
            summary,
            load_gradcam_review_spec(),
        )


def _synthetic_oof(
    identifiers: list[int],
    targets: dict[int, str],
    candidate: str,
    experiment_id: str,
) -> pd.DataFrame:
    rows = []
    for index, identifier in enumerate(identifiers):
        true_label = targets[identifier]
        label_index = SEASON_LABELS.index(true_label)
        correct = index % 6 < 3
        predicted_index = label_index if correct else (label_index + 1) % len(SEASON_LABELS)
        probabilities = np.full(len(SEASON_LABELS), 0.01 / 3)
        probabilities[predicted_index] = 0.99
        row: dict[str, object] = {
            "id": identifier,
            "fold": identifier % 5,
            "y_true": true_label,
            "y_pred": SEASON_LABELS[predicted_index],
            "run_id": f"{candidate.lower()}-fold-{identifier % 5}",
        }
        row.update(
            {
                f"prob_{label}": float(probabilities[position])
                for position, label in enumerate(SEASON_LABELS)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_gradcam_builder_is_deterministic_and_preserves_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    spec = replace(load_gradcam_review_spec(), expected_row_count=24)
    identifiers: list[int] = []
    targets: dict[int, str] = {}
    for label_index, label in enumerate(SEASON_LABELS):
        for offset in range(6):
            identifier = 1000 + label_index * 10 + offset
            identifiers.append(identifier)
            targets[identifier] = label
    image_rows = []
    for identifier in identifiers:
        image_path = root / f"images/{identifier}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 80), color=(80, 120, 160)).save(image_path)
        image_rows.append(
            {
                "id": identifier,
                "partition": "development",
                "season": targets[identifier],
                "cv_fold": identifier % 5,
                "path": f"images/{identifier}.jpg",
                "sha256": compute_sha256(image_path),
                "articleType": "Tshirts",
                "year": 2012,
                "productDisplayName": f"Product {identifier}",
            }
        )
    splits = pd.DataFrame(
        image_rows
        + [
            {
                "id": 9999,
                "partition": "holdout",
                "season": "",
                "cv_fold": np.nan,
                "path": "protected.jpg",
                "sha256": "f" * 64,
                "articleType": "",
                "year": 2012,
                "productDisplayName": "Protected",
            }
        ]
    )
    development = splits.loc[splits["partition"].eq("development")].reset_index(drop=True)
    assignments = development.loc[:, ["id"]].copy()
    assignments["article_type_shortcut"] = np.where(
        np.arange(len(assignments)) % 6 < 3,
        "aligned",
        "conflict",
    )
    assignments["acquisition_year"] = "dominant_2011_2012"
    assignments["file_size_quartile"] = "q3"
    assignments["product_family_size"] = "singleton"
    assignments["image_mode"] = "rgb"

    packs = []
    calibrated_parts = []
    for candidate, experiment_id in (
        ("C2", "g3-c2-t0-resnet18"),
        ("I2", "g4-i2-article-type-lambda-0-3-c1"),
    ):
        oof = _synthetic_oof(identifiers, targets, candidate, experiment_id)
        registry_rows = []
        for fold in range(5):
            checkpoint = _write(root / f"checkpoints/{candidate}-{fold}.pt", "checkpoint")
            registry_rows.append(
                {
                    "candidate": candidate,
                    "experiment_id": experiment_id,
                    "seed": "2753",
                    "fold": str(fold),
                    "run_id": f"{candidate.lower()}-fold-{fold}",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": compute_sha256(checkpoint),
                    "history_sha256": "h" * 64,
                }
            )
        registry = pd.DataFrame(registry_rows)
        packs.append(
            SimpleNamespace(
                candidate=candidate,
                experiment_id=experiment_id,
                seed=2753,
                oof=oof,
                registry=registry,
            )
        )
        calibrated = oof.drop(columns=["run_id"]).copy()
        calibrated.insert(0, "seed", 2753)
        calibrated.insert(0, "experiment_id", experiment_id)
        calibrated.insert(0, "candidate", candidate)
        calibrated_parts.append(calibrated)
    calibrated_path = root / "calibrated.csv"
    pd.concat(calibrated_parts, ignore_index=True).to_csv(calibrated_path, index=False)

    split_path = _write(root / "data/splits.csv", "placeholder\n")
    label_path = root / "data/labels.json"
    label_path.write_text(
        json.dumps(
            {
                "season": {
                    "classes": list(SEASON_LABELS),
                    "label_to_index": {label: index for index, label in enumerate(SEASON_LABELS)},
                },
                "articleType": {"num_classes": 124},
            }
        ),
        encoding="utf-8",
    )
    common_paths = {
        name: _write(root / f"upstream/{name}.json", "{}\n")
        for name in ("bootstrap", "calibration", "robustness", "slice", "stability")
    }
    registry_path = root / "upstream/registry.csv"
    pd.concat([pack.registry for pack in packs], ignore_index=True).to_csv(
        registry_path,
        index=False,
    )
    stability_summary_path = _write(root / "upstream/stability.csv", "candidate\nC2\nI2\n")
    slice_config = _write(root / "upstream/slice-config.json", "{}\n")
    coverage = {"rows": 24, "protected_inputs": 0}
    coverage_hash = canonical_sha256(coverage)
    prediction_artifacts = {
        f"run-{index}": {"path": "x", "sha256": "p" * 64} for index in range(20)
    }
    prediction_paths = {
        f"primary-{index}": _write(root / f"upstream/pred-{index}.csv", "id\n1\n")
        for index in range(10)
    }

    monkeypatch.setattr(evidence_module, "load_gradcam_review_spec", lambda *a, **k: spec)
    monkeypatch.setattr(
        evidence_module,
        "load_verified_bootstrap_manifest",
        lambda *a, **k: (
            {
                "stability_coverage_sha256": coverage_hash,
                "slice_assignment_sha256": "s" * 64,
            },
            common_paths["bootstrap"],
            {
                "calibration_manifest": common_paths["calibration"],
                "slice_manifest": common_paths["slice"],
                "stability_manifest": common_paths["stability"],
                "canonical_inputs": {"splits": split_path, "label_maps": label_path},
            },
        ),
    )
    monkeypatch.setattr(
        evidence_module,
        "load_verified_calibration_manifest",
        lambda *a, **k: (
            {"stability_coverage_sha256": coverage_hash},
            common_paths["calibration"],
            {
                "robustness_manifest": common_paths["robustness"],
                "slice_manifest": common_paths["slice"],
                "canonical_inputs": {"splits": split_path, "label_maps": label_path},
                "temporary_calibrated_oof": calibrated_path,
                "input_predictions": prediction_paths,
            },
        ),
    )
    monkeypatch.setattr(
        evidence_module,
        "load_verified_robustness_manifest",
        lambda *a, **k: ({}, common_paths["robustness"], {}),
    )
    monkeypatch.setattr(
        evidence_module,
        "load_verified_slice_manifest",
        lambda *a, **k: (
            {"coverage_sha256": coverage_hash, "input_predictions": prediction_artifacts},
            common_paths["slice"],
            {
                "stability_manifest": common_paths["stability"],
                "canonical_inputs": {"splits": split_path, "label_maps": label_path},
                "artifacts": {"registry_snapshot": registry_path},
                "analysis_config": slice_config,
            },
        ),
    )
    monkeypatch.setattr(
        evidence_module,
        "load_verified_stability_manifest",
        lambda *a, **k: (
            {"coverage_sha256": coverage_hash},
            common_paths["stability"],
            {
                "artifacts": {"seed_stability": stability_summary_path},
                "input_configs": {},
            },
        ),
    )
    monkeypatch.setattr(evidence_module, "load_splits", lambda *_a, **_k: splits)
    monkeypatch.setattr(
        evidence_module,
        "get_samples",
        lambda *_a, **_k: development.copy(),
    )
    monkeypatch.setattr(evidence_module, "load_declared_config_hashes", lambda *_a, **_k: {})
    monkeypatch.setattr(evidence_module, "load_slice_analysis_spec", lambda *_a, **_k: object())
    monkeypatch.setattr(
        evidence_module,
        "load_candidate_oof_packs",
        lambda *_a, **_k: (
            packs,
            {pack.experiment_id: coverage for pack in packs},
            prediction_artifacts,
        ),
    )
    monkeypatch.setattr(
        evidence_module,
        "build_slice_assignments",
        lambda *_a, **_k: SimpleNamespace(
            assignments=assignments,
            assignment_sha256="s" * 64,
        ),
    )
    monkeypatch.setattr(
        evidence_module,
        "capture_git_state",
        lambda *_a, **_k: {"commit": "g" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        evidence_module,
        "verify_implementation_at_head",
        lambda *_a, **_k: ["src/fashion/task2/gradcam.py"],
    )
    monkeypatch.setattr(evidence_module, "implementation_sha256", lambda *_a, **_k: "i" * 64)
    monkeypatch.setattr(evidence_module, "capture_runtime", lambda: {"python": "3.12"})
    frames = {
        fold: (
            development.loc[~development["cv_fold"].eq(fold)].copy(),
            development.loc[development["cv_fold"].eq(fold)].copy(),
        )
        for fold in range(5)
    }
    monkeypatch.setattr(evidence_module, "canonical_validation_frames", lambda *_a: frames)

    def fake_stats(registry_row: dict[str, object], **_kwargs: object) -> FoldImageStats:
        fold = int(registry_row["fold"])
        training = frames[fold][0]
        return FoldImageStats(
            validation_fold=fold,
            image_size=(80, 60),
            image_count=len(training),
            content_pixel_count=1,
            mean=(0.5, 0.5, 0.5),
            std=(0.2, 0.2, 0.2),
            training_id_sha256="t" * 64,
        )

    monkeypatch.setattr(evidence_module, "fold_stats_from_history", fake_stats)
    monkeypatch.setattr(
        evidence_module,
        "build_image_transform",
        lambda *_a, **_k: lambda _path: torch.zeros((3, 80, 60), dtype=torch.float32),
    )
    monkeypatch.setattr(evidence_module, "build_robustness_model", lambda *_a, **_k: object())

    class FakeModel:
        def to(self, *_args: object, **_kwargs: object) -> FakeModel:
            return self

        def float(self) -> FakeModel:
            return self

    monkeypatch.setattr(
        evidence_module,
        "load_robustness_checkpoint",
        lambda _model, row, **_k: (
            FakeModel(),
            {
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "checkpoint_bytes": 10,
            },
            Path(str(row["checkpoint_path"])),
        ),
    )

    def fake_gradcam(
        _model: object,
        _image: torch.Tensor,
        *,
        candidate: str,
        target_index: int,
    ) -> GradCamComputation:
        probabilities = np.full(4, 0.01 / 3, dtype=np.float32)
        probabilities[target_index] = 0.99
        return GradCamComputation(
            heatmap=np.linspace(0.0, 1.0, 4800, dtype=np.float32).reshape(80, 60),
            logits=np.log(probabilities),
            probabilities=probabilities,
            target_index=target_index,
            activation_shape=(1, 8, 5, 4),
            gradient_shape=(1, 8, 5, 4),
            zero_heatmap=False,
        )

    monkeypatch.setattr(evidence_module, "compute_gradcam", fake_gradcam)

    def fake_plot(
        _selected: pd.DataFrame,
        _overlays: object,
        output_path: Path,
        **_kwargs: object,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"deterministic-png")
        return output_path

    monkeypatch.setattr(evidence_module, "plot_gradcam_contact_sheet", fake_plot)
    kwargs = {
        "project_root": root,
        "analysis_config_path": root / "config.json",
        "bootstrap_manifest_path": common_paths["bootstrap"],
        "evidence_directory": root / "results/evidence",
        "figure_directory": root / "results/figures",
        "temporary_directory": root / "tmp/gradcam",
    }
    _write(root / "config.json", "{}\n")

    first = build_gradcam_failure_evidence(**kwargs)
    second = build_gradcam_failure_evidence(**kwargs)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["holdout_opened"] is False
    assert first["causal_failure_claim_allowed"] is False
    assert first["ultimate_winner_frozen"] is False
    selected = pd.read_csv(root / "results/evidence/selected_examples.csv")
    reviewed = pd.read_csv(root / "results/evidence/attention_metrics.csv")
    failures = pd.read_csv(root / "results/evidence/failure_taxonomy.csv")
    assert len(selected) == 48
    assert len(reviewed) == 48
    assert len(failures) == 24
    assert 9999 not in set(selected["id"])
    assert Path(first["temporary_heatmaps"]["path"]).name == "heatmaps.npy"
