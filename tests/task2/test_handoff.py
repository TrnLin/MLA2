from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import fashion.task2.handoff as handoff_module
from fashion.data.hashing import compute_sha256
from fashion.task2.handoff import (
    audit_task2_artifacts,
    build_task2_handoff_evidence,
    load_verified_task2_handoff,
)
from fashion.task2.inference import SeasonPrediction
from fashion.task2.multitask import load_i2_config
from fashion.train.artifacts import (
    ArtifactVerificationError,
    canonical_sha256,
)
from fashion.train.registry import RUN_COLUMNS

PROJECT_ROOT = Path(handoff_module.__file__).resolve().parents[3]


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _declaration(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": compute_sha256(path),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    partition: str = "development",
) -> dict[str, object]:
    root = tmp_path.resolve()
    ultimate_path = _write(root / "ultimate/manifest.json", "{}\n")
    model_manifest = _write(root / "models/task2_season.manifest.json", "{}\n")
    freeze_path = _write(root / "evidence/selection_freeze.json", "{}\n")
    bundle_path = _write(root / "models/task2_season.pt", "weights")
    history_path = _write(root / "evidence/history.csv", "epoch,loss\n1,1.0\n")
    runtime_path = _write(root / "evidence/runtime.json", "{}\n")
    label_maps_path = _write(root / "data/label_maps.json", "{}\n")
    inference_source = _write(root / "src/fashion/task2/inference.py", "# image only\n")
    selected_config_path = root / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json"
    selected_config_path.parent.mkdir(parents=True, exist_ok=True)
    selected_config_path.write_text(
        (PROJECT_ROOT / "configs/task2/g4_i2_article_type_lambda_0_3_c1.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    image_path = _write(root / "data/images/1163.jpg", "fixture")
    splits_path = root / "data/splits.csv"
    pd.DataFrame(
        [
            {
                "id": "1163",
                "path": image_path.relative_to(root).as_posix(),
                "partition": partition,
                "has_season_label": True,
            }
        ]
    ).to_csv(splits_path, index=False)
    freeze_declaration = _declaration(freeze_path, root)
    ultimate = {
        "selected_candidate": "I2",
        "selected_experiment_id": "g4-i2",
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "artifacts": {"selection_freeze": freeze_declaration},
    }
    refit = {
        "selected_candidate": "I2",
        "selected_experiment_id": "g4-i2",
        "run_id": "refit-run",
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "scratch": True,
        "weights": None,
        "final_eligible": True,
        "selection_freeze": freeze_declaration,
        "bundle": _declaration(bundle_path, root),
        "artifacts": {
            "history": _declaration(history_path, root),
            "runtime": _declaration(runtime_path, root),
        },
        "canonical_inputs": {
            "splits": _declaration(splits_path, root),
            "label_maps": _declaration(label_maps_path, root),
        },
    }
    refit_rule = {"epochs": 24, "seed": 2753}
    freeze = {
        "limitations": ["brightness shift remains difficult"],
        "refit_rule": refit_rule,
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    freeze_declaration = _declaration(freeze_path, root)
    ultimate["artifacts"]["selection_freeze"] = freeze_declaration
    refit.update(
        {
            "selection_freeze": freeze_declaration,
            "selected_config": _declaration(selected_config_path, root),
            "seed": 2753,
            "final_epoch": 24,
            "model_family": "smallcnn",
            "git_commit": "a" * 40,
            "implementation_sha256": "b" * 64,
            "parameter_count": 123,
            "loader_audit": {"train_transform_id": "transform-v1"},
        }
    )
    selected_config = load_i2_config(selected_config_path)
    registry_row = {column: "" for column in RUN_COLUMNS}
    registry_row.update(
        {
            "run_id": "refit-run",
            "task": "task2",
            "stage": "development_refit",
            "experiment_id": "task2-season-i2-refit",
            "model_family": "smallcnn",
            "benchmark_only": "false",
            "final_eligible": "true",
            "scratch": "true",
            "seed": "2753",
            "git_commit": "a" * 40,
            "git_dirty": "false",
            "config_sha256": canonical_sha256(
                {
                    "selected_config": selected_config.to_dict(),
                    "refit_rule": refit_rule,
                    "selection_freeze_sha256": freeze_declaration["sha256"],
                }
            ),
            "split_sha256": refit["canonical_inputs"]["splits"]["sha256"],
            "label_map_sha256": refit["canonical_inputs"]["label_maps"]["sha256"],
            "implementation_sha256": "b" * 64,
            "transform_id": "transform-v1",
            "loss_id": selected_config.loss_id,
            "epochs_requested": "24",
            "epochs_completed": "24",
            "parameter_count": "123",
            "checkpoint_path": refit["bundle"]["path"],
            "checkpoint_sha256": refit["bundle"]["sha256"],
            "history_path": refit["artifacts"]["history"]["path"],
            "history_sha256": refit["artifacts"]["history"]["sha256"],
            "status": "completed",
        }
    )
    registry_path = root / "results/runs.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([registry_row], columns=RUN_COLUMNS).to_csv(registry_path, index=False)
    monkeypatch.setattr(
        handoff_module,
        "load_verified_ultimate_judgement_manifest",
        lambda *args, **kwargs: (
            ultimate,
            ultimate_path,
            {"selection_freeze": freeze_path},
        ),
    )
    monkeypatch.setattr(
        handoff_module,
        "load_verified_development_refit_manifest",
        lambda *args, **kwargs: (
            refit,
            model_manifest,
            {"labels": list(handoff_module.SEASON_LABELS)},
        ),
    )
    monkeypatch.setattr(
        handoff_module,
        "_load_verified_development_refit_package",
        lambda *args, **kwargs: (
            refit,
            model_manifest,
            {"labels": list(handoff_module.SEASON_LABELS)},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        handoff_module,
        "load_verified_selection_freeze",
        lambda *args, **kwargs: (freeze, freeze_path),
    )
    prediction = SeasonPrediction(
        image_path=image_path.as_posix(),
        predicted_label="Summer",
        probabilities={
            "Fall": 0.05,
            "Spring": 0.05,
            "Summer": 0.8,
            "Winter": 0.1,
        },
        confidence=0.8,
        review_required=None,
        latency_ms=12.5,
        run_id="refit-run",
        manifest_sha256=compute_sha256(model_manifest),
        bundle_sha256=compute_sha256(bundle_path),
    )
    return {
        "root": root,
        "ultimate_path": ultimate_path,
        "model_manifest": model_manifest,
        "registry_path": registry_path,
        "inference_source": inference_source,
        "prediction": prediction,
    }


def test_handoff_audits_and_packages_only_the_locked_task2_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )

    manifest, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    verified, verified_path, verified_audit, smoke = load_verified_task2_handoff(
        manifest_path,
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
    )

    assert set(audit["artifact"]) == handoff_module._REQUIRED_AUDIT_ARTIFACTS
    assert audit["status"].eq("PASS").all()
    assert manifest["status"] == "ready_for_group_freeze"
    assert manifest["task2_component_ready"] is True
    assert manifest["group_freeze_verified"] is False
    assert manifest["notebook_06_unlocked"] is False
    assert manifest["holdout_opened"] is False
    assert manifest["evaluation_claim_allowed"] is False
    assert verified == manifest
    assert verified_path == manifest_path
    assert len(verified_audit) == len(audit)
    assert "registry_snapshot" in manifest["artifacts"]
    assert (
        verified_audit.set_index("artifact").loc["registry_binding", "path"]
        == manifest["artifacts"]["registry_snapshot"]["path"]
    )
    assert smoke["input_scope"] == "one labelled development image"
    assert smoke["holdout_opened"] is False
    assert "latency_ms" not in smoke


def test_handoff_rejects_inference_from_a_different_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    changed = replace(fixture["prediction"], manifest_sha256="a" * 64)

    with pytest.raises(ValueError, match="different model manifest"):
        build_task2_handoff_evidence(
            audit,
            changed,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_rejects_a_protected_smoke_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, partition="holdout")
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )

    with pytest.raises(ValueError, match="labelled development data"):
        build_task2_handoff_evidence(
            audit,
            fixture["prediction"],
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_verifier_rejects_tampered_smoke_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    _, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    smoke_path = manifest_path.parent / "inference_smoke.json"
    smoke_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_task2_handoff(
            manifest_path,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
        )


def test_handoff_loads_from_committed_registry_snapshot_without_live_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    _, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    Path(fixture["registry_path"]).unlink()

    manifest, _, _, _ = load_verified_task2_handoff(
        manifest_path,
        project_root=fixture["root"],
    )

    assert manifest["artifacts"]["registry_snapshot"]["sha256"]


def test_handoff_rejects_rehashed_semantic_audit_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    _, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    audit_path = manifest_path.parent / "artifact_audit.csv"
    changed = pd.read_csv(audit_path, dtype="string").fillna("")
    changed.loc[changed["artifact"].eq("inference_source"), "role"] = "fabricated"
    changed.to_csv(audit_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["artifact_audit"]["sha256"] = compute_sha256(audit_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="audit.*recomputed|recomputed.*audit"):
        load_verified_task2_handoff(
            manifest_path,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
        )


def test_handoff_rejects_rehashed_semantic_smoke_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    _, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    smoke_path = manifest_path.parent / "inference_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["predicted_label"] = "Fall"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["inference_smoke"]["sha256"] = compute_sha256(smoke_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="smoke prediction changed"):
        load_verified_task2_handoff(
            manifest_path,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
        )


def test_handoff_accepts_identical_retry_and_rejects_changed_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    output = Path(fixture["root"]) / "evidence/final_handoff"
    first, first_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=output,
    )
    retried, retried_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=output,
    )
    changed = replace(
        fixture["prediction"],
        probabilities={
            "Fall": 0.08,
            "Spring": 0.07,
            "Summer": 0.75,
            "Winter": 0.10,
        },
        confidence=0.75,
    )

    assert retried == first
    assert retried_path == first_path
    with pytest.raises(ValueError, match="different content"):
        build_task2_handoff_evidence(
            audit,
            changed,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=output,
        )


def test_handoff_rejects_a_fabricated_caller_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    audit.loc[audit["artifact"].eq("inference_source"), "role"] = "fabricated"

    with pytest.raises(ValueError, match="caller audit.*recomputed"):
        build_task2_handoff_evidence(
            audit,
            fixture["prediction"],
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_serialises_builders_with_an_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    output = Path(fixture["root"]) / "evidence/final_handoff"
    output.mkdir(parents=True)
    (output / ".task2-handoff.lock").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="handoff build is already running"):
        build_task2_handoff_evidence(
            audit,
            fixture["prediction"],
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=output,
        )


@pytest.mark.parametrize(
    ("probabilities", "predicted_label", "confidence", "latency_ms"),
    [
        ({"Fall": -0.1, "Spring": 0.1, "Summer": 0.9, "Winter": 0.1}, "Summer", 0.9, 1.0),
        ({"Fall": math.nan, "Spring": 0.1, "Summer": 0.8, "Winter": 0.1}, "Summer", 0.8, 1.0),
        ({"Fall": math.inf, "Spring": 0.1, "Summer": -math.inf, "Winter": 0.1}, "Spring", 0.1, 1.0),
        ({"Fall": 0.05, "Spring": 0.05, "Summer": 0.8, "Winter": 0.1}, "Fall", 0.05, 1.0),
        ({"Fall": 0.05, "Spring": 0.05, "Summer": 0.8, "Winter": 0.1}, "Summer", 0.7, 1.0),
        ({"Fall": 0.05, "Spring": 0.05, "Summer": 0.8, "Winter": 0.1}, "Summer", 0.8, math.nan),
    ],
    ids=("negative", "nan", "infinite", "argmax", "confidence", "latency"),
)
def test_handoff_rejects_invalid_smoke_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probabilities: dict[str, float],
    predicted_label: str,
    confidence: float,
    latency_ms: float,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    invalid = replace(
        fixture["prediction"],
        probabilities=probabilities,
        predicted_label=predicted_label,
        confidence=confidence,
        latency_ms=latency_ms,
    )

    with pytest.raises(ValueError, match="smoke prediction changed"):
        build_task2_handoff_evidence(
            audit,
            invalid,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_source_has_no_final_evaluation_unlock() -> None:
    source = Path(handoff_module.__file__).read_text(encoding="utf-8")

    assert "evaluation_unlocked" not in source
    assert "load_protected" not in source
    assert '"notebook_06_unlocked": False' in source
