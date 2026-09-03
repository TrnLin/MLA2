from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import fashion.task4 as task4
import fashion.task4.learned_evidence as learned
from fashion.data.splits import cv_assignment_digest
from fashion.task4.models import B1_WEIGHT_ORIGIN, SCRATCH_WEIGHT_ORIGIN

experiments = importlib.import_module("fashion.task4.experiments")


class _TimingTinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        values = torch.zeros((len(images), 128), dtype=torch.float32, device=images.device)
        values[:, :3] = images.mean(dim=(2, 3))
        values[:, 3] = self.anchor
        return torch.nn.functional.normalize(values, dim=1)


def _task6_frames_for_session(
    fixtures: object,
    splits: pd.DataFrame,
    session: task4.TrainingSessionConfig,
    checkpoint: task4.CheckpointRecord,
    query_ids: list[int],
) -> dict[str, pd.DataFrame]:
    fold = session.validation_fold
    ids = splits["id"].to_numpy(dtype=np.int64)
    features = np.zeros((len(ids), 128), dtype=np.float32)
    features[:, 0] = 1.0
    indexes = {
        source: fixtures._index(source, ids, features, fold=fold) for source in ("teacher", "v1")
    }
    quality = learned.evaluate_learned_quality(splits, indexes, fold=fold)
    primary_views, family_views = learned.build_development_views(
        splits,
        validation_fold=fold,
    )
    analysis = learned.evaluate_learned_analysis(
        quality,
        primary_views=primary_views,
        family_views=family_views,
        canvas_indexes={"wide": indexes["v1"], "tall": indexes["v1"]},
        gallery_index=indexes["v1"],
        fold=fold,
    )
    gallery = learned.evaluate_gallery_sources(splits, indexes, fold=fold)
    timing = pd.DataFrame.from_records(
        [
            {
                "scope": "development",
                "fold": fold,
                "query_id": query_id,
                "query_source": query,
                "gallery_source": gallery_source,
                "encoding_seconds": 0.1,
                "search_seconds": 0.1,
                "end_to_end_seconds": 0.2,
            }
            for query, gallery_source in learned.source_directions()
            for query_id in query_ids
        ]
    )
    return {
        "quality_summary": quality.summary.assign(
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
        ),
        "query_metrics": quality.query_metrics.assign(
            checkpoint_fingerprint=checkpoint.sha256,
            config_fingerprint=session.config_hash,
        ),
        "rankings": learned.assemble_learned_rankings(quality),
        "failure_slices": analysis.failure_slices,
        "canvas_summary": analysis.canvas_summary,
        "canvas_per_query": analysis.canvas_per_query,
        "canvas_rankings": learned.assemble_canvas_rankings(analysis),
        "examples": learned.assemble_learned_examples(analysis, quality),
        "timing_samples": timing,
        "timing_summary": pd.concat(
            [
                learned.summarize_timings(rows)
                for _, rows in timing.groupby(
                    ["query_source", "gallery_source"],
                    sort=False,
                )
            ],
            ignore_index=True,
        ),
        "gallery_comparison": gallery.comparison,
        "gallery_rankings": learned.assemble_gallery_rankings(gallery),
    }


def _real_task6_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session: task4.TrainingSessionConfig | None = None,
    selected_gallery_policy: str = "two_view",
) -> tuple[Path, object, object, pd.DataFrame]:
    module_name = "_task7_task6_test_fixtures"
    fixtures = sys.modules.get(module_name)
    if fixtures is None:
        fixture_path = Path(__file__).with_name("test_learned_evidence.py")
        spec = importlib.util.spec_from_file_location(module_name, fixture_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Task 6 test fixtures could not be loaded")
        fixtures = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = fixtures
        spec.loader.exec_module(fixtures)
    splits = pd.DataFrame(
        [
            fixtures._split_row(1000 + fold * 100 + offset, fold)
            for fold in range(5)
            for offset in range(11)
        ]
    )
    selected_session = session or fixtures._session(
        split_fingerprint=fixtures.cv_assignment_digest(splits),
    )
    work_dir = tmp_path / f"fixture-{selected_session.run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fixtures._checkpoint(work_dir, selected_session)
    monkeypatch.setattr(learned, "ROOT", tmp_path)
    source_artifacts, provenance = fixtures._source_artifact_fixture(
        work_dir,
        splits,
        selected_session,
        checkpoint,
    )
    query_ids = (
        splits.loc[
            splits["cv_fold"].eq(selected_session.validation_fold),
            "id",
        ]
        .astype(int)
        .tolist()
    )
    fixtures._patch_artifact_cost_builder(
        monkeypatch,
        checkpoint,
        selected_session,
        len(query_ids),
        provenance,
    )
    fixture_cost_builder = learned.build_learned_cost_record

    def build_cost(*args: object, **kwargs: object) -> dict[str, object]:
        record = fixture_cost_builder(*args, **kwargs)
        record.update(
            {
                "run_id": selected_session.run_id,
                "run_kind": selected_session.run_kind,
                "method": selected_session.expected_registry_identity.method,
                "fold": selected_session.validation_fold,
                "checkpoint_sha256": checkpoint.sha256,
                "config_hash": selected_session.config_hash,
                "split_fingerprint": selected_session.split_fingerprint,
            }
        )
        rows = len(splits)
        payload_bytes = rows * 128 * np.dtype(np.float32).itemsize
        index_bytes = payload_bytes + rows * np.dtype(np.int64).itemsize
        for source in ("teacher", "v1"):
            source_record = record["per_source_index_cost"][source]
            source_record["rows"] = rows
            source_record["payload_bytes"] = payload_bytes
            source_record["index_bytes"] = index_bytes
            record["feature_bytes"][source] = payload_bytes
            record["index_bytes"][source] = index_bytes
        record["selected_gallery_policy"] = selected_gallery_policy
        record["selected_policy_total_index_bytes"] = (
            index_bytes * 2 if selected_gallery_policy == "two_view" else index_bytes
        )
        del record["measurement_sha256"]
        record["measurement_sha256"] = hashlib.sha256(learned._canonical_json(record)).hexdigest()
        return record

    monkeypatch.setattr(learned, "build_learned_cost_record", build_cost)
    frames = _task6_frames_for_session(
        fixtures,
        splits,
        selected_session,
        checkpoint,
        query_ids,
    )
    for frame in frames.values():
        if "fold" in frame:
            frame["fold"] = selected_session.validation_fold
        if "method" in frame:
            frame["method"] = selected_session.expected_registry_identity.method
    manifest_path = learned.write_learned_artifacts(
        evidence_dir=tmp_path / "results/evidence/task4/learned" / selected_session.run_id,
        run_id=selected_session.run_id,
        run_kind=selected_session.run_kind,
        checkpoint=checkpoint,
        session=selected_session,
        canonical_splits=splits,
        frames=frames,
        source_artifacts=source_artifacts,
        selected_gallery_policy=selected_gallery_policy,
    )
    return manifest_path, selected_session, checkpoint, splits


def test_runner_task6_input_converts_loaded_checkpoint_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    runner_path = Path(__file__).parents[2] / "scripts/task4/run_model_comparisons.py"
    spec = importlib.util.spec_from_file_location("task4_checkpoint_boundary_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    loaded = type(
        "Loaded",
        (),
        {
            "epoch": checkpoint.epoch,
            "score": checkpoint.score,
            "sha256": checkpoint.sha256,
        },
    )()
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: loaded)

    evidence_input = runner._task6_input_from_request(
        {
            "run_id": session.run_id,
            "checkpoint_path": checkpoint.path,
            "manifest_path": manifest_path,
        },
        artifacts={session.run_id: artifact},
        splits=splits,
    )

    assert isinstance(evidence_input.checkpoint, task4.CheckpointRecord)
    assert evidence_input.validated().identity.run_id == session.run_id


def test_deployment_input_preserves_equivalent_registry_checkpoint_path_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    runner_path = Path(__file__).parents[2] / "scripts/task4/run_model_comparisons.py"
    spec = importlib.util.spec_from_file_location("task4_path_identity_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    config_path = runner.write_experiment_config_artifact(
        tmp_path / "experiment_config.json",
        session=session,
        checkpoint_sha256=checkpoint.sha256,
    )
    loaded = type(
        "Loaded",
        (),
        {
            "epoch": checkpoint.epoch,
            "score": checkpoint.score,
            "sha256": checkpoint.sha256,
        },
    )()
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        runner,
        "_stability_input_from_request",
        lambda *args, **kwargs: object(),
    )
    candidate_row = {
        "run_id": session.run_id,
        "checkpoint_path": str(checkpoint.path),
    }
    deployment_spec = {
        "registry_row": candidate_row,
        "candidate_score": 0.5,
        "finalist_config_artifact_path": config_path,
        "stability_rows": [{} for _ in range(5)],
        "stability_manifests": [{"run_id": f"fold-{fold}"} for fold in range(5)],
        "candidate_manifest": {
            "run_id": session.run_id,
            "checkpoint_path": checkpoint.path.relative_to(tmp_path),
            "manifest_path": manifest_path,
        },
    }

    deployment_input, _ = runner._deployment_inputs_from_request(
        {"deployment_inputs": [deployment_spec, deployment_spec]},
        artifacts={session.run_id: artifact},
        splits=splits,
    )

    assert (
        deployment_input.candidate_evidence_manifest.validated().identity.run_id
        == session.run_id
    )
    assert deployment_input.candidate_evidence_manifest.checkpoint.path == checkpoint.path


def _row(
    method: str,
    score: float,
    *,
    run_id: str | None = None,
    status: str = "completed",
    run_kind: str = "candidate",
    fold: int = 1,
    pretrained: bool = False,
    deployment_eligibility: str = "eligible",
    weight_origin: str = SCRATCH_WEIGHT_ORIGIN,
    source_robustness_ratio: float = 0.9,
    p95_end_to_end_seconds: float = 0.2,
    index_bytes: int = 100,
    parent_run_id: str = "",
    architecture: str | None = None,
) -> dict[str, object]:
    selected_architecture = architecture or ("resnet34" if method == "R2" else "resnet18")
    objective = {
        "R1": "vicreg",
        "R2": "vicreg",
        "R3": "vicreg",
        "R4": "vicreg_triplet",
        "R5": "content_mask_mse",
        "B1": "vicreg",
    }[method]
    return {
        "run_id": run_id or f"{method.lower()}-{run_kind}-{fold}",
        "parent_run_id": parent_run_id,
        "status": status,
        "run_kind": run_kind,
        "fold": fold,
        "method": method,
        "architecture": selected_architecture,
        "objective": objective,
        "development_winner_score": score,
        "pretrained": pretrained,
        "deployment_eligibility": deployment_eligibility,
        "weight_origin": weight_origin,
        "source_robustness_ratio": source_robustness_ratio,
        "p95_end_to_end_seconds": p95_end_to_end_seconds,
        "index_bytes": index_bytes,
    }


def _quality_summary(
    *,
    teacher_teacher: float,
    v1_v1: float,
    teacher_v1: float = 0.4,
    v1_teacher: float = 0.6,
) -> pd.DataFrame:
    values = {
        ("teacher", "teacher"): teacher_teacher,
        ("v1", "v1"): v1_v1,
        ("teacher", "v1"): teacher_v1,
        ("v1", "teacher"): v1_teacher,
    }
    return pd.DataFrame(
        [
            {
                "protocol": "primary",
                "metric": "ndcg",
                "k": 10,
                "aggregation": "query_mean",
                "query_source": query_source,
                "gallery_source": gallery_source,
                "value": value,
            }
            for (query_source, gallery_source), value in values.items()
        ]
    )


def _changed_signature_fields(
    left: experiments.ExperimentConfig,
    right: experiments.ExperimentConfig,
) -> set[str]:
    return {
        field
        for field, left_value, right_value in zip(
            experiments.EXPERIMENT_FACTOR_FIELDS,
            left.factor_signature,
            right.factor_signature,
            strict=True,
        )
        if left_value != right_value
    }


def test_package_exports_stable_experiment_interfaces() -> None:
    assert task4.ExperimentConfig is experiments.ExperimentConfig
    assert task4.ExperimentConfigArtifact is experiments.ExperimentConfigArtifact
    assert task4.ValidatedTask6Evidence is experiments.ValidatedTask6Evidence
    assert task4.select_stability_finalists is experiments.select_stability_finalists
    assert task4.select_deployment_candidate is experiments.select_deployment_candidate
    assert task4.select_gallery_source is experiments.select_gallery_source


def test_winner_score_reuses_the_frozen_equal_same_source_metric() -> None:
    summary = _quality_summary(teacher_teacher=0.2, v1_v1=0.8)

    assert experiments.development_winner_score(summary) == pytest.approx(0.5)


def test_incremental_configs_change_one_declared_factor_and_reuse_training_configs() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.6, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.7, checkpoint_sha256="2" * 64
    )
    artifacts = {"r1": r1_artifact, "r2": r2_artifact}
    r3 = experiments.derive_r3_config(
        [r1_row, r2_row],
        config_artifacts=artifacts,
    )
    r3_row, r3_artifact = _persisted_candidate(
        r3, run_id="r3", score=0.8, checkpoint_sha256="3" * 64
    )
    artifacts["r3"] = r3_artifact
    r4 = experiments.derive_r4_config(
        [r1_row, r2_row, r3_row],
        config_artifacts=artifacts,
    )

    assert _changed_signature_fields(experiments.R1_CONFIG, experiments.R2_CONFIG) == {
        "architecture"
    }
    assert _changed_signature_fields(experiments.R2_CONFIG, r3) == {"augmentation_policy"}
    assert _changed_signature_fields(r3, r4) == {"objective"}
    assert _changed_signature_fields(experiments.R1_CONFIG, experiments.B1_CONFIG) == {
        "weight_origin"
    }
    assert r3.parent_run_id == "r2"
    assert r4.parent_run_id == "r3"
    assert experiments.R5_CONFIG.parent_run_id is None
    assert experiments.B1_CONFIG.run_kind == "benchmark"
    assert experiments.B1_CONFIG.comparison_only
    assert experiments.B1_CONFIG.weight_origin == B1_WEIGHT_ORIGIN

    session = r4.training_session(
        run_id="r4-run",
        split_fingerprint="a" * 64,
    )
    assert isinstance(session, task4.TrainingSessionConfig)
    assert session.candidate is r4.candidate
    assert session.hyperparameters is r4.hyperparameters
    assert session.parent_run_id == "r3"
    assert session.validation_fold == 1
    assert session.model_metadata.weight_origin == SCRATCH_WEIGHT_ORIGIN

    with pytest.raises(FrozenInstanceError):
        experiments.R1_CONFIG.parent_run_id = "changed"


def test_resolved_matrix_contains_all_six_configs_and_cannot_be_mutated() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.6, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.7, checkpoint_sha256="2" * 64
    )
    artifacts = {"r1": r1_artifact, "r2": r2_artifact}
    r3 = experiments.derive_r3_config(
        [r1_row, r2_row],
        config_artifacts=artifacts,
    )
    r3_row, r3_artifact = _persisted_candidate(
        r3, run_id="r3", score=0.8, checkpoint_sha256="3" * 64
    )
    artifacts["r3"] = r3_artifact

    matrix = experiments.build_experiment_matrix(
        [r1_row, r2_row, r3_row],
        config_artifacts=artifacts,
    )

    assert tuple(matrix) == ("R1", "R2", "R3", "R4", "R5", "B1")
    assert matrix["R3"].parent_run_id == "r2"
    assert matrix["R4"].parent_run_id == "r3"
    with pytest.raises(TypeError):
        matrix["R1"] = experiments.R2_CONFIG


def test_parent_selection_uses_exact_score_then_declared_candidate_order() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.7, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.7, checkpoint_sha256="2" * 64
    )
    artifacts = {"r1": r1_artifact, "r2": r2_artifact}
    r3 = experiments.derive_r3_config(
        [r1_row, r2_row],
        config_artifacts=artifacts,
    )
    r3_row, r3_artifact = _persisted_candidate(
        r3, run_id="r3", score=0.7, checkpoint_sha256="3" * 64
    )
    artifacts["r3"] = r3_artifact
    rows = [r2_row, r1_row, r3_row]

    assert (
        experiments.select_parent(rows, config_artifacts=artifacts, candidates=("R1", "R2"))[
            "run_id"
        ]
        == "r1"
    )
    assert (
        experiments.select_parent(rows, config_artifacts=artifacts, candidates=("R1", "R2", "R3"))[
            "run_id"
        ]
        == "r1"
    )


def test_pretrained_incomplete_and_ineligible_runs_cannot_be_stability_finalists() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.8, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.8, checkpoint_sha256="2" * 64
    )
    rows = [
        r1_row,
        r2_row,
        _row("R3", 0.7, status="failed"),
        _row("R4", 0.99, deployment_eligibility="comparison_only"),
        _row(
            "B1",
            1.0,
            run_kind="benchmark",
            pretrained=True,
            deployment_eligibility="comparison_only",
            weight_origin=B1_WEIGHT_ORIGIN,
        ),
        _row("R5", 0.6, run_kind="smoke"),
    ]

    finalists = experiments.select_stability_finalists(
        rows,
        config_artifacts={"r1": r1_artifact, "r2": r2_artifact},
    )

    assert tuple(finalist.method for finalist in finalists) == ("R1", "R2")
    assert tuple(finalist.candidate_run_id for finalist in finalists) == ("r1", "r2")
    assert all(finalist.config_artifact is not None for finalist in finalists)


def test_r4_stability_config_inherits_the_selected_parents_geometry_state() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.9, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.7, checkpoint_sha256="2" * 64
    )
    artifacts = {"r1": r1_artifact, "r2": r2_artifact}
    r3_config = experiments.derive_r3_config([r1_row, r2_row], config_artifacts=artifacts)
    r3_row, r3_artifact = _persisted_candidate(
        r3_config, run_id="r3", score=0.8, checkpoint_sha256="3" * 64
    )
    artifacts["r3"] = r3_artifact
    plain = experiments.derive_r4_config([r1_row, r2_row, r3_row], config_artifacts=artifacts)

    r3_row["development_winner_score"] = 1.0
    geometry = experiments.derive_r4_config([r1_row, r2_row, r3_row], config_artifacts=artifacts)

    assert plain.augmentation_policy is task4.AugmentationPolicy.NONE
    assert geometry.augmentation_policy is task4.AugmentationPolicy.GEOMETRY


def test_stability_plan_requires_five_fresh_folds_and_never_reuses_candidate_run() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.7, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG,
        run_id="candidate-r2",
        score=0.8,
        checkpoint_sha256="2" * 64,
    )
    finalist = experiments.select_stability_finalists(
        [r1_row, r2_row],
        config_artifacts={"r1": r1_artifact, "candidate-r2": r2_artifact},
    )[0]

    plan = experiments.build_stability_plan(finalist, attempt_token="first")

    assert tuple(run.fold for run in plan) == (0, 1, 2, 3, 4)
    assert all(run.run_kind == "stability" for run in plan)
    assert all(run.parent_run_id == "candidate-r2" for run in plan)
    assert all(run.run_id != "candidate-r2" for run in plan)
    assert len({run.run_id for run in plan}) == 5


def test_selected_stability_plan_builds_the_existing_training_session_contract() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG, run_id="r1", score=0.7, checkpoint_sha256="1" * 64
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG, run_id="r2", score=0.8, checkpoint_sha256="2" * 64
    )
    finalists = experiments.select_stability_finalists(
        [r1_row, r2_row],
        config_artifacts={"r1": r1_artifact, "r2": r2_artifact},
    )
    run = experiments.build_stability_plan(finalists[0], attempt_token="first")[3]

    session = run.training_session()

    assert isinstance(session, task4.TrainingSessionConfig)
    assert session.run_kind == "stability"
    assert session.validation_fold == 3
    assert session.parent_run_id == "r2"
    assert session.candidate.candidate == "R2"
    assert session.hyperparameters is experiments.R2_CONFIG.hyperparameters


def test_stability_aggregation_uses_validated_five_fold_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session, checkpoint, _ = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    finalist = experiments.StabilityFinalist(artifact.identity, 0.8, artifact)
    summary = _aggregated_stability(finalist, tmp_path, monkeypatch)

    assert summary.method == "R1"
    assert len(summary.run_ids) == 5
    assert summary.mean == pytest.approx(1.0)
    assert summary.standard_deviation == pytest.approx(0.0)


def test_pooled_spread_uses_both_finalist_sample_standard_deviations() -> None:
    assert experiments.pooled_spread(0.2, 0.4) == pytest.approx(math.sqrt(0.1))


def _deployment(
    method: str,
    *,
    mean: float,
    standard_deviation: float,
    source_robustness_ratio: float = 0.9,
    wide: float = 0.7,
    tall: float = 0.7,
    p95: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2),
    indexes: tuple[int, ...] = (100, 100),
    pretrained: bool = False,
    weight_origin: str = SCRATCH_WEIGHT_ORIGIN,
) -> experiments.DeploymentEvidence:
    identity = experiments.EvidenceIdentity(
        f"{method.lower()}-run",
        method,
        1,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    return experiments.DeploymentEvidence(
        identity=identity,
        method=method,
        stability_mean=mean,
        stability_standard_deviation=standard_deviation,
        source_robustness_ratio=source_robustness_ratio,
        wide_canvas_ndcg_at_10=wide,
        tall_canvas_ndcg_at_10=tall,
        direction_p95_end_to_end_seconds=tuple(
            zip(
                (
                    ("teacher", "teacher"),
                    ("teacher", "v1"),
                    ("v1", "teacher"),
                    ("v1", "v1"),
                ),
                p95,
                strict=True,
            )
        ),
        source_index_bytes=tuple(zip(("teacher", "v1"), indexes, strict=True)),
        cpu_measurement=experiments.CPUMeasurementIdentity(
            route="spawned_batch_one_checkpoint_extraction_v1",
            measurement_sha256="d" * 64,
            cpu="test cpu",
            operating_system="test os",
            thread_count=1,
        ),
        pretrained=pretrained,
        weight_origin=weight_origin,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        _deployment("B1", mean=0.9, standard_deviation=0.1, pretrained=True),
        _deployment(
            "R1",
            mean=0.9,
            standard_deviation=0.1,
            weight_origin=B1_WEIGHT_ORIGIN,
        ),
        _deployment(
            "R1",
            mean=0.9,
            standard_deviation=0.1,
            p95=(0.2, 0.2, 0.2, 1.0),
        ),
        _deployment(
            "R1",
            mean=0.9,
            standard_deviation=0.1,
            indexes=(100, 1024**3),
        ),
    ],
)
def test_deployment_cost_and_scratch_gates_are_strict(
    candidate: experiments.DeploymentEvidence,
) -> None:
    assert not experiments.passes_deployment_gates(candidate)


def test_mean_gap_above_pooled_spread_beats_practical_tie_breakers() -> None:
    lower = _deployment(
        "R1",
        mean=0.70,
        standard_deviation=0.01,
        source_robustness_ratio=0.99,
        wide=0.99,
        tall=0.99,
        p95=(0.01,) * 4,
        indexes=(1, 1),
    )
    higher = _deployment(
        "R2",
        mean=0.72,
        standard_deviation=0.01,
        source_robustness_ratio=0.5,
        wide=0.5,
        tall=0.5,
        p95=(0.9,) * 4,
        indexes=(1000, 1000),
    )

    assert experiments._select_deployment_evidence((lower, higher)).method == "R2"


def test_deployment_inputs_are_extracted_from_registry_and_evidence_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    task6 = experiments.ValidatedTask6Evidence.from_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    row = {
        **session.expected_registry_identity.as_dict(),
        "status": "completed",
        "development_winner_score": 0.8,
        "checkpoint_sha256": checkpoint.sha256,
        "source_robustness_ratio": 0.95,
    }
    finalist = experiments.StabilityFinalist(
        identity=artifact.identity,
        candidate_score=0.8,
        config_artifact=artifact,
    )
    stability = _aggregated_stability(finalist, tmp_path, monkeypatch)

    evidence = experiments.deployment_evidence_from_artifacts(
        row,
        stability,
        task6,
    )

    assert evidence.stability_mean == pytest.approx(1.0)
    assert evidence.source_robustness_ratio == task6.source_robustness_ratio
    assert evidence.source_robustness_ratio != 0.95
    assert evidence.wide_canvas_ndcg_at_10 == dict(task6.canvas_ndcg_at_10)["wide"]
    assert evidence.tall_canvas_ndcg_at_10 == dict(task6.canvas_ndcg_at_10)["tall"]
    assert evidence.direction_p95_end_to_end_seconds == task6.direction_p95_end_to_end_seconds
    assert evidence.source_index_bytes == task6.source_index_bytes
    assert experiments.passes_deployment_gates(evidence)


@pytest.mark.parametrize(
    ("left_overrides", "right_overrides", "winner"),
    [
        ({"source_robustness_ratio": 0.91}, {}, "R1"),
        (
            {"source_robustness_ratio": 0.9, "wide": 0.8, "tall": 0.8},
            {"source_robustness_ratio": 0.9},
            "R1",
        ),
        (
            {"source_robustness_ratio": 0.9, "p95": (0.1,) * 4},
            {"source_robustness_ratio": 0.9, "p95": (0.2,) * 4},
            "R1",
        ),
        (
            {"source_robustness_ratio": 0.9, "indexes": (50, 50)},
            {"source_robustness_ratio": 0.9, "indexes": (100, 100)},
            "R1",
        ),
    ],
)
def test_practical_stability_ties_follow_the_frozen_lexicographic_order(
    left_overrides: dict[str, object],
    right_overrides: dict[str, object],
    winner: str,
) -> None:
    common = {"mean": 0.7, "standard_deviation": 0.1}
    left = _deployment("R1", **common, **left_overrides)
    right = _deployment("R2", **common, **right_overrides)

    assert experiments._select_deployment_evidence((left, right)).method == winner


def test_gallery_quality_ties_prefer_smaller_then_faster_and_two_view_uses_v1() -> None:
    options = (
        experiments.GalleryOption("teacher", 0.800000, 100, 0.20),
        experiments.GalleryOption("v1", 0.800009, 90, 0.30),
        experiments.GalleryOption("two_view", 0.800008, 90, 0.10),
    )

    selected = experiments._select_gallery_options(options)

    assert selected.policy == "two_view"
    assert experiments.query_normalization_source(selected.policy) == "v1"
    assert experiments.query_normalization_source("teacher") == "teacher"


def test_gallery_options_read_validated_quality_and_derive_policy_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    evidence_input = experiments.Task6ManifestInput(
        manifest_path,
        session,
        checkpoint,
        splits,
    )
    task6 = evidence_input.validated()
    _stub_gallery_policy_timing_writer(
        monkeypatch,
        values=(0.3, 0.2, 0.4),
    )
    practical = experiments.derive_gallery_practical_evidence(
        evidence_input,
        policy_timing_output_path=tmp_path / "gallery-timing.json",
    )

    options = experiments.gallery_options_from_evidence(task6, practical=practical)
    quality = dict(task6.gallery_quality_at_10)
    storage = dict(task6.source_index_bytes)

    assert [(option.policy, option.quality, option.index_bytes) for option in options] == [
        ("teacher", quality["teacher"], storage["teacher"]),
        ("v1", quality["v1"], storage["v1"]),
        ("two_view", quality["two_view"], storage["teacher"] + storage["v1"]),
    ]
    assert [option.p95_end_to_end_seconds for option in options] == pytest.approx((0.3, 0.2, 0.4))


def _persisted_candidate(
    config: experiments.ExperimentConfig,
    *,
    run_id: str,
    score: float,
    checkpoint_sha256: str,
) -> tuple[dict[str, object], object]:
    session = config.training_session(run_id=run_id, split_fingerprint="a" * 64)
    row = {
        **session.expected_registry_identity.as_dict(),
        "status": "completed",
        "development_winner_score": score,
        "checkpoint_sha256": checkpoint_sha256,
        "source_robustness_ratio": 0.9,
    }
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint_sha256,
    )
    return row, artifact


def _selection_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> experiments.ValidatedTask6Evidence:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    return experiments.ValidatedTask6Evidence.from_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )


def _write_generated_gallery_timing(
    path: Path,
    evidence_input: experiments.Task6ManifestInput,
    *,
    values: tuple[float, float, float] = (0.3, 0.2, 0.4),
    warmup_queries: int = 0,
    timed_queries: int = 22,
) -> Path:
    samples = [
        {
            "scope": "development",
            "fold": evidence_input.session.validation_fold,
            "query_id": query_id,
            "query_source": query_source,
            "gallery_policy": policy,
            "encoding_seconds": value / 2,
            "search_seconds": value / 2,
            "end_to_end_seconds": value,
        }
        for policy, value in zip(
            ("teacher", "v1", "two_view"),
            values,
            strict=True,
        )
        for query_source in ("teacher", "v1")
        for query_id in range(1, timed_queries // 2 + 1)
    ]
    summary = [
        {
            "gallery_policy": policy,
            "metric": metric,
            "percentile": percentile,
            "value_seconds": metric_value,
            "timed_queries": timed_queries,
            "measurement_kind": "measured",
        }
        for policy, value in zip(
            ("teacher", "v1", "two_view"),
            values,
            strict=True,
        )
        for metric, metric_value in (
            ("encoding", value / 2),
            ("search", value / 2),
            ("end_to_end", value),
        )
        for percentile in ("p50", "p95")
    ]
    record = {
        "schema_version": 1,
        "scope": "development",
        "run_id": evidence_input.session.run_id,
        "run_kind": evidence_input.session.run_kind,
        "method": evidence_input.session.expected_registry_identity.method,
        "fold": evidence_input.session.validation_fold,
        "checkpoint_sha256": evidence_input.checkpoint.sha256,
        "config_hash": evidence_input.session.config_hash,
        "split_fingerprint": evidence_input.session.split_fingerprint,
        "measurement_route": learned.GALLERY_POLICY_TIMING_ROUTE,
        "measurement_kind": "measured",
        "hardware": {
            "cpu": "test cpu",
            "logical_cores": 1,
            "operating_system": "test os",
            "python_version": "test python",
            "numpy_version": np.__version__,
            "thread_count": 1,
            "thread_environment": {name: "1" for name in learned.THREAD_VARIABLES},
            "native_thread_pools": [
                {
                    "user_api": "blas",
                    "internal_api": "test",
                    "num_threads": 1,
                    "prefix": "test",
                }
            ],
        },
        "warmup_queries": warmup_queries,
        "timed_queries": timed_queries,
        "timing_samples": samples,
        "timing_summary": summary,
    }
    record["measurement_sha256"] = hashlib.sha256(learned._canonical_json(record)).hexdigest()
    path.write_bytes(learned._canonical_json(record))
    return path


def _stub_gallery_policy_timing_writer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    values: tuple[float, float, float],
) -> None:
    def write(
        destination: str | Path,
        *,
        manifest_path: str | Path,
        session: task4.TrainingSessionConfig,
        checkpoint: task4.CheckpointRecord,
        canonical_splits: pd.DataFrame,
        **_: object,
    ) -> Path:
        evidence_input = experiments.Task6ManifestInput(
            manifest_path=Path(manifest_path),
            session=session,
            checkpoint=checkpoint,
            canonical_splits=canonical_splits,
        )
        return _write_generated_gallery_timing(
            Path(destination),
            evidence_input,
            values=values,
        )

    monkeypatch.setattr(learned, "write_gallery_policy_timing_artifact", write)


def test_gallery_policy_timing_producer_runs_actual_policy_search_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    query_ids = sorted(
        splits.loc[splits["cv_fold"].eq(session.validation_fold), "id"].astype(int).tolist()
    )
    single_view_calls: list[tuple[str, int]] = []
    two_view_calls: list[tuple[str, int, int]] = []
    real_build_search = learned.build_protocol_a_search
    real_two_view = learned.rank_two_view_gallery

    def build_search(**kwargs: object):
        gallery_index = kwargs["gallery_index"]
        search = real_build_search(**kwargs)

        def wrapped(query_id: int, feature: np.ndarray) -> pd.DataFrame:
            single_view_calls.append((gallery_index.source, query_id))
            return search(query_id, feature)

        return wrapped

    def rank_two_view(**kwargs: object) -> pd.DataFrame:
        query_index = kwargs["query_index"]
        two_view_calls.append(
            (
                query_index.source,
                int(query_index.ids[0]),
                len(query_index.ids),
            )
        )
        return real_two_view(**kwargs)

    monkeypatch.setattr(learned, "build_protocol_a_search", build_search)
    monkeypatch.setattr(learned, "rank_two_view_gallery", rank_two_view)
    monkeypatch.setattr(learned, "_load_evidence_model", lambda *args: _TimingTinyEncoder())

    artifact_path = learned.write_gallery_policy_timing_artifact(
        tmp_path / "gallery-policy-timing.json",
        manifest_path=manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
        policy=learned.TimingPolicy(warmup_queries=1),
        clock_ns=iter(range(10_000)).__next__,
    )

    record = json.loads(artifact_path.read_text(encoding="utf-8"))
    validated = learned.validate_gallery_policy_timing_artifact(
        artifact_path,
        session=session,
        checkpoint=checkpoint,
    )
    calls_per_policy = (len(query_ids) + 1) * 2
    assert record["measurement_route"] == learned.GALLERY_POLICY_TIMING_ROUTE
    assert record["warmup_queries"] == 1
    assert record["timed_queries"] == len(query_ids) * 2
    assert set(validated) == {"teacher", "v1", "two_view"}
    assert [source for source, _ in single_view_calls].count("teacher") == calls_per_policy
    assert [source for source, _ in single_view_calls].count("v1") == calls_per_policy
    assert len(two_view_calls) == calls_per_policy
    assert {source for source, _, _ in two_view_calls} == {"teacher", "v1"}
    assert {width for _, _, width in two_view_calls} == {1}
    assert sorted({query_id for _, query_id in single_view_calls}) == query_ids
    assert sorted({query_id for _, query_id, _ in two_view_calls}) == query_ids


def test_gallery_policy_timing_producer_revalidates_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    destination = tmp_path / "gallery-policy-timing.json"
    original = b'{"existing":"artifact"}\n'
    destination.write_bytes(original)
    validated_paths: list[Path] = []

    def reject_generated_artifact(
        artifact_path: str | Path,
        *,
        session: task4.TrainingSessionConfig,
        checkpoint: task4.CheckpointRecord,
    ) -> dict[str, float]:
        validated_paths.append(Path(artifact_path))
        raise ValueError("forced staging validation failure")

    monkeypatch.setattr(
        learned,
        "validate_gallery_policy_timing_artifact",
        reject_generated_artifact,
    )
    monkeypatch.setattr(learned, "_load_evidence_model", lambda *args: _TimingTinyEncoder())

    with pytest.raises(ValueError, match="forced staging validation failure"):
        learned.write_gallery_policy_timing_artifact(
            destination,
            manifest_path=manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
            policy=learned.TimingPolicy(warmup_queries=0),
            clock_ns=iter(range(10_000)).__next__,
        )

    assert destination.read_bytes() == original
    assert validated_paths
    assert validated_paths[-1] != destination


def _aggregated_stability(
    finalist: experiments.StabilityFinalist,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> experiments.StabilitySummary:
    plan = experiments.build_stability_plan(finalist, attempt_token="summary")
    rows = []
    artifacts = {}
    evidence_manifests = {}
    for fold, run in enumerate(plan):
        session = run.training_session()
        manifest_path, _, checkpoint, splits = _real_task6_package(
            tmp_path,
            monkeypatch,
            session=session,
        )
        checkpoint_sha256 = checkpoint.sha256
        rows.append(
            {
                **session.expected_registry_identity.as_dict(),
                "status": "completed",
                "development_winner_score": 0.5 + fold * 0.1,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
        artifacts[run.run_id] = experiments.ExperimentConfigArtifact.from_session(
            session,
            checkpoint_sha256=checkpoint_sha256,
        )
        evidence_manifests[run.run_id] = experiments.Task6ManifestInput(
            manifest_path=manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )
    return experiments.summarize_stability(
        rows,
        finalist=finalist,
        config_artifacts=artifacts,
        evidence_manifests=evidence_manifests,
    )


def test_finalist_rejects_stale_parent_and_extra_factor_config_artifacts() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG,
        run_id="r1",
        score=0.6,
        checkpoint_sha256="1" * 64,
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG,
        run_id="r2",
        score=0.7,
        checkpoint_sha256="2" * 64,
    )
    expected_r3 = experiments.derive_r3_config(
        [r1_row, r2_row],
        config_artifacts={"r1": r1_artifact, "r2": r2_artifact},
    )
    stale_r3 = replace(expected_r3, parent_run_id="r1")
    stale_row, stale_artifact = _persisted_candidate(
        stale_r3,
        run_id="r3",
        score=0.9,
        checkpoint_sha256="3" * 64,
    )
    artifacts = {"r1": r1_artifact, "r2": r2_artifact, "r3": stale_artifact}

    with pytest.raises(ValueError, match="parent|chain"):
        experiments.select_stability_finalists(
            [r1_row, r2_row, stale_row],
            config_artifacts=artifacts,
        )

    extra_factor = replace(expected_r3, objective="vicreg_triplet")
    with pytest.raises(ValueError, match="config|factor|objective"):
        _persisted_candidate(
            extra_factor,
            run_id="r3-extra",
            score=0.9,
            checkpoint_sha256="4" * 64,
        )


def _deployment_decision_input(
    manifest_path: Path,
    session: task4.TrainingSessionConfig,
    checkpoint: task4.CheckpointRecord,
    splits: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> experiments.DeploymentDecisionInput:
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    finalist = experiments.StabilityFinalist(artifact.identity, 0.8, artifact)
    rows = []
    artifacts = {}
    manifests = {}
    for run in experiments.build_stability_plan(finalist, attempt_token="decision"):
        fold_session = run.training_session()
        fold_manifest, _, fold_checkpoint, fold_splits = _real_task6_package(
            tmp_path,
            monkeypatch,
            session=fold_session,
        )
        rows.append(
            {
                **fold_session.expected_registry_identity.as_dict(),
                "status": "completed",
                "development_winner_score": 999.0,
                "checkpoint_sha256": fold_checkpoint.sha256,
            }
        )
        artifacts[run.run_id] = experiments.ExperimentConfigArtifact.from_session(
            fold_session,
            checkpoint_sha256=fold_checkpoint.sha256,
        )
        manifests[run.run_id] = experiments.Task6ManifestInput(
            fold_manifest,
            fold_session,
            fold_checkpoint,
            fold_splits,
        )
    return experiments.DeploymentDecisionInput(
        registry_row={
            **session.expected_registry_identity.as_dict(),
            "status": "completed",
            "checkpoint_sha256": checkpoint.sha256,
            "source_robustness_ratio": 999.0,
        },
        finalist=finalist,
        stability_rows=tuple(rows),
        stability_config_artifacts=artifacts,
        stability_evidence_manifests=manifests,
        candidate_evidence_manifest=experiments.Task6ManifestInput(
            manifest_path,
            session,
            checkpoint,
            splits,
        ),
    )


def test_stability_rejects_cross_parent_or_config_and_requires_lineage() -> None:
    r1_row, r1_artifact = _persisted_candidate(
        experiments.R1_CONFIG,
        run_id="r1",
        score=0.8,
        checkpoint_sha256="1" * 64,
    )
    r2_row, r2_artifact = _persisted_candidate(
        experiments.R2_CONFIG,
        run_id="r2",
        score=0.7,
        checkpoint_sha256="2" * 64,
    )
    finalist = experiments.select_stability_finalists(
        [r1_row, r2_row],
        config_artifacts={"r1": r1_artifact, "r2": r2_artifact},
    )[0]
    plan = experiments.build_stability_plan(finalist, attempt_token="first")
    rows = []
    artifacts = {}
    for index, run in enumerate(plan):
        session = run.training_session()
        checkpoint_sha256 = str(index + 3) * 64
        rows.append(
            {
                **session.expected_registry_identity.as_dict(),
                "status": "completed",
                "development_winner_score": 0.5 + index * 0.1,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
        artifacts[run.run_id] = experiments.ExperimentConfigArtifact.from_session(
            session,
            checkpoint_sha256=checkpoint_sha256,
        )
    rows[0]["parent_run_id"] = "other-finalist"

    with pytest.raises(ValueError, match="parent|lineage"):
        experiments.summarize_stability(
            rows,
            finalist=finalist,
            config_artifacts=artifacts,
            evidence_manifests={},
        )

    rows[0]["parent_run_id"] = finalist.identity.run_id
    artifacts[plan[0].run_id] = replace(
        artifacts[plan[0].run_id],
        candidate_config_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="config"):
        experiments.summarize_stability(
            rows,
            finalist=finalist,
            config_artifacts=artifacts,
            evidence_manifests={},
        )


def test_stability_retry_ids_are_unique_and_keep_finalist_parent() -> None:
    row, artifact = _persisted_candidate(
        experiments.R1_CONFIG,
        run_id="r1",
        score=0.8,
        checkpoint_sha256="1" * 64,
    )
    other_row, other_artifact = _persisted_candidate(
        experiments.R2_CONFIG,
        run_id="r2",
        score=0.7,
        checkpoint_sha256="2" * 64,
    )
    finalist = experiments.select_stability_finalists(
        [row, other_row],
        config_artifacts={"r1": artifact, "r2": other_artifact},
    )[0]

    first = experiments.build_stability_plan(finalist, attempt_token="attempt-1")
    retry = experiments.build_stability_plan(finalist, attempt_token="attempt-2")

    assert {run.run_id for run in first}.isdisjoint(run.run_id for run in retry)
    assert {run.parent_run_id for run in (*first, *retry)} == {"r1"}


@pytest.mark.parametrize(
    ("directions", "indexes", "message"),
    [
        (
            (
                (("teacher", "teacher"), 0.1),
                (("teacher", "teacher"), 0.2),
                (("v1", "teacher"), 0.3),
                (("v1", "v1"), 0.4),
            ),
            (("teacher", 40), ("v1", 60)),
            "direction",
        ),
        (
            (
                (("teacher", "teacher"), 0.1),
                (("teacher", "v1"), 0.2),
                (("v1", "teacher"), 0.3),
            ),
            (("teacher", 40), ("v1", 60)),
            "direction",
        ),
        (
            (
                (("teacher", "teacher"), 0.1),
                (("teacher", "v1"), 0.2),
                (("v1", "teacher"), 0.3),
                (("v1", "v1"), 0.4),
            ),
            (("teacher", 40),),
            "index",
        ),
        (
            (
                (("teacher", "teacher"), 0.1),
                (("teacher", "v1"), 0.2),
                (("v1", "teacher"), 0.3),
                (("v1", "v1"), 0.4),
            ),
            (("teacher", 40), ("v1", 60), ("other", 1)),
            "index",
        ),
    ],
)
def test_validated_evidence_rejects_duplicate_missing_or_extra_cost_labels(
    directions: tuple[tuple[tuple[str, str], float], ...],
    indexes: tuple[tuple[str, int], ...],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cost_record = next(record for record in manifest["artifacts"] if record["name"] == "cost")
    cost_path = manifest_path.parent / cost_record["path"]
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    timing = [
        record
        for record in cost["timing_summary"]
        if not (record["metric"] == "end_to_end" and record["percentile"] == "p95")
    ]
    timing.extend(
        {
            "query_source": direction[0],
            "gallery_source": direction[1],
            "metric": "end_to_end",
            "percentile": "p95",
            "value_seconds": value,
            "unit": "seconds",
            "timed_queries": cost["timed_queries"],
        }
        for direction, value in directions
    )
    cost["timing_summary"] = timing
    cost["index_bytes"] = dict(indexes)
    del cost["measurement_sha256"]
    cost["measurement_sha256"] = hashlib.sha256(learned._canonical_json(cost)).hexdigest()
    cost_path.write_bytes(learned._canonical_json(cost))
    cost_record["sha256"] = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{message}|timing|index"):
        experiments.ValidatedTask6Evidence.from_manifest(
            manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_task6_selection_evidence_cannot_bypass_validated_artifact_factory() -> None:
    identity = experiments.EvidenceIdentity("run", "R1", 1, "a" * 64, "b" * 64, "c" * 64)

    with pytest.raises(TypeError, match="validated"):
        experiments.ValidatedTask6Evidence(
            identity=identity,
            canvas_ndcg_at_10=(("wide", 0.7), ("tall", 0.6)),
            direction_p95_end_to_end_seconds=(
                (("teacher", "teacher"), 0.1),
                (("teacher", "v1"), 0.2),
                (("v1", "teacher"), 0.3),
                (("v1", "v1"), 0.4),
            ),
            source_index_bytes=(("teacher", 40), ("v1", 60)),
            cpu_measurement=experiments.CPUMeasurementIdentity(
                route="spawned_batch_one_checkpoint_extraction_v1",
                measurement_sha256="f" * 64,
                cpu="test cpu",
                operating_system="test os",
                thread_count=1,
            ),
            development_winner_score=0.75,
            source_robustness_ratio=0.9,
            gallery_quality_at_10=(("teacher", 0.7), ("v1", 0.8), ("two_view", 0.9)),
        )


def test_validated_manifest_factory_reopens_identity_bound_selection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )

    reopened = experiments.ValidatedTask6Evidence.from_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )

    assert reopened.identity.run_id == session.run_id
    assert reopened.identity.checkpoint_sha256 == checkpoint.sha256
    assert dict(reopened.source_index_bytes).keys() == {"teacher", "v1"}


def test_deployment_rejects_valid_task6_evidence_from_another_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    evidence = experiments.ValidatedTask6Evidence.from_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )
    other_session = replace(session, run_id="registry-run")
    artifact = experiments.ExperimentConfigArtifact.from_session(
        other_session,
        checkpoint_sha256=checkpoint.sha256,
    )
    row = {
        **other_session.expected_registry_identity.as_dict(),
        "status": "completed",
        "development_winner_score": 0.8,
        "checkpoint_sha256": checkpoint.sha256,
        "source_robustness_ratio": 0.9,
    }
    finalist = experiments.StabilityFinalist(
        identity=artifact.identity,
        candidate_score=0.8,
        config_artifact=artifact,
    )
    stability = _aggregated_stability(finalist, tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="identity|run"):
        experiments.deployment_evidence_from_artifacts(row, stability, evidence)


def test_gallery_requires_validated_exact_ndcg_at_10_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    evidence_input = experiments.Task6ManifestInput(
        manifest_path,
        session,
        checkpoint,
        splits,
    )
    evidence = evidence_input.validated()
    _stub_gallery_policy_timing_writer(
        monkeypatch,
        values=(0.3, 0.2, 0.4),
    )
    practical = experiments.derive_gallery_practical_evidence(
        evidence_input,
        policy_timing_output_path=tmp_path / "gallery-timing.json",
    )
    options = experiments.gallery_options_from_evidence(
        evidence,
        practical=practical,
    )

    assert [option.quality for option in options] == [
        value for _, value in evidence.gallery_quality_at_10
    ]
    assert experiments.select_gallery_source(
        evidence_input,
        policy_timing_output_path=tmp_path / "gallery-selection-timing.json",
    ) == experiments._select_gallery_options(options)


def test_caller_authored_validated_artifact_factory_is_not_public() -> None:
    assert not hasattr(experiments.ValidatedTask6Evidence, "from_validated_artifacts")


def test_trusted_factory_rejects_forged_frame_even_with_updated_recorded_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canvas_record = next(
        record for record in manifest["artifacts"] if record["name"] == "canvas_summary"
    )
    canvas_path = manifest_path.parent / canvas_record["path"]
    canvas = pd.read_csv(canvas_path)
    canvas["run_id"] = "forged-run"
    canvas.to_csv(canvas_path, index=False)
    canvas_record["sha256"] = hashlib.sha256(canvas_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance|run_id"):
        experiments.ValidatedTask6Evidence.from_manifest(
            manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_trusted_factory_rejects_post_validation_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    real_validate = learned.validate_learned_manifest

    def validate_then_mutate(*args: object, **kwargs: object) -> dict[str, object]:
        manifest = real_validate(*args, **kwargs)
        canvas_record = next(
            record for record in manifest["artifacts"] if record["name"] == "canvas_summary"
        )
        canvas_path = manifest_path.parent / canvas_record["path"]
        canvas_path.write_bytes(canvas_path.read_bytes() + b"\n")
        return manifest

    monkeypatch.setattr(learned, "validate_learned_manifest", validate_then_mutate)

    with pytest.raises(ValueError, match="hash"):
        experiments.ValidatedTask6Evidence.from_manifest(
            manifest_path,
            session=session,
            checkpoint=checkpoint,
            canonical_splits=splits,
        )


def test_stability_summary_cannot_be_constructed_outside_aggregation_factory() -> None:
    row, artifact = _persisted_candidate(
        experiments.R1_CONFIG,
        run_id="r1",
        score=0.8,
        checkpoint_sha256="1" * 64,
    )
    finalist = experiments.StabilityFinalist(
        identity=artifact.identity,
        candidate_score=float(row["development_winner_score"]),
        config_artifact=artifact,
    )
    identities = tuple(
        replace(
            artifact.identity,
            run_id=f"stability-{fold}",
            fold=fold,
            config_hash=str(fold + 2) * 64,
            checkpoint_sha256=str(fold + 3) * 64,
        )
        for fold in range(5)
    )

    with pytest.raises(TypeError, match="aggregation"):
        experiments.StabilitySummary(
            finalist=finalist,
            run_identities=identities,
            mean=0.99,
            standard_deviation=0.0,
        )


def test_gallery_practical_evidence_cannot_be_caller_authored() -> None:
    identity = experiments.EvidenceIdentity("run", "R1", 1, "a" * 64, "b" * 64, "c" * 64)

    with pytest.raises(TypeError, match="validated"):
        experiments.GalleryPracticalEvidence(
            identity=identity,
            policy_p95_end_to_end_seconds=(
                ("teacher", 0.01),
                ("v1", 0.01),
                ("two_view", 0.01),
            ),
        )


def test_stability_outcome_ignores_invented_registry_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_manifest, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    candidate_artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    finalist = experiments.StabilityFinalist(
        identity=candidate_artifact.identity,
        candidate_score=0.8,
        config_artifact=candidate_artifact,
    )
    rows = []
    config_artifacts = {}
    evidence_manifests = {}
    expected_scores = []
    for run in experiments.build_stability_plan(finalist, attempt_token="validated"):
        stability_session = run.training_session()
        manifest_path, _, fold_checkpoint, fold_splits = _real_task6_package(
            tmp_path,
            monkeypatch,
            session=stability_session,
        )
        rows.append(
            {
                **stability_session.expected_registry_identity.as_dict(),
                "status": "completed",
                "development_winner_score": 99.0,
                "checkpoint_sha256": fold_checkpoint.sha256,
            }
        )
        config_artifacts[run.run_id] = experiments.ExperimentConfigArtifact.from_session(
            stability_session,
            checkpoint_sha256=fold_checkpoint.sha256,
        )
        evidence_manifests[run.run_id] = experiments.Task6ManifestInput(
            manifest_path=manifest_path,
            session=stability_session,
            checkpoint=fold_checkpoint,
            canonical_splits=fold_splits,
        )
        expected_scores.append(
            json.loads(manifest_path.read_text(encoding="utf-8"))["selected_metrics"][
                "development_winner_score"
            ]
        )

    summary = experiments.summarize_stability(
        rows,
        finalist=finalist,
        config_artifacts=config_artifacts,
        evidence_manifests=evidence_manifests,
    )

    assert summary.mean == pytest.approx(sum(expected_scores) / 5)
    assert summary.mean != 99.0
    assert candidate_manifest.is_file()


def test_deployment_evidence_ignores_invented_registry_robustness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    evidence = experiments.ValidatedTask6Evidence.from_manifest(
        manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )
    artifact = experiments.ExperimentConfigArtifact.from_session(
        session,
        checkpoint_sha256=checkpoint.sha256,
    )
    finalist = experiments.StabilityFinalist(artifact.identity, 0.8, artifact)
    summary = _aggregated_stability(finalist, tmp_path, monkeypatch)
    row = {
        **session.expected_registry_identity.as_dict(),
        "status": "completed",
        "checkpoint_sha256": checkpoint.sha256,
        "source_robustness_ratio": 999.0,
    }

    deployment = experiments.deployment_evidence_from_artifacts(row, summary, evidence)

    assert deployment.source_robustness_ratio == evidence.source_robustness_ratio
    assert deployment.source_robustness_ratio != 999.0


def test_forged_intermediate_tokens_cannot_reach_deployment_decision() -> None:
    forged_left = _deployment("R1", mean=0.99, standard_deviation=0.0)
    forged_right = _deployment("R2", mean=0.01, standard_deviation=0.0)

    with pytest.raises((TypeError, ValueError), match="canonical|manifest|input"):
        experiments.select_deployment_candidate((forged_left, forged_right))


def test_deployment_decision_revalidates_canonical_candidate_and_fold_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_package = _real_task6_package(tmp_path, monkeypatch)
    _, _, _, splits = left_package
    right_session = experiments.R2_CONFIG.training_session(
        run_id="run-2",
        split_fingerprint=cv_assignment_digest(splits),
    )
    right_package = _real_task6_package(
        tmp_path,
        monkeypatch,
        session=right_session,
    )
    candidates = tuple(
        _deployment_decision_input(
            manifest_path,
            session,
            checkpoint,
            canonical_splits,
            tmp_path,
            monkeypatch,
        )
        for manifest_path, session, checkpoint, canonical_splits in (
            left_package,
            right_package,
        )
    )

    selected = experiments.select_deployment_candidate(candidates)

    assert selected.method in {"R1", "R2"}
    assert selected.source_robustness_ratio == pytest.approx(1.0)
    assert selected.stability_mean == pytest.approx(1.0)


def _canonical_deployment_decision_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    tuple[experiments.DeploymentDecisionInput, experiments.DeploymentDecisionInput],
    experiments.Task6ManifestInput,
]:
    left_package = _real_task6_package(
        tmp_path,
        monkeypatch,
        selected_gallery_policy="v1",
    )
    _, _, _, splits = left_package
    right_session = experiments.R2_CONFIG.training_session(
        run_id="run-2",
        split_fingerprint=cv_assignment_digest(splits),
    )
    right_package = _real_task6_package(
        tmp_path,
        monkeypatch,
        session=right_session,
    )
    candidates = tuple(
        _deployment_decision_input(
            manifest_path,
            session,
            checkpoint,
            canonical_splits,
            tmp_path,
            monkeypatch,
        )
        for manifest_path, session, checkpoint, canonical_splits in (
            left_package,
            right_package,
        )
    )
    left_manifest, left_session, left_checkpoint, left_splits = left_package
    return (
        candidates,
        experiments.Task6ManifestInput(
            left_manifest,
            left_session,
            left_checkpoint,
            left_splits,
        ),
    )


def test_post_stability_deployment_artifact_is_derived_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, _ = _canonical_deployment_decision_fixture(tmp_path, monkeypatch)
    artifact_path = experiments.write_post_stability_deployment_artifact(
        tmp_path / "deployment.json",
        candidates,
    )

    decision = experiments.validate_post_stability_deployment_artifact(
        artifact_path,
        candidates,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert decision.winner.method == "R1"
    assert payload["selected_model"]["run_id"] == decision.winner.identity.run_id
    assert payload["selected_model"]["all_deployment_gates_passed"] is True
    assert payload["holdout_opened"] is False
    assert payload["quarantine_opened"] is False
    assert payload["official_teacher_test_opened"] is False
    assert all(summary["method"] != "B1" for summary in payload["stability_summaries"])

    mutations = (
        lambda changed: changed["stability_summaries"][0].__setitem__("mean", 999.0),
        lambda changed: changed["source_artifacts"][0]["candidate_manifest"].__setitem__(
            "path", "other.json"
        ),
        lambda changed: changed["source_artifacts"][0]["candidate_manifest"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda changed: changed["selected_model"].__setitem__("run_id", "wrong-winner"),
    )
    for mutate in mutations:
        changed = json.loads(json.dumps(payload))
        mutate(changed)
        artifact_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="derived|canonical|disagree"):
            experiments.validate_post_stability_deployment_artifact(
                artifact_path,
                candidates,
            )


def test_final_gallery_decision_marks_candidate_cost_policy_as_pre_study_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, selected_manifest = _canonical_deployment_decision_fixture(
        tmp_path,
        monkeypatch,
    )
    deployment_path = experiments.write_post_stability_deployment_artifact(
        tmp_path / "deployment.json",
        candidates,
    )
    deployment = experiments.validate_post_stability_deployment_artifact(
        deployment_path,
        candidates,
    )
    timing_path = _write_generated_gallery_timing(
        tmp_path / "gallery-timing.json",
        selected_manifest,
        values=(0.1, 0.2, 0.3),
    )
    decision_path = experiments.write_final_gallery_decision_artifact(
        tmp_path / "gallery-decision.json",
        deployment=deployment,
        evidence_manifest=selected_manifest,
        timing_artifact_path=timing_path,
    )

    decision = experiments.validate_final_gallery_decision_artifact(
        decision_path,
        deployment=deployment,
        evidence_manifest=selected_manifest,
        timing_artifact_path=timing_path,
    )
    payload = json.loads(decision_path.read_text(encoding="utf-8"))

    assert payload["candidate_cost_policy"]["policy"] == "v1"
    assert payload["candidate_cost_policy"]["role"] == "pre_study_cost_assumption"
    assert payload["final_policy"]["policy"] == decision.policy
    assert payload["final_policy"]["source"] == "three_policy_development_study"

    payload["final_policy"]["policy"] = "v1" if decision.policy != "v1" else "teacher"
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="derived|canonical|disagree"):
        experiments.validate_final_gallery_decision_artifact(
            decision_path,
            deployment=deployment,
            evidence_manifest=selected_manifest,
            timing_artifact_path=timing_path,
        )

    decision_path = experiments.write_final_gallery_decision_artifact(
        decision_path,
        deployment=deployment,
        evidence_manifest=selected_manifest,
        timing_artifact_path=timing_path,
    )
    canonical_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    final_policy = canonical_decision["final_policy"]
    result_path = tmp_path / "gallery-result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "gallery",
                "phase_result": {
                    "deployment_winner": deployment.winner.identity.run_id,
                    "selected_gallery": decision.policy,
                        "selected_gallery_p95_seconds": final_policy[
                            "p95_end_to_end_seconds"
                        ],
                        "selected_gallery_index_bytes": final_policy["index_bytes"],
                },
            }
        ),
        encoding="utf-8",
    )
    experiments.link_gallery_result_to_final_decision(
        result_path,
        decision_path=decision_path,
        decision=decision,
    )
    linked = json.loads(result_path.read_text(encoding="utf-8"))
    assert linked["phase_result"]["final_decision"]["policy"] == decision.policy
    assert linked["phase_result"]["candidate_cost_policy"]["role"] == (
        "pre_study_cost_assumption"
    )

    linked["phase_result"]["selected_gallery"] = (
        "v1" if decision.policy != "v1" else "teacher"
    )
    result_path.write_text(json.dumps(linked), encoding="utf-8")
    with pytest.raises(ValueError, match="gallery|policy|disagree"):
        experiments.link_gallery_result_to_final_decision(
            result_path,
            decision_path=decision_path,
            decision=decision,
        )


def test_gallery_decision_rejects_caller_authored_timing_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, session, checkpoint, splits = _real_task6_package(
        tmp_path,
        monkeypatch,
    )
    evidence_input = experiments.Task6ManifestInput(
        manifest_path=manifest_path,
        session=session,
        checkpoint=checkpoint,
        canonical_splits=splits,
    )
    timing_path = _write_generated_gallery_timing(
        tmp_path / "caller-authored-gallery-timing.json",
        evidence_input,
        values=(0.01, 0.01, 0.01),
    )

    with pytest.raises(TypeError, match="measured_policy_timing_path"):
        experiments.select_gallery_source(
            evidence_input,
            measured_policy_timing_path=timing_path,
        )
