"""Stability-evidence coverage must accept valid undefined queries and stay strict.

Protocol A nDCG is undefined for a query whose ``articleType`` is absent from the
gallery, so ``quality_summary.query_count`` is the number of *scorable* queries. Every
canonical development fold has at least one such query, which is why the lightweight
stability route must compare recorded counts against the frozen evaluator's recomputed
scorable count instead of the total canonical query count.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import fashion.task4.learned_evidence as learned
import fashion.task4.training as training
from fashion.data.splits import cv_assignment_digest
from fashion.task4.cache import DevelopmentImageCache
from fashion.task4.experiments import StabilityEvidenceInput
from fashion.task4.training import (
    AugmentationPolicy,
    CandidateConfig,
    SourcePolicy,
    TrainingHyperparameters,
    TrainingSessionConfig,
)
from fashion.train.registry import (
    TASK4_RUN_COLUMNS as RUN_COLUMNS,
)
from fashion.train.registry import (
    Task4RunRegistry as RunRegistry,
)
from tests.task4.test_learned_evidence import (  # reuse the frozen canonical fixtures
    CONTRACT,
    _splits,
    _statistics,
    _write_cache_files,
)

UNDEFINED_QUERY_ID = 100
PARENT_RUN_ID = "task4-candidate-r3-fixture"


def _parent_session(splits: pd.DataFrame) -> TrainingSessionConfig:
    return TrainingSessionConfig(
        run_id=PARENT_RUN_ID,
        run_kind="candidate",
        candidate=CandidateConfig("R1", "resnet18"),
        hyperparameters=TrainingHyperparameters(
            warmup_epochs=1,
            planned_epochs=2,
            checkpoint_epochs=(2,),
        ),
        objective="vicreg",
        source_policy=SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=AugmentationPolicy.NONE,
        validation_fold=1,
        split_fingerprint=cv_assignment_digest(splits),
    )


def _stability_session(splits: pd.DataFrame, *, fold: int = 1) -> TrainingSessionConfig:
    return TrainingSessionConfig(
        run_id=f"{PARENT_RUN_ID}-stability-fixture-fold-{fold}",
        run_kind="stability",
        candidate=CandidateConfig("R1", "resnet18"),
        hyperparameters=TrainingHyperparameters(
            warmup_epochs=1,
            planned_epochs=2,
            checkpoint_epochs=(2,),
        ),
        objective="vicreg",
        source_policy=SourcePolicy.TEACHER_V1_PAIRS,
        augmentation_policy=AugmentationPolicy.NONE,
        validation_fold=fold,
        split_fingerprint=cv_assignment_digest(splits),
        parent_run_id=PARENT_RUN_ID,
    )


def _running_stability_row(session: TrainingSessionConfig) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in RUN_COLUMNS}
    row.update(
        {
            "schema_version": "1",
            "started_at_utc": "2026-08-31T01:02:03Z",
            "status": "running",
            "git_commit": "c" * 40,
            "dirty_tree": False,
            **session.expected_registry_identity.as_dict(),
        }
    )
    return row


def _undefined_query_splits(fold: int = 1) -> pd.DataFrame:
    splits = _splits(fold)
    query = splits["id"].eq(UNDEFINED_QUERY_ID)
    splits.loc[query, ["articleType", "baseColour"]] = ["UndefinedType", "Orange"]
    splits.loc[query, "product_family_group"] = "singleton-family"
    return splits


class _StabilityFixture:
    """Everything ``build_stability_evidence`` needs, wired to a temporary root."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        splits: pd.DataFrame,
        fold: int = 1,
    ) -> None:
        self.splits = splits
        self.session = _stability_session(splits, fold=fold)
        model = self.session.build_cpu_model()
        optimizer = training.build_optimizer(model, self.session.hyperparameters)
        scheduler = training.WarmupCosineScheduler(
            optimizer,
            steps_per_epoch=1,
            config=self.session.hyperparameters,
        )
        checkpoint = training.save_checkpoint(
            tmp_path / "stability-epoch-2.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=training.make_grad_scaler("cpu"),
            epoch=2,
            session=self.session,
            score=0.5,
        )
        del model, optimizer, scheduler
        self.registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
        self.registry.append(_running_stability_row(_parent_session(splits)))
        self.registry.append(_running_stability_row(self.session))
        self.registry = RunRegistry(tmp_path / "runs.csv", project_root=tmp_path)
        registry_row = next(
            row for row in self.registry.read() if row["run_id"] == self.session.run_id
        )
        self.result = learned.reconstruct_training_result(
            [checkpoint.path],
            session=self.session,
            registry_row=registry_row,
        )
        development_ids = np.sort(
            splits.loc[splits["partition"].eq("development"), "id"].to_numpy(
                dtype=np.int64
            )
        )
        self.caches: dict[str, DevelopmentImageCache] = {}
        self.statistics: dict[str, dict[str, object]] = {}
        self.statistics_paths: dict[str, Path] = {}
        for source, colour in (("teacher", 64), ("v1", 192)):
            images = np.full(
                (len(development_ids), 320, 240, 3), colour, dtype=np.uint8
            )
            bounds = np.tile(
                np.array([[1, 1, 319, 239]], dtype=np.int32), (len(images), 1)
            )
            cache_dir = tmp_path / "cache" / source
            cache_dir.mkdir(parents=True)
            cache = DevelopmentImageCache(
                cache_dir,
                development_ids,
                images,
                bounds,
                {
                    "scope": "development",
                    "source": source,
                    "source_fingerprint": hashlib.sha256(source.encode()).hexdigest(),
                    "contract": CONTRACT.to_dict(),
                    "array_shape": list(images.shape),
                },
            )
            _write_cache_files(cache)
            self.caches[source] = cache
            self.statistics[source] = _statistics(cache, fold=fold)
            self.statistics[source]["split_fingerprint"] = self.session.split_fingerprint
            statistics_path = tmp_path / f"{source}-statistics.json"
            statistics_path.write_bytes(
                learned._canonical_json(self.statistics[source])
            )
            self.statistics_paths[source] = statistics_path
        monkeypatch.setattr(learned, "ROOT", tmp_path)
        self.evidence_root = tmp_path / "results/evidence/task4"
        self.feature_cache_root = tmp_path / "features"

    def build(self) -> learned.StabilityEvidenceResult:
        return learned.build_stability_evidence(
            self.registry,
            result=self.result,
            session=self.session,
            splits=self.splits,
            caches=self.caches,
            statistics=self.statistics,
            statistics_paths=self.statistics_paths,
            feature_cache_root=self.feature_cache_root,
            evidence_root=self.evidence_root,
            completed_at="2026-08-31T04:00:00Z",
        )


def _quality_for(fixture: _StabilityFixture) -> learned.LearnedQualityEvaluation:
    indexes = {
        source: learned.ensure_learned_feature_index(
            cache=fixture.caches[source],
            statistics=fixture.statistics[source],
            session=fixture.session,
            checkpoint=fixture.result.best_checkpoint,
            cache_root=fixture.feature_cache_root,
        ).index
        for source in ("teacher", "v1")
    }
    return learned.evaluate_learned_quality(
        fixture.splits,
        indexes,
        fold=fixture.session.validation_fold,
    )


def _views(fixture: _StabilityFixture) -> tuple[Any, Any]:
    return learned.build_development_views(
        fixture.splits,
        validation_fold=fixture.session.validation_fold,
    )


def test_stability_evidence_completes_a_fold_with_valid_undefined_primary_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    primary, _ = _views(fixture)
    total = len(primary.queries)

    built = fixture.build()

    payload = json.loads(Path(built.manifest_path).read_text(encoding="utf-8"))
    assert payload["coverage"] == {
        "total_query_count": total,
        "scorable_query_count": total - 1,
        "primary_coverage": (total - 1) / total,
    }
    assert built.primary_coverage == pytest.approx((total - 1) / total)
    assert built.registry_row["status"] == "completed"


def test_stability_coverage_counts_match_the_frozen_evaluator_scorable_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    quality = _quality_for(fixture)
    primary, family = _views(fixture)
    recorded = pd.to_numeric(
        quality.summary.loc[
            quality.summary["protocol"].eq("primary")
            & quality.summary["metric"].eq("ndcg")
            & pd.to_numeric(quality.summary["k"], errors="coerce").eq(10)
            & quality.summary["aggregation"].eq("query_mean"),
            "query_count",
        ]
    )

    coverage = learned._stability_primary_coverage(quality, primary, family)

    assert set(recorded) == {len(primary.queries) - 1}
    assert coverage["total_query_count"] == len(primary.queries)
    assert coverage["scorable_query_count"] == len(primary.queries) - 1


def test_stability_coverage_rejects_counts_that_disagree_with_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    quality = _quality_for(fixture)
    primary, family = _views(fixture)
    forged = quality.summary.copy()
    selected = (
        forged["protocol"].eq("primary")
        & forged["metric"].eq("ndcg")
        & pd.to_numeric(forged["k"], errors="coerce").eq(10)
        & forged["aggregation"].eq("query_mean")
    )
    forged.loc[selected, "query_count"] = len(primary.queries) - 2

    with pytest.raises(ValueError, match="primary coverage is incomplete"):
        learned._stability_primary_coverage(
            learned.LearnedQualityEvaluation(
                summary=forged,
                query_metrics=quality.query_metrics,
                pair_evaluations=quality.pair_evaluations,
                selected_metrics=quality.selected_metrics,
                provenance=quality.provenance,
            ),
            primary,
            family,
        )


def test_stability_coverage_rejects_null_counts_and_missing_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    quality = _quality_for(fixture)
    primary, family = _views(fixture)
    selected = (
        quality.summary["protocol"].eq("primary")
        & quality.summary["metric"].eq("ndcg")
        & pd.to_numeric(quality.summary["k"], errors="coerce").eq(10)
        & quality.summary["aggregation"].eq("query_mean")
    )

    nulled = quality.summary.copy()
    nulled.loc[selected, "query_count"] = np.nan
    with pytest.raises(ValueError, match="primary coverage is incomplete"):
        learned._stability_primary_coverage(
            learned.LearnedQualityEvaluation(
                summary=nulled,
                query_metrics=quality.query_metrics,
                pair_evaluations=quality.pair_evaluations,
                selected_metrics=quality.selected_metrics,
                provenance=quality.provenance,
            ),
            primary,
            family,
        )

    dropped = quality.summary.loc[
        ~(
            selected
            & quality.summary["query_source"].eq("v1")
            & quality.summary["gallery_source"].eq("teacher")
        )
    ]
    with pytest.raises(ValueError, match="four primary nDCG@10 rows"):
        learned._stability_primary_coverage(
            learned.LearnedQualityEvaluation(
                summary=dropped,
                query_metrics=quality.query_metrics,
                pair_evaluations=quality.pair_evaluations,
                selected_metrics=quality.selected_metrics,
                provenance=quality.provenance,
            ),
            primary,
            family,
        )


def test_stability_artifact_validation_recomputes_the_recorded_coverage_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    built = fixture.build()
    manifest_path = Path(built.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    reopened = learned.validate_stability_evidence_artifact(
        manifest_path,
        session=fixture.session,
        checkpoint=fixture.result.best_checkpoint,
        canonical_splits=fixture.splits,
    )
    assert reopened["coverage"] == payload["coverage"]

    for field, value in (
        ("scorable_query_count", payload["coverage"]["scorable_query_count"] - 1),
        ("total_query_count", payload["coverage"]["total_query_count"] + 1),
        ("primary_coverage", 1.0),
        ("primary_coverage", 0.0),
    ):
        tampered = json.loads(json.dumps(payload))
        tampered["coverage"][field] = value
        manifest_path.write_bytes(learned._canonical_json(tampered))
        with pytest.raises(ValueError, match="coverage"):
            learned.validate_stability_evidence_artifact(
                manifest_path,
                session=fixture.session,
                checkpoint=fixture.result.best_checkpoint,
                canonical_splits=fixture.splits,
            )


def test_task7_stability_input_accepts_honest_partial_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _StabilityFixture(
        tmp_path, monkeypatch, splits=_undefined_query_splits()
    )
    built = fixture.build()

    validated = StabilityEvidenceInput(
        artifact_path=Path(built.manifest_path),
        session=fixture.session,
        checkpoint=fixture.result.best_checkpoint,
        canonical_splits=fixture.splits,
    ).validated()

    primary, _ = _views(fixture)
    assert validated.total_query_count == len(primary.queries)
    assert validated.scorable_query_count == len(primary.queries) - 1
    assert validated.primary_coverage == pytest.approx(
        (len(primary.queries) - 1) / len(primary.queries)
    )
