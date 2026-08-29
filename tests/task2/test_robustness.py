from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import fashion.task2.robustness as robustness_module
from fashion.data.hashing import compute_sha256
from fashion.data.torch import FoldImageStats
from fashion.task2.robustness import (
    CostProtocol,
    PerturbedTensorTransform,
    RobustnessCandidate,
    RobustnessCondition,
    RobustnessCostSpec,
    RobustnessProtocol,
    apply_robustness_condition,
    build_robustness_model,
    build_robustness_tables,
    fold_stats_from_history,
    load_robustness_checkpoint,
    load_robustness_cost_spec,
    measure_deployment_cost,
    model_tensor_bytes,
    predict_robustness_fold,
    reconcile_clean_probe,
    robustness_cache_key,
    run_or_load_deployment_cost,
    run_or_load_fold_probe,
)
from fashion.train.artifacts import ArtifactVerificationError, canonical_sha256
from fashion.train.metrics import SEASON_LABELS


class TinySeasonModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 80 * 60, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))


def _stats(*, fold: int = 0, image_count: int = 4, training_hash: str = "a" * 64):
    return FoldImageStats(
        validation_fold=fold,
        image_size=(80, 60),
        image_count=image_count,
        content_pixel_count=image_count * 80 * 60,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        training_id_sha256=training_hash,
    )


def _conditions() -> tuple[RobustnessCondition, ...]:
    return (
        RobustnessCondition(
            "clean",
            "none",
            source="re_inferred_and_verified_against_frozen_oof",
        ),
        RobustnessCondition("jpeg_quality_85", "jpeg_reencode", quality=85, subsampling=2),
        RobustnessCondition("brightness_0_85", "brightness", factor=0.85),
        RobustnessCondition("brightness_1_15", "brightness", factor=1.15),
        RobustnessCondition("gaussian_blur_radius_1", "gaussian_blur", radius=1.0),
    )


def _spec(*, expected_row_count: int = 20) -> RobustnessCostSpec:
    return RobustnessCostSpec(
        analysis_id="g6-robustness-cost",
        expected_row_count=expected_row_count,
        candidates=(
            RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
            RobustnessCandidate("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
        ),
        conditions=_conditions(),
        robustness=RobustnessProtocol(
            batch_size=128,
            num_workers=4,
            pin_memory=True,
            folds=tuple(range(5)),
            cache_directory="tmp/task2/robustness",
            clean_reference="verified_frozen_oof",
            clean_min_prediction_agreement=1.0,
            clean_max_probability_delta=0.0001,
            amp_matches_training_evaluation=True,
            evaluation_scope=(
                "all_valid_development_rows_once_through_their_validation_fold_checkpoint"
            ),
            perturbation_order=("decode_exif_rgb_then_perturb_then_original_fold_normalisation"),
            prediction_artifacts_are_temporary=True,
        ),
        cost=CostProtocol(
            batch_size=1,
            checkpoint_fold=0,
            cpu_threads=1,
            devices=("cpu", "cuda_if_available"),
            fixed_input_rule="lowest_id_in_checkpoint_validation_fold",
            model_only_warmups=30,
            model_only_repeats=200,
            end_to_end_warmups=10,
            end_to_end_repeats=50,
            synchronise_accelerator_each_repeat=True,
            memory_metrics=(
                "parameter_and_buffer_bytes",
                "training_checkpoint_bytes",
                "process_rss_delta_bytes",
                "peak_cuda_allocated_bytes",
            ),
        ),
        material_macro_f1_degradation=0.01,
    )


def _make_images(root: Path) -> pd.DataFrame:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for identifier, label in enumerate(SEASON_LABELS, start=1):
        array = np.zeros((80, 60, 3), dtype=np.uint8)
        array[:, :, 0] = identifier * 35
        array[:, :, 1] = np.arange(60, dtype=np.uint8)
        path = image_dir / f"{identifier}.jpg"
        Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)
        rows.append(
            {
                "id": identifier,
                "path": path.relative_to(root).as_posix(),
                "season": label,
                "partition": "development",
                "cv_fold": 0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_predictions() -> dict[tuple[str, str], pd.DataFrame]:
    outputs = {}
    targets = [SEASON_LABELS[(identifier - 1) // 5] for identifier in range(1, 21)]
    for candidate in ("C2", "I2"):
        for condition in _conditions():
            rows = []
            for identifier, truth in enumerate(targets, start=1):
                true_index = SEASON_LABELS.index(truth)
                make_error = condition.condition != "clean" and (
                    identifier % (5 if candidate == "C2" else 7) == 0
                )
                predicted_index = (true_index + 1) % 4 if make_error else true_index
                probabilities = np.full(4, 0.1)
                probabilities[predicted_index] = 0.7
                rows.append(
                    {
                        "id": identifier,
                        "fold": (identifier - 1) % 5,
                        "y_true": truth,
                        "y_pred": SEASON_LABELS[predicted_index],
                        **{
                            f"prob_{label}": probabilities[index]
                            for index, label in enumerate(SEASON_LABELS)
                        },
                    }
                )
            outputs[(candidate, condition.condition)] = pd.DataFrame(rows)
    return outputs


def test_frozen_robustness_cost_contract_loads() -> None:
    spec = load_robustness_cost_spec()

    assert spec.expected_row_count == 32_753
    assert [row.candidate for row in spec.candidates] == ["C2", "I2"]
    assert [row.condition for row in spec.conditions] == [
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    ]
    assert spec.material_macro_f1_degradation == 0.01


def test_robustness_conditions_are_deterministic_and_change_pixels() -> None:
    array = np.arange(80 * 60 * 3, dtype=np.uint8).reshape(80, 60, 3)
    image = Image.fromarray(array, mode="RGB")

    jpeg_a = np.asarray(apply_robustness_condition(image, _conditions()[1]))
    jpeg_b = np.asarray(apply_robustness_condition(image, _conditions()[1]))
    darker = np.asarray(apply_robustness_condition(image, _conditions()[2]))
    brighter = np.asarray(apply_robustness_condition(image, _conditions()[3]))
    blurred = np.asarray(apply_robustness_condition(image, _conditions()[4]))

    assert np.array_equal(jpeg_a, jpeg_b)
    assert not np.array_equal(jpeg_a, array)
    assert darker.mean() < array.mean() < brighter.mean()
    assert not np.array_equal(blurred, array)


def test_perturbed_transform_preserves_declared_tensor_shape(tmp_path: Path) -> None:
    frame = _make_images(tmp_path)
    path = tmp_path / frame.iloc[0]["path"]

    tensors = [
        PerturbedTensorTransform(stats=_stats(), condition=condition)(path)
        for condition in _conditions()
    ]

    assert all(tensor.shape == (3, 80, 60) for tensor in tensors)
    assert all(tensor.dtype == torch.float32 for tensor in tensors)
    assert torch.equal(
        tensors[1], PerturbedTensorTransform(stats=_stats(), condition=_conditions()[1])(path)
    )


def test_fold_stats_restore_exact_history_training_ids(tmp_path: Path) -> None:
    training_ids = [1, 2, 3, 4]
    stats = _stats(training_hash="")
    stats = FoldImageStats(
        **{
            **stats.to_dict(),
            "training_id_sha256": canonical_sha256(training_ids),
        }
    )
    history = {
        "run_id": "run-1",
        "experiment_id": "g3-c2-t0-resnet18",
        "fold": 0,
        "seed": 2753,
        "loader_audit": {"stats": stats.to_dict()},
    }
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history), encoding="utf-8")
    row = {
        "run_id": "run-1",
        "experiment_id": "g3-c2-t0-resnet18",
        "fold": 0,
        "seed": 2753,
        "history_path": path,
        "history_sha256": compute_sha256(path),
    }

    restored = fold_stats_from_history(
        row,
        project_root=tmp_path,
        expected_training_ids=training_ids,
    )

    assert restored == stats


def test_candidate_models_remain_scratch_and_image_only_at_inference() -> None:
    c2 = build_robustness_model("C2")
    i2 = build_robustness_model("I2", article_type_classes=124)
    images = torch.zeros(2, 3, 80, 60)

    assert c2(images).shape == (2, 4)
    assert i2.predict_season_logits(images).shape == (2, 4)


def test_checkpoint_loader_rejects_metric_drift(tmp_path: Path) -> None:
    model = build_robustness_model("I2", article_type_classes=124)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "labels": SEASON_LABELS,
        "best_epoch": 3,
        "best_macro_f1": 0.75,
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    row = {
        "checkpoint_path": path,
        "checkpoint_sha256": compute_sha256(path),
        "best_epoch": 3,
        "primary_metric_value": 0.70,
    }

    with pytest.raises(ValueError, match="metric differs"):
        load_robustness_checkpoint(model, row, project_root=tmp_path)


def test_fold_prediction_and_hash_cache_are_exact(tmp_path: Path) -> None:
    frame = _make_images(tmp_path)
    candidate = RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753)
    condition = _conditions()[2]
    model = TinySeasonModel()
    row = {
        "run_id": "checkpoint-run",
        "fold": 0,
        "checkpoint_sha256": "a" * 64,
        "history_sha256": "b" * 64,
    }
    kwargs = {
        "candidate": candidate,
        "condition": condition,
        "registry_row": row,
        "validation_frame": frame,
        "stats": _stats(),
        "label_to_index": {label: index for index, label in enumerate(SEASON_LABELS)},
        "analysis_config_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "label_map_sha256": "e" * 64,
        "implementation_sha256_value": "f" * 64,
        "cache_directory": tmp_path / "cache",
        "project_root": tmp_path,
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "device": "cpu",
        "use_amp": False,
        "git_commit": "1" * 40,
    }

    first = run_or_load_fold_probe(model, mode="run_or_load", **kwargs)
    second = run_or_load_fold_probe(model, mode="load", **kwargs)

    assert first.record["source"] == "run"
    assert second.record["source"] == "cache"
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    prediction_path = Path(second.record["prediction_path"])
    prediction_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        run_or_load_fold_probe(model, mode="load", **kwargs)


def test_cache_key_changes_with_condition() -> None:
    base = {
        "analysis_config_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "label_map_sha256": "c" * 64,
        "implementation_sha256_value": "d" * 64,
        "candidate": RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
        "fold": 0,
        "checkpoint_sha256": "e" * 64,
        "history_sha256": "f" * 64,
    }

    darker = robustness_cache_key(condition=_conditions()[2], **base)
    brighter = robustness_cache_key(condition=_conditions()[3], **base)

    assert darker != brighter


def test_robustness_tables_report_paired_degradation() -> None:
    tables = build_robustness_tables(_synthetic_predictions(), _spec())

    assert len(tables.pooled_metrics) == 10
    assert len(tables.fold_metrics) == 50
    assert len(tables.candidate_comparison) == 5
    perturbed = tables.pooled_metrics.loc[tables.pooled_metrics["condition"].ne("clean")]
    assert perturbed["delta_macro_f1_vs_clean"].lt(0).all()
    assert (
        tables.candidate_comparison.loc[
            tables.candidate_comparison["condition"].ne("clean"),
            "i2_minus_c2_macro_f1",
        ]
        .gt(0)
        .all()
    )


def test_model_storage_and_cpu_latency_probe_are_measured(tmp_path: Path) -> None:
    frame = _make_images(tmp_path)
    model = TinySeasonModel()
    protocol = CostProtocol(
        batch_size=1,
        checkpoint_fold=0,
        cpu_threads=1,
        devices=("cpu",),
        fixed_input_rule="lowest_id_in_checkpoint_validation_fold",
        model_only_warmups=0,
        model_only_repeats=2,
        end_to_end_warmups=0,
        end_to_end_repeats=2,
        synchronise_accelerator_each_repeat=True,
        memory_metrics=(
            "parameter_and_buffer_bytes",
            "training_checkpoint_bytes",
            "process_rss_delta_bytes",
            "peak_cuda_allocated_bytes",
        ),
    )

    result = measure_deployment_cost(
        model,
        candidate=RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
        checkpoint_run_id="run-1",
        checkpoint_sha256="a" * 64,
        checkpoint_bytes=123,
        transform=PerturbedTensorTransform(stats=_stats(), condition=_conditions()[0]),
        image_path=tmp_path / frame.iloc[0]["path"],
        input_id=1,
        protocol=protocol,
        requested_device="cpu",
    )

    assert result["available"] is True
    assert result["parameter_and_buffer_bytes"] == model_tensor_bytes(model)
    assert result["model_only_median_ms"] > 0
    assert result["end_to_end_median_ms"] > result["model_only_median_ms"]


def test_deployment_cost_cache_reuses_only_hash_valid_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = CostProtocol(
        batch_size=1,
        checkpoint_fold=0,
        cpu_threads=1,
        devices=("cpu",),
        fixed_input_rule="lowest_id_in_checkpoint_validation_fold",
        model_only_warmups=0,
        model_only_repeats=1,
        end_to_end_warmups=0,
        end_to_end_repeats=1,
        synchronise_accelerator_each_repeat=True,
        memory_metrics=(
            "parameter_and_buffer_bytes",
            "training_checkpoint_bytes",
            "process_rss_delta_bytes",
            "peak_cuda_allocated_bytes",
        ),
    )
    measured = {
        "candidate": "C2",
        "experiment_id": "g3-c2-t0-resnet18",
        "device": "cpu",
        "available": True,
        "model_only_median_ms": 1.25,
    }
    monkeypatch.setattr(
        robustness_module,
        "measure_deployment_cost",
        lambda *args, **kwargs: measured,
    )
    kwargs = {
        "candidate": RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
        "checkpoint_run_id": "run-1",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_bytes": 123,
        "transform": object(),
        "image_path": tmp_path / "unused.jpg",
        "input_id": 1,
        "protocol": protocol,
        "requested_device": "cpu",
        "analysis_config_sha256": "b" * 64,
        "implementation_sha256_value": "c" * 64,
        "stats_sha256": "d" * 64,
        "runtime_sha256": "e" * 64,
        "cache_directory": tmp_path / "cache",
    }

    first = run_or_load_deployment_cost(TinySeasonModel(), mode="run", **kwargs)
    monkeypatch.setattr(
        robustness_module,
        "measure_deployment_cost",
        lambda *args, **kwargs: pytest.fail("valid cost cache was not reused"),
    )
    second = run_or_load_deployment_cost(TinySeasonModel(), mode="load", **kwargs)

    assert first["source"] == "run"
    assert second["source"] == "cache"
    assert second["model_only_median_ms"] == 1.25
    result_path = Path(second["result_path"])
    result_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        run_or_load_deployment_cost(TinySeasonModel(), mode="load", **kwargs)


def test_direct_fold_prediction_uses_only_declared_validation_ids(tmp_path: Path) -> None:
    frame = _make_images(tmp_path)

    predictions, metadata = predict_robustness_fold(
        TinySeasonModel(),
        candidate=RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
        condition=_conditions()[4],
        validation_frame=frame,
        stats=_stats(),
        label_to_index={label: index for index, label in enumerate(SEASON_LABELS)},
        project_root=tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        device="cpu",
        use_amp=False,
        checkpoint_run_id="run-1",
        probe_id="probe-1",
    )

    assert set(predictions["id"]) == set(frame["id"])
    assert set(predictions["fold"]) == {0}
    assert metadata["rows"] == 4


def test_clean_condition_uses_the_same_fold_inference_pipeline(tmp_path: Path) -> None:
    frame = _make_images(tmp_path)

    predictions, metadata = predict_robustness_fold(
        TinySeasonModel(),
        candidate=RobustnessCandidate("C2", "g3-c2-t0-resnet18", 2753),
        condition=_conditions()[0],
        validation_frame=frame,
        stats=_stats(),
        label_to_index={label: index for index, label in enumerate(SEASON_LABELS)},
        project_root=tmp_path,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        device="cpu",
        use_amp=False,
        checkpoint_run_id="run-1",
        probe_id="clean-probe-1",
    )

    assert set(predictions["condition"]) == {"clean"}
    assert set(predictions["id"]) == set(frame["id"])
    assert metadata["rows"] == len(frame)
    assert load_robustness_cost_spec().conditions[0].source == (
        "re_inferred_and_verified_against_frozen_oof"
    )


def test_clean_probe_must_reproduce_frozen_oof_probabilities() -> None:
    spec = _spec()
    candidate = spec.candidates[0]
    frozen = _synthetic_predictions()[("C2", "clean")]
    probed = frozen.copy()
    probed["prob_Fall"] += 0.00001

    audit = reconcile_clean_probe(
        probed,
        frozen,
        candidate=candidate,
        protocol=spec.robustness,
    )

    assert audit["prediction_agreement"] == 1.0
    assert audit["maximum_probability_delta"] == pytest.approx(0.00001)
    probed.loc[0, "prob_Fall"] += 0.001
    with pytest.raises(ValueError, match="drift beyond"):
        reconcile_clean_probe(
            probed,
            frozen,
            candidate=candidate,
            protocol=spec.robustness,
        )
