"""Audited G6 robustness, efficiency, and machine-cost evidence for Task 2."""

from __future__ import annotations

import gc
import io
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from fashion.config import ROOT, TASK2_EVIDENCE_DIR, TASK2_FIGURE_DIR
from fashion.data.dataset import get_samples, load_splits
from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import _portable_artifact_path, _resolve_evidence_path
from fashion.task2.robustness import (
    ROBUSTNESS_CONFIG_PATH,
    ROBUSTNESS_IMPLEMENTATION_PATHS,
    PerturbedTensorTransform,
    RobustnessCandidate,
    RobustnessCostSpec,
    RobustnessTables,
    build_robustness_model,
    build_robustness_tables,
    canonical_validation_frames,
    fold_stats_from_history,
    load_robustness_checkpoint,
    load_robustness_cost_spec,
    reconcile_clean_probe,
    run_or_load_deployment_cost,
    run_or_load_fold_probe,
)
from fashion.task2.slice_evidence import (
    load_candidate_oof_packs,
    load_declared_config_hashes,
    load_verified_stability_manifest,
)
from fashion.task2.slices import load_slice_analysis_spec
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.cache import implementation_sha256
from fashion.train.metrics import SEASON_LABELS, validate_oof
from fashion.train.reproducibility import capture_git_state, capture_runtime

DEFAULT_SLICE_MANIFEST = Path("results/evidence/task2/shortcut_error_slices/manifest.json")
DEFAULT_EVIDENCE_DIRECTORY = TASK2_EVIDENCE_DIR / "robustness_cost"


def _verify_declaration(
    declaration: Mapping[str, Any],
    *,
    project_root: Path,
    name: str,
) -> Path:
    path = _resolve_evidence_path(str(declaration.get("path", "")), project_root=project_root)
    digest = str(declaration.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError(f"{name} has an invalid SHA-256 declaration")
    verify_artifact(path, digest)
    return path


def load_verified_slice_manifest(
    path: str | Path = DEFAULT_SLICE_MANIFEST,
    *,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify the complete G6 slice boundary before downstream robustness analysis."""
    root = Path(project_root)
    resolved = _resolve_evidence_path(path, project_root=root)
    if not resolved.is_file():
        raise ValueError(f"slice evidence manifest does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_identity = {
        "schema_version": "1.0.0",
        "gate": "G6-SLICE",
        "decision_status": "closed",
        "analysis_role": "development_oof_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
    }
    mismatches = [
        field for field, expected in expected_identity.items() if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"slice evidence boundary changed: {mismatches}")
    artifacts = {
        name: _verify_declaration(
            declaration,
            project_root=root,
            name=f"slice artifact {name}",
        )
        for name, declaration in dict(manifest.get("artifacts", {})).items()
    }
    required_artifacts = {"registry_snapshot", "decision", "slice_metrics"}
    missing = required_artifacts - set(artifacts)
    if missing:
        raise ValueError(f"slice evidence lacks robustness inputs: {sorted(missing)}")
    canonical_inputs = {
        name: _verify_declaration(
            declaration,
            project_root=root,
            name=f"canonical input {name}",
        )
        for name, declaration in dict(manifest.get("canonical_inputs", {})).items()
    }
    if set(canonical_inputs) != {"splits", "label_maps"}:
        raise ValueError("slice evidence changed its canonical input set")
    analysis_config = _verify_declaration(
        manifest.get("analysis_config", {}),
        project_root=root,
        name="slice analysis config",
    )
    stability_manifest = _verify_declaration(
        manifest.get("stability_manifest", {}),
        project_root=root,
        name="seed-stability manifest",
    )
    predictions = {
        run_id: _verify_declaration(
            declaration,
            project_root=root,
            name=f"slice input prediction {run_id}",
        )
        for run_id, declaration in dict(manifest.get("input_predictions", {})).items()
    }
    if len(predictions) != 20:
        raise ValueError("slice evidence must retain exactly 20 seed-stability predictions")
    with artifacts["decision"].open(encoding="utf-8") as handle:
        decision = json.load(handle)
    if (
        decision.get("current_candidate") != "I2"
        or decision.get("candidate_selection_affected") is not False
        or decision.get("ultimate_winner_frozen") is not False
    ):
        raise ValueError("slice decision no longer preserves the G5 I2 candidate")
    return (
        manifest,
        resolved,
        {
            "artifacts": artifacts,
            "canonical_inputs": canonical_inputs,
            "analysis_config": analysis_config,
            "stability_manifest": stability_manifest,
            "input_predictions": predictions,
            "decision": decision,
        },
    )


def _plot_robustness(
    tables: RobustnessTables,
    output_path: str | Path,
    *,
    material_threshold: float,
) -> Path:
    labels = {
        "clean": "Clean",
        "jpeg_quality_85": "JPEG 85",
        "brightness_0_85": "Brightness 0.85",
        "brightness_1_15": "Brightness 1.15",
        "gaussian_blur_radius_1": "Blur r=1",
    }
    order = list(labels)
    colors = {"C2": "#2F66E8", "I2": "#18A34A"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    x = np.arange(len(order))
    for candidate in ("C2", "I2"):
        subset = tables.pooled_metrics.loc[
            tables.pooled_metrics["candidate"].eq(candidate)
        ].set_index("condition")
        axes[0].plot(
            x,
            [float(subset.loc[name, "macro_f1"]) for name in order],
            marker="o" if candidate == "C2" else "s",
            linewidth=2.5,
            color=colors[candidate],
            label=candidate,
        )
        axes[1].plot(
            x,
            [float(subset.loc[name, "delta_macro_f1_vs_clean"]) for name in order],
            marker="o" if candidate == "C2" else "s",
            linewidth=2.5,
            color=colors[candidate],
            label=candidate,
        )
    for axis in axes:
        axis.set_xticks(x, [labels[name] for name in order], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Pooled five-fold OOF macro-F1")
    axes[0].set_title("Absolute robustness")
    axes[1].axhline(0, color="#64748B", linewidth=1)
    axes[1].axhline(
        -material_threshold,
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label="Material decline",
    )
    axes[1].set_ylabel("Macro-F1 change from clean")
    axes[1].set_title("Paired degradation")
    axes[1].legend()
    figure.suptitle(
        "Frozen primary-seed finalists under controlled image perturbations",
        fontsize=15,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    path = Path(output_path)
    atomic_write_bytes(path, buffer.getvalue())
    return path


def _plot_deployment_cost(cost: pd.DataFrame, output_path: str | Path) -> Path:
    available = cost.loc[cost["available"].astype(str).str.lower().eq("true")].copy()
    if available.empty:
        raise ValueError("deployment-cost figure has no available device measurements")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    candidates = ("C2", "I2")
    x = np.arange(2)
    width = 0.34
    cpu = available.loc[available["device"].eq("cpu")].set_index("candidate")
    cuda = available.loc[available["device"].eq("cuda")].set_index("candidate")
    axes[0].bar(
        x - width / 2,
        [float(cpu.loc[name, "model_only_median_ms"]) for name in candidates],
        width,
        color="#60A5FA",
        label="CPU model-only",
    )
    if set(cuda.index) == set(candidates):
        axes[0].bar(
            x + width / 2,
            [float(cuda.loc[name, "model_only_median_ms"]) for name in candidates],
            width,
            color="#34D399",
            label="CUDA model-only",
        )
    axes[0].set_xticks(x, candidates)
    axes[0].set_ylabel("Median single-image latency (ms)")
    axes[0].set_title("Model-only latency, batch size 1")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    tensor_mb = [
        float(cpu.loc[name, "parameter_and_buffer_bytes"]) / (1024**2) for name in candidates
    ]
    checkpoint_mb = [
        float(cpu.loc[name, "training_checkpoint_bytes"]) / (1024**2) for name in candidates
    ]
    axes[1].bar(
        x - width / 2,
        tensor_mb,
        width,
        color="#818CF8",
        label="Parameter + buffer bytes",
    )
    axes[1].bar(
        x + width / 2,
        checkpoint_mb,
        width,
        color="#F59E0B",
        label="Training checkpoint bytes",
    )
    axes[1].set_xticks(x, candidates)
    axes[1].set_ylabel("MiB")
    axes[1].set_title("Deployment state vs training artifact")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Machine-specific deployment cost",
        fontsize=15,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    path = Path(output_path)
    atomic_write_bytes(path, buffer.getvalue())
    return path


def build_robustness_cost_decision(
    tables: RobustnessTables,
    cost: pd.DataFrame,
    spec: RobustnessCostSpec,
) -> dict[str, Any]:
    """Summarise measured risk without reopening the frozen candidate choice."""
    perturbed = tables.pooled_metrics.loc[tables.pooled_metrics["condition"].ne("clean")]
    worst = {}
    material = {}
    for candidate in ("C2", "I2"):
        rows = perturbed.loc[perturbed["candidate"].eq(candidate)]
        worst_row = rows.sort_values("delta_macro_f1_vs_clean", kind="stable").iloc[0]
        worst[candidate] = {
            "condition": str(worst_row["condition"]),
            "macro_f1": float(worst_row["macro_f1"]),
            "delta_macro_f1_vs_clean": float(worst_row["delta_macro_f1_vs_clean"]),
            "prediction_agreement_with_clean": float(worst_row["prediction_agreement_with_clean"]),
        }
        material[candidate] = int(rows["material_macro_f1_degradation"].astype(bool).sum())
    comparison = tables.candidate_comparison.set_index("condition")
    condition_deltas = {
        condition: float(comparison.loc[condition, "i2_minus_c2_macro_f1"])
        for condition in comparison.index
    }
    available = cost.loc[cost["available"].astype(str).str.lower().eq("true")].copy()
    cost_summary: dict[str, Any] = {}
    for device in sorted(set(available["device"])):
        indexed = available.loc[available["device"].eq(device)].set_index("candidate")
        if set(indexed.index) != {"C2", "I2"}:
            continue
        cost_summary[device] = {
            "c2_model_only_median_ms": float(indexed.loc["C2", "model_only_median_ms"]),
            "i2_model_only_median_ms": float(indexed.loc["I2", "model_only_median_ms"]),
            "i2_to_c2_model_only_latency_ratio": float(
                indexed.loc["I2", "model_only_median_ms"]
                / indexed.loc["C2", "model_only_median_ms"]
            ),
            "c2_end_to_end_median_ms": float(indexed.loc["C2", "end_to_end_median_ms"]),
            "i2_end_to_end_median_ms": float(indexed.loc["I2", "end_to_end_median_ms"]),
        }
    cpu = available.loc[available["device"].eq("cpu")].set_index("candidate")
    size_summary = {
        "c2_parameter_and_buffer_bytes": int(cpu.loc["C2", "parameter_and_buffer_bytes"]),
        "i2_parameter_and_buffer_bytes": int(cpu.loc["I2", "parameter_and_buffer_bytes"]),
        "i2_to_c2_parameter_and_buffer_ratio": float(
            cpu.loc["I2", "parameter_and_buffer_bytes"]
            / cpu.loc["C2", "parameter_and_buffer_bytes"]
        ),
    }
    return {
        "schema_version": "1.0.0",
        "gate": "G6-ROBUSTNESS-COST",
        "decision_status": "closed",
        "analysis_role": "development_stress_and_machine_cost_diagnosis_only",
        "current_candidate": "I2",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "material_macro_f1_degradation_threshold": spec.material_macro_f1_degradation,
        "material_condition_count_by_candidate": material,
        "worst_condition_by_candidate": worst,
        "i2_minus_c2_macro_f1_by_condition": condition_deltas,
        "i2_above_c2_in_every_condition": all(value > 0 for value in condition_deltas.values()),
        "deployment_cost_by_device": cost_summary,
        "model_size": size_summary,
        "interpretation_rule": (
            "Controlled perturbations and machine cost can weaken a deployment claim, "
            "but this post-modelling analysis cannot reopen G5 model selection."
        ),
        "limitations": [
            "Perturbations are synthetic and do not cover every real acquisition shift.",
            "Latency and process memory are specific to the recorded machine and warm cache.",
            "Only the primary seed is probed because random-seed ordering was tested in G5.",
            "Softmax confidence remains uncalibrated until the calibration gate.",
        ],
        "next_question": "Run cross-fitted calibration and paired grouped bootstrap.",
    }


def build_robustness_cost_evidence(
    *,
    project_root: str | Path = ROOT,
    analysis_config_path: str | Path = ROBUSTNESS_CONFIG_PATH,
    slice_manifest_path: str | Path = DEFAULT_SLICE_MANIFEST,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = TASK2_FIGURE_DIR,
    mode: str = "run_or_load",
) -> dict[str, Any]:
    """Run/load the frozen primary-seed stress grid and write audited evidence."""
    if mode not in {"run", "load", "run_or_load"}:
        raise ValueError(f"unknown robustness evidence mode: {mode}")
    root = Path(project_root)
    config_path = _resolve_evidence_path(analysis_config_path, project_root=root)
    spec = load_robustness_cost_spec(config_path)
    _, resolved_slice_manifest, slice_sections = load_verified_slice_manifest(
        slice_manifest_path,
        project_root=root,
    )
    stability_manifest, _, stability_sections = load_verified_stability_manifest(
        slice_sections["stability_manifest"],
        project_root=root,
    )
    split_path = slice_sections["canonical_inputs"]["splits"]
    label_map_path = slice_sections["canonical_inputs"]["label_maps"]
    splits = load_splits(split_path)
    with label_map_path.open(encoding="utf-8") as handle:
        label_maps = json.load(handle)
    season_map = label_maps.get("season", {})
    if tuple(season_map.get("classes", ())) != tuple(SEASON_LABELS):
        raise ValueError("robustness changed the canonical Season labels")
    label_to_index = {
        str(label): int(index)
        for label, index in dict(season_map.get("label_to_index", {})).items()
    }
    if label_to_index != {label: index for index, label in enumerate(SEASON_LABELS)}:
        raise ValueError("robustness changed the canonical Season indices")
    article_type_classes = int(label_maps.get("articleType", {}).get("num_classes", -1))
    if article_type_classes != 124:
        raise ValueError("I2 robustness requires the canonical 124 ArticleType classes")

    development = get_samples(splits, partition="development", target="season").reset_index(
        drop=True
    )
    if len(development) != spec.expected_row_count:
        raise ValueError("robustness development Season row count changed")
    expected_ids = development["id"].astype(int).tolist()
    expected_targets = dict(
        zip(development["id"].astype(int), development["season"].astype(str), strict=True)
    )
    protected_ids = set(
        splits.loc[splits["partition"].isin(["holdout", "quarantine"]), "id"].astype(int)
    )
    registry = pd.read_csv(
        slice_sections["artifacts"]["registry_snapshot"],
        dtype=str,
        keep_default_na=False,
    )
    stability_summary = pd.read_csv(stability_sections["artifacts"]["seed_stability"])
    config_hashes = load_declared_config_hashes(stability_sections["input_configs"])
    slice_spec = load_slice_analysis_spec(slice_sections["analysis_config"])
    all_packs, _, _ = load_candidate_oof_packs(
        registry,
        slice_spec,
        project_root=root,
        expected_ids=expected_ids,
        expected_targets=expected_targets,
        protected_ids=protected_ids,
        split_sha256=compute_sha256(split_path),
        label_map_sha256=compute_sha256(label_map_path),
        config_sha256_by_experiment=config_hashes,
        stability_summary=stability_summary,
    )
    pack_by_identity = {(pack.candidate, pack.experiment_id, pack.seed): pack for pack in all_packs}
    selected_packs = []
    for candidate in spec.candidates:
        identity = (candidate.candidate, candidate.experiment_id, candidate.seed)
        if identity not in pack_by_identity:
            raise ValueError(f"robustness lacks frozen OOF pack: {identity}")
        selected_packs.append(pack_by_identity[identity])

    git_state = capture_git_state(root)
    if git_state.get("commit") is None:
        raise ValueError("robustness evidence requires a Git commit")
    if git_state.get("dirty") is not False:
        raise ValueError("robustness probes require a clean tracked Git worktree")
    runtime = capture_runtime()
    runtime_sha256 = canonical_sha256(runtime)
    config_sha256 = compute_sha256(config_path)
    split_sha256 = compute_sha256(split_path)
    label_map_sha256 = compute_sha256(label_map_path)
    implementation_hash = implementation_sha256(
        *ROBUSTNESS_IMPLEMENTATION_PATHS,
        root=root,
    )
    frames = canonical_validation_frames(splits)
    cache_root = _resolve_evidence_path(spec.robustness.cache_directory, project_root=root)
    clean_condition = spec.conditions[0]

    fold_stats: dict[tuple[str, int], Any] = {}
    checkpoint_audit_rows: list[dict[str, Any]] = []
    for pack in selected_packs:
        candidate_spec = RobustnessCandidate(pack.candidate, pack.experiment_id, pack.seed)
        for registry_row in pack.registry.sort_values("fold", kind="stable").to_dict(
            orient="records"
        ):
            fold = int(registry_row["fold"])
            training, validation = frames[fold]
            stats = fold_stats_from_history(
                registry_row,
                project_root=root,
                expected_training_ids=training["id"].astype(int).tolist(),
            )
            if stats.image_size != (80, 60) or stats.image_count != len(training):
                raise ValueError("robustness fold stats changed the frozen P0 geometry")
            if len(validation) != int(
                pack.oof.loc[pd.to_numeric(pack.oof["fold"]).astype(int).eq(fold)].shape[0]
            ):
                raise ValueError("robustness validation frame differs from frozen OOF fold")
            fold_stats[(pack.candidate, fold)] = stats
            checkpoint_path = _resolve_evidence_path(
                str(registry_row["checkpoint_path"]), project_root=root
            )
            history_path = _resolve_evidence_path(
                str(registry_row["history_path"]), project_root=root
            )
            verify_artifact(checkpoint_path, str(registry_row["checkpoint_sha256"]))
            verify_artifact(history_path, str(registry_row["history_sha256"]))
            checkpoint_audit_rows.append(
                {
                    "candidate": candidate_spec.candidate,
                    "experiment_id": candidate_spec.experiment_id,
                    "seed": candidate_spec.seed,
                    "fold": fold,
                    "run_id": str(registry_row["run_id"]),
                    "checkpoint_path": _portable_artifact_path(checkpoint_path, fallback_root=root),
                    "checkpoint_sha256": str(registry_row["checkpoint_sha256"]),
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "history_path": _portable_artifact_path(history_path, fallback_root=root),
                    "history_sha256": str(registry_row["history_sha256"]),
                    "fold_stats_sha256": canonical_sha256(stats.to_dict()),
                    "training_id_sha256": stats.training_id_sha256,
                    "validation_products": len(validation),
                }
            )
    for fold in range(5):
        c2_hash = canonical_sha256(fold_stats[("C2", fold)].to_dict())
        i2_hash = canonical_sha256(fold_stats[("I2", fold)].to_dict())
        if c2_hash != i2_hash:
            raise ValueError(f"C2 and I2 robustness stats differ for fold {fold}")

    cost_records: list[dict[str, Any]] = []
    for pack in selected_packs:
        candidate = RobustnessCandidate(pack.candidate, pack.experiment_id, pack.seed)
        row = (
            pack.registry.loc[
                pd.to_numeric(pack.registry["fold"], errors="raise")
                .astype(int)
                .eq(spec.cost.checkpoint_fold)
            ]
            .iloc[0]
            .to_dict()
        )
        stats = fold_stats[(candidate.candidate, spec.cost.checkpoint_fold)]
        validation = frames[spec.cost.checkpoint_fold][1]
        input_row = validation.sort_values("id", kind="stable").iloc[0]
        input_id = int(input_row["id"])
        image_path = root / str(input_row["path"])
        transform = PerturbedTensorTransform(stats=stats, condition=clean_condition)
        for requested_device in spec.cost.devices:
            gc.collect()
            rss_baseline = int(psutil.Process().memory_info().rss)
            model = build_robustness_model(
                candidate.candidate,
                article_type_classes=article_type_classes,
            )
            model, checkpoint_metadata, _ = load_robustness_checkpoint(
                model,
                row,
                project_root=root,
            )
            result = run_or_load_deployment_cost(
                model,
                candidate=candidate,
                checkpoint_run_id=str(row["run_id"]),
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                checkpoint_bytes=int(checkpoint_metadata["checkpoint_bytes"]),
                transform=transform,
                image_path=image_path,
                input_id=input_id,
                protocol=spec.cost,
                requested_device=requested_device,
                analysis_config_sha256=config_sha256,
                implementation_sha256_value=implementation_hash,
                stats_sha256=canonical_sha256(stats.to_dict()),
                runtime_sha256=runtime_sha256,
                input_image_sha256=str(input_row["sha256"]),
                cache_directory=cache_root,
                mode=mode,
                rss_baseline_before_model_load=rss_baseline,
            )
            result.pop("source", None)
            cost_records.append(result)
            del model
            gc.collect()
    deployment_cost = pd.DataFrame(cost_records).sort_values(["candidate", "device"], kind="stable")

    prediction_frames: dict[tuple[str, str], pd.DataFrame] = {}
    probe_records: list[dict[str, Any]] = []
    clean_reconciliation_rows: list[dict[str, Any]] = []
    for pack in selected_packs:
        candidate = RobustnessCandidate(pack.candidate, pack.experiment_id, pack.seed)
        condition_parts: dict[str, list[pd.DataFrame]] = {
            condition.condition: [] for condition in spec.conditions
        }
        for registry_row in pack.registry.sort_values("fold", kind="stable").to_dict(
            orient="records"
        ):
            fold = int(registry_row["fold"])
            model = build_robustness_model(
                candidate.candidate,
                article_type_classes=article_type_classes,
            )
            model, _, _ = load_robustness_checkpoint(
                model,
                registry_row,
                project_root=root,
            )
            for condition in spec.conditions:
                result = run_or_load_fold_probe(
                    model,
                    candidate=candidate,
                    condition=condition,
                    registry_row=registry_row,
                    validation_frame=frames[fold][1],
                    stats=fold_stats[(candidate.candidate, fold)],
                    label_to_index=label_to_index,
                    analysis_config_sha256=config_sha256,
                    split_sha256=split_sha256,
                    label_map_sha256=label_map_sha256,
                    implementation_sha256_value=implementation_hash,
                    cache_directory=cache_root,
                    mode=mode,
                    project_root=root,
                    batch_size=spec.robustness.batch_size,
                    num_workers=spec.robustness.num_workers,
                    pin_memory=spec.robustness.pin_memory,
                    device="auto",
                    use_amp=spec.robustness.amp_matches_training_evaluation,
                    git_commit=str(git_state["commit"]),
                )
                condition_parts[condition.condition].append(result.predictions)
                record = dict(result.record)
                record.pop("source", None)
                probe_records.append(record)
            del model
            gc.collect()
        for condition in spec.conditions:
            combined = pd.concat(condition_parts[condition.condition], ignore_index=True)
            validate_oof(
                combined,
                expected_ids=expected_ids,
                expected_targets=expected_targets,
                protected_ids=protected_ids,
                labels=SEASON_LABELS,
            )
            prediction_frames[(candidate.candidate, condition.condition)] = combined
            if condition.condition == "clean":
                clean_reconciliation_rows.append(
                    reconcile_clean_probe(
                        combined,
                        pack.oof,
                        candidate=candidate,
                        protocol=spec.robustness,
                    )
                )

    tables = build_robustness_tables(prediction_frames, spec)
    decision = build_robustness_cost_decision(tables, deployment_cost, spec)
    checkpoint_audit = pd.DataFrame(checkpoint_audit_rows).sort_values(
        ["candidate", "fold"], kind="stable"
    )
    probe_registry = pd.json_normalize(probe_records, sep="_").sort_values(
        ["candidate", "condition", "fold"], kind="stable"
    )
    clean_reconciliation = pd.DataFrame(clean_reconciliation_rows).sort_values(
        "candidate", kind="stable"
    )

    evidence_root = _resolve_evidence_path(evidence_directory, project_root=root)
    figure_root = _resolve_evidence_path(figure_directory, project_root=root)
    paths = {
        "pooled_metrics": evidence_root / "pooled_metrics.csv",
        "fold_metrics": evidence_root / "fold_metrics.csv",
        "candidate_comparison": evidence_root / "candidate_comparison.csv",
        "probe_registry": evidence_root / "probe_registry.csv",
        "checkpoint_audit": evidence_root / "checkpoint_audit.csv",
        "deployment_cost": evidence_root / "deployment_cost.csv",
        "clean_reconciliation": evidence_root / "clean_reconciliation.csv",
        "runtime": evidence_root / "runtime.json",
        "decision": evidence_root / "decision.json",
        "robustness_figure": figure_root / "robustness_comparison.png",
        "deployment_cost_figure": figure_root / "deployment_cost.png",
    }
    atomic_write_csv(paths["pooled_metrics"], tables.pooled_metrics)
    atomic_write_csv(paths["fold_metrics"], tables.fold_metrics)
    atomic_write_csv(paths["candidate_comparison"], tables.candidate_comparison)
    atomic_write_csv(paths["probe_registry"], probe_registry)
    atomic_write_csv(paths["checkpoint_audit"], checkpoint_audit)
    atomic_write_csv(paths["deployment_cost"], deployment_cost)
    atomic_write_csv(paths["clean_reconciliation"], clean_reconciliation)
    atomic_write_json(paths["runtime"], runtime)
    atomic_write_json(paths["decision"], decision)
    _plot_robustness(
        tables,
        paths["robustness_figure"],
        material_threshold=spec.material_macro_f1_degradation,
    )
    _plot_deployment_cost(deployment_cost, paths["deployment_cost_figure"])

    artifact_manifest = {
        name: {
            "path": _portable_artifact_path(path, fallback_root=root),
            "sha256": compute_sha256(path),
        }
        for name, path in paths.items()
    }
    probe_inputs = {
        str(row["probe_id"]): {
            "path": _portable_artifact_path(
                _resolve_evidence_path(str(row["prediction_path"]), project_root=root),
                fallback_root=root,
            ),
            "sha256": str(row["prediction_sha256"]),
        }
        for row in probe_records
    }
    cost_inputs = {
        f"{row['candidate']}-{row['device']}": {
            "path": _portable_artifact_path(
                _resolve_evidence_path(str(row["result_path"]), project_root=root),
                fallback_root=root,
            ),
            "sha256": str(row["result_sha256"]),
        }
        for row in cost_records
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-ROBUSTNESS-COST",
        "decision_status": "closed",
        "analysis_role": "development_stress_and_machine_cost_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "git_commit": str(git_state["commit"]),
        "git_dirty": False,
        "analysis_config": {
            "path": _portable_artifact_path(config_path, fallback_root=root),
            "sha256": config_sha256,
        },
        "slice_manifest": {
            "path": _portable_artifact_path(resolved_slice_manifest, fallback_root=root),
            "sha256": compute_sha256(resolved_slice_manifest),
        },
        "stability_coverage_sha256": stability_manifest["coverage_sha256"],
        "canonical_inputs": {
            "splits": {
                "path": _portable_artifact_path(split_path, fallback_root=root),
                "sha256": split_sha256,
            },
            "label_maps": {
                "path": _portable_artifact_path(label_map_path, fallback_root=root),
                "sha256": label_map_sha256,
            },
        },
        "implementation_sha256": implementation_hash,
        "runtime_sha256": runtime_sha256,
        "input_checkpoints": {
            str(row["run_id"]): {
                "path": str(row["checkpoint_path"]),
                "sha256": str(row["checkpoint_sha256"]),
            }
            for row in checkpoint_audit_rows
        },
        "input_probe_predictions": probe_inputs,
        "input_cost_results": cost_inputs,
        "artifacts": artifact_manifest,
    }
    manifest_path = evidence_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = compute_sha256(manifest_path)
    return manifest


__all__ = [
    "build_robustness_cost_decision",
    "build_robustness_cost_evidence",
    "load_verified_slice_manifest",
]
