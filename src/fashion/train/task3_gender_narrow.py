"""G-N64: one scratch width screen with a frozen three-point validation-loss budget."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import (
    NARROW_GEM3_CHANNELS,
    NARROW_GEM3_FAMILY,
    Task3BaselineConfig,
    baseline_parameter_count,
    narrow_gem3_parameter_count,
)
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import check, decision, oof_metrics, paired_family_bootstrap
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_ieee import POLICY, evaluate_gender_ieee
from fashion.train.task3_gender_precision import probability_difference
from fashion.train.task3_gender_repair import load_sources
from fashion.train.task3_gender_weight_decay import _pool, _robustness, _verify_baseline_controls

NAME = "gender_narrow64"
FOLDS = (0, 4)
RULE_VERSION = "gn64_loss003_gap005_v1"
MAX_VALIDATION_LOSS = 0.03
MIN_GAP_REDUCTION = 0.05
MEMORY_LIMIT = 3_000_000_000
TRAINING_PRECISION = {
    "backends": "none",
    "backends.cuda.matmul": "none",
    "backends.cudnn": "none",
    "backends.cudnn.conv": "tf32",
    "backends.cudnn.rnn": "tf32",
}
RUNTIME = {"torch": "2.11.0+cu128", "cuda": "12.8", "gpu": "NVIDIA L4"}


def _at_least(value, bound):
    """Allow decimal-boundary roundoff only, never an extra model-score budget."""
    return np.isfinite(value) and (value >= bound or np.isclose(value, bound, rtol=0, atol=1e-12))


def narrow_config(spec, *, fold, device_name):
    """Change only the last convolution's width and its architecture identifier."""
    if spec.to_dict() != dataset_v2_spec(NAME, spec.parent_run_ids).to_dict():
        raise ValueError("G-N64 requires the frozen narrow-width recipe")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("G-N64 requires CUDA and only folds 0 and 4")
    return replace(
        Task3BaselineConfig(target="gender"),
        model_family=NARROW_GEM3_FAMILY,
        channels=NARROW_GEM3_CHANNELS,
    )


def read_precision_prerequisites(path, *, root=ROOT):
    """Verify the completed 04w evidence from its saved probability arrays, without fitting."""
    if path is None:
        raise ValueError("The completed 04w precision diagnostic is required")
    directory = Path(path)
    status_path = directory / "precision_status.json"
    status = json.loads(status_path.read_text())
    if (
        status["status"] != "complete_for_review"
        or status["training_performed"]
        or not status["ieee_all_comparisons_pass"]
        or not status["earlier_summaries_reproduced"]
        or not status["settings_restored"]
    ):
        raise ValueError("The completed precision check has not passed its review prerequisites")
    if status["runtime"] != RUNTIME or status["runtime_default_settings"] != TRAINING_PRECISION:
        raise ValueError("The precision source does not match the frozen G2 runtime controls")
    if status["ieee_settings"] != {key: "ieee" for key in TRAINING_PRECISION}:
        raise ValueError("The comparison precision was not full IEEE FP32")
    if status["split_sha256"] != compute_sha256(Path(root) / "data/processed/splits.csv"):
        raise ValueError("The precision source uses a different canonical split")
    pairs = {(r["model"], r["fold"]) for r in status["runs"]}
    if len(status["runs"]) != 4 or pairs != {
        (m, f) for m in ("G2", "CompactBlurCNN") for f in FOLDS
    }:
        raise ValueError("The precision source needs all four completed models")
    files = [status_path, directory / "precision_comparisons.csv"]
    comparisons = pd.read_csv(files[1])
    key_columns = ["run_id", "partition", "mode", "batch_size", "order"]
    if len(comparisons) != 144 or comparisons.duplicated(key_columns).any():
        raise ValueError("Precision evidence needs 144 unique comparisons")
    for run in status["runs"]:
        if not run["weights_and_buffers_unchanged"]:
            raise ValueError("Precision diagnostic changed model state")
        for partition in ("train", "validation"):
            file = directory / run["run_id"] / f"{partition}_probabilities.npz"
            files.append(file)
            with np.load(file, allow_pickle=False) as arrays:
                if len(arrays["ids"]) != 160 or len(np.unique(arrays["ids"])) != 160:
                    raise ValueError("Invalid precision sample IDs")
                for mode in ("runtime_default", "ieee"):
                    reference = arrays[f"{mode}_reference"]
                    for batch in (1, 32, 128):
                        for order in ("forward", "reverse", "shuffle"):
                            actual = arrays[f"{mode}_b{batch}_{order}"]
                            for probabilities in (reference, actual):
                                if (
                                    probabilities.shape != (160, 5)
                                    or not np.isfinite(probabilities).all()
                                    or (probabilities < 0).any()
                                    or not np.allclose(probabilities.sum(1), 1, atol=1e-6)
                                ):
                                    raise ValueError("Invalid precision probabilities")
                            difference = probability_difference(reference, actual)
                            if mode == "ieee" and not difference["pass"]:
                                raise ValueError("A saved IEEE probability comparison fails")
                            saved = comparisons[
                                (comparisons.run_id == run["run_id"])
                                & (comparisons.partition == partition)
                                & (comparisons["mode"] == mode)
                                & (comparisons.batch_size == batch)
                                & (comparisons.order == order)
                            ]
                            if len(saved) != 1:
                                raise ValueError("Missing precision comparison")
                            saved = saved.iloc[0]
                            if (
                                abs(saved.max_abs_difference - difference["max_abs_difference"])
                                > 1e-12
                                or saved.prediction_flips != difference["prediction_flips"]
                                or saved["pass"] != difference["pass"]
                            ):
                                raise ValueError(
                                    "Precision summary differs from saved probabilities"
                                )
                            if mode == "runtime_default" and (
                                not saved.earlier_summary_reproduced
                                or abs(
                                    saved.earlier_max_abs_difference
                                    - difference["max_abs_difference"]
                                )
                                > 1e-7
                            ):
                                raise ValueError("The earlier precision result did not reproduce")
    return {
        "status": status,
        "artifact_sha256": {str(p.relative_to(directory)): compute_sha256(p) for p in files},
    }


def require_narrow_prerequisites(path, *, root=ROOT):
    """Prevent direct trainer entry from bypassing precision/data checks."""
    evidence = read_precision_prerequisites(path, root=root)
    import torch

    from fashion.train.task3_gender_precision import precision_settings

    if not torch.cuda.is_available():
        raise RuntimeError("Use a fresh Colab L4 GPU runtime")
    actual = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    if actual != RUNTIME or precision_settings(torch) != TRAINING_PRECISION:
        raise RuntimeError("Use the same fresh L4 runtime and training precision as 04w")
    return evidence


def _screen_contract(name):
    if name == NAME:
        return RULE_VERSION, narrow_gem3_parameter_count()
    if name == "gender_dropout_030":
        return "gdrop030_loss003_gap005_v1", baseline_parameter_count("gender")
    raise ValueError("Unknown frozen gender screen")


def _screen_config(spec, *, fold, device_name):
    _screen_contract(spec.name)
    if spec.name == "gender_dropout_030":
        from fashion.train.task3_gender_dropout import dropout_config

        return dropout_config(spec, fold=fold, device_name=device_name)
    return narrow_config(spec, fold=fold, device_name=device_name)


def evaluate_gender_narrow_screen(
    child, sources, classes, *, repetitions=10_000, experiment_name=NAME
):
    """All comparisons use matched IEEE snapshots, with unchanged class/corruption guards."""
    rule, parameters = _screen_contract(experiment_name)
    if set(child) != set(FOLDS) or any(set(sources[n]) != set(FOLDS) for n in ("G2", "E6")):
        raise ValueError("G-N64 screen requires exactly folds 0 and 4")
    for group in (child, sources["G2"], sources["E6"]):
        if any(run["metrics"].get("comparison_precision") != POLICY for run in group.values()):
            raise ValueError("Every model must use matched IEEE evaluation")
    candidate = oof_metrics(_pool(child), classes)
    parent = oof_metrics(_pool(sources["G2"]), classes)
    interval = paired_family_bootstrap(
        _pool(child), _pool(sources["G2"]), classes=classes, repetitions=repetitions, seed=2753
    )
    checks, folds = [], []

    def add(name, value, rule, passed):
        checks.append(check(name, value, rule, bool(passed)))

    delta = candidate["macro_f1"] - parent["macro_f1"]
    add(
        "validation_delta",
        delta,
        ">= -0.030 versus matched IEEE G2",
        _at_least(delta, -MAX_VALIDATION_LOSS),
    )
    add(
        "validation_ci_lower",
        interval["lower_95"],
        ">= -0.030",
        _at_least(interval["lower_95"], -MAX_VALIDATION_LOSS),
    )
    for fold in FOLDS:
        c, p = child[fold]["metrics"], sources["G2"][fold]["metrics"]
        for metrics in (c, p):
            expected = metrics["final_train_eval_macro_f1"] - metrics["macro_f1"]
            if not np.isclose(
                expected, metrics["final_train_validation_macro_f1_gap"], rtol=0, atol=1e-8
            ):
                raise ValueError(
                    "Clean train/validation gap arithmetic differs from the checkpoint scores"
                )
        gap_reduction = (
            p["final_train_validation_macro_f1_gap"] - c["final_train_validation_macro_f1_gap"]
        )
        fold_delta = c["macro_f1"] - p["macro_f1"]
        add(
            f"fold_{fold}.validation_delta",
            fold_delta,
            ">= -0.030",
            _at_least(fold_delta, -MAX_VALIDATION_LOSS),
        )
        add(f"fold_{fold}.gap_reduction", gap_reduction, "> 0", gap_reduction > 0)
        memory = c["peak_memory_bytes"]
        add(
            f"fold_{fold}.gpu_memory_bytes",
            memory,
            "0 < value < 3000000000",
            np.isfinite(memory) and 0 < memory < MEMORY_LIMIT,
        )
        add(
            f"fold_{fold}.parameters",
            c["parameter_count"],
            f"== {parameters}",
            c["parameter_count"] == parameters,
        )
        folds.append(
            {
                "fold": fold,
                "candidate_train_f1": c["final_train_eval_macro_f1"],
                "candidate_validation_f1": c["macro_f1"],
                "candidate_gap": c["final_train_validation_macro_f1_gap"],
                "parent_train_f1": p["final_train_eval_macro_f1"],
                "parent_validation_f1": p["macro_f1"],
                "parent_gap": p["final_train_validation_macro_f1_gap"],
                "gap_reduction": gap_reduction,
                "train_seconds": c["train_seconds"],
                "latency_ms_batch_1": c["latency_ms_batch_1"],
                "peak_memory_bytes": memory,
            }
        )
    mean_reduction = float(np.mean([f["gap_reduction"] for f in folds]))
    add(
        "mean_gap_reduction",
        mean_reduction,
        ">= 0.050",
        _at_least(mean_reduction, MIN_GAP_REDUCTION),
    )
    current = {r["class_name"]: r["f1"] for r in candidate["per_class"]}
    changes = {r["class_name"]: current[r["class_name"]] - r["f1"] for r in parent["per_class"]}
    add("class_f1", changes, "every class >= -0.020", _at_least(min(changes.values()), -0.020))
    for name, limit in (("nll", 0.020), ("ece_15", 0.010)):
        value = candidate[name] - parent[name]
        add(name, value, f"<= {limit} versus matched IEEE G2", value <= limit)
    robust = _robustness(child) - _robustness(sources["E6"])
    for name, value in robust.items():
        bound = 0.030 if name == "translation_003" else -0.020
        add(f"robustness_vs_e6.{name}", float(value), f">= {bound}", _at_least(value, bound))
    return {
        "status": decision(checks),
        "phase": "screen",
        "rule_version": rule,
        "checks": checks,
        "candidate": candidate,
        "comparison": parent,
        "bootstrap": interval,
        "folds": folds,
        "mean_gap_reduction": mean_reduction,
        "required_validation_f1": parent["macro_f1"] - MAX_VALIDATION_LOSS,
        "maximum_mean_clean_gap": float(np.mean([f["parent_gap"] for f in folds]))
        - MIN_GAP_REDUCTION,
        "robustness_change_vs_e6": robust.to_dict(),
        "comparison_precision": POLICY,
        "training_precision": TRAINING_PRECISION,
        "speed_cap": None,
        "independent_test_evidence": False,
        "next_step": (
            "Stop. A pass permits planning confirmation separately; it is not model acceptance."
        ),
    }


def check_gender_narrow_sources(
    *,
    g2_directory,
    e6_directory,
    source_registry_path,
    precision_directory,
    root=ROOT,
    experiment_name=NAME,
):
    _screen_contract(experiment_name)
    evidence = read_precision_prerequisites(precision_directory, root=root)
    sources, classes = load_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        registry_path=source_registry_path,
        root=root,
    )
    for group in sources.values():
        for run in group.values():
            _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
    for run in evidence["status"]["runs"]:
        if run["model"] == "G2" and run["source_sha256"] != sources["G2"][run["fold"]]["sha256"]:
            raise ValueError("G2 source differs from the completed precision diagnostic")
    spec = dataset_v2_spec(experiment_name, [sources["G2"][f]["run_id"] for f in range(5)])
    return sources, classes, spec, evidence


def run_gender_narrow_screen(
    *,
    g2_directory,
    e6_directory,
    source_registry_path,
    precision_directory,
    output_root,
    registry_path,
    root=ROOT,
    registry_mirrors=(),
    device_name="cuda",
    experiment_name=NAME,
):
    """Evaluate references, fit only folds 0/4, then score the same-precision snapshots."""
    from fashion.train.task3_baseline import _json_dump, run_task3_baseline_fold
    from fashion.train.task3_dataset_v2 import _reusable_fold

    rule, _ = _screen_contract(experiment_name)
    root, output_root = Path(root), Path(output_root)
    sources, classes, spec, evidence = check_gender_narrow_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        source_registry_path=source_registry_path,
        precision_directory=precision_directory,
        root=root,
        experiment_name=experiment_name,
    )
    expected = _screen_config(spec, fold=0, device_name=device_name)
    require_narrow_prerequisites(precision_directory, root=root)
    splits = load_splits(root / "data/processed/splits.csv")
    destination = output_root / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(
        {
            "spec": spec.to_dict(),
            "baseline_controls": expected.to_dict(),
            "source_registry_sha256": compute_sha256(source_registry_path),
            "source_sha256": {
                r["run_id"]: r["sha256"] for group in sources.values() for r in group.values()
            },
            "precision_evidence_sha256": evidence["artifact_sha256"],
            "folds": list(FOLDS),
            "optimizer_steps": 0,
            "rule_version": rule,
            "training_precision": TRAINING_PRECISION,
            "comparison_precision": POLICY,
        },
        destination / "source_audit.json",
    )
    matched = {"G2": {}, "E6": {}}
    for name in matched:
        for fold in FOLDS:
            run = sources[name][fold]
            matched[name][fold] = evaluate_gender_ieee(
                run,
                splits=splits,
                classes=classes,
                root=root,
                output=destination / "comparison_ieee_v2" / run["run_id"],
            )
    child = {}
    for fold in FOLDS:
        result = _reusable_fold(spec, fold, output_root=output_root)
        if result is None:
            result = run_task3_baseline_fold(
                "gender",
                fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,
                parent_run_directory=sources["G2"][fold]["directory"],
                prerequisite_path=precision_directory,
            )
        run = inspect_gender_run(
            Path(result["run_dir"]),
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
        )
        _verify_baseline_controls(run["config"], expected)
        if (
            run["config"].get("training_precision_settings") != TRAINING_PRECISION
            or run["config"].get("precision_evidence_sha256") != evidence["artifact_sha256"]
        ):
            raise ValueError("G-N64 training precision evidence changed")
        if (
            run["config"]["child_experiment"] != spec.to_dict()
            or run["config"]["parent_run_id"] != sources["G2"][fold]["run_id"]
        ):
            raise ValueError("G-N64 candidate configuration or matched parent disagrees")
        memory = run["metrics"]["peak_memory_bytes"]
        if not np.isfinite(memory) or not 0 < memory < MEMORY_LIMIT:
            stopped = {
                "status": "fail",
                "phase": "screen",
                "fold": fold,
                "reason": "GPU memory must stay below 3 GB; later folds were not started.",
            }
            _json_dump(stopped, destination / "screen_decision.json")
            return stopped
        run["directory"] = result["run_dir"]
        child[fold] = evaluate_gender_ieee(
            run,
            splits=splits,
            classes=classes,
            root=root,
            output=destination / "comparison_ieee_v2" / run["run_id"],
        )
    report = evaluate_gender_narrow_screen(child, matched, classes, experiment_name=experiment_name)
    report["registry_and_artifact_integrity"] = True
    report["run_ids"] = {str(f): child[f]["run_id"] for f in FOLDS}
    _json_dump(report, destination / "screen_decision.json")
    pd.DataFrame(report["folds"]).to_csv(destination / "clean_gap_comparison.csv", index=False)
    _pool(child).sort_values("id").to_csv(destination / "ieee_oof_predictions.csv", index=False)
    return report
