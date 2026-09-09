"""G-WD1: screen stronger AdamW decay against matched G2, with clean gap checks."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import (
    check,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    robustness_changes,
)
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_repair import load_sources

NAME = "gender_weight_decay_001"
FOLDS = (0, 4)
MEMORY_LIMIT = 3_000_000_000
RULE_VERSION = "gwd1_gap_and_validation_v1"


def weight_decay_config(spec, *, fold, device_name):
    """Keep every baseline control except the single frozen decay change."""
    expected = dataset_v2_spec(NAME, spec.parent_run_ids)
    if spec.to_dict() != expected.to_dict():
        raise ValueError("G-WD1 requires the frozen weight-decay specification")
    if fold not in FOLDS:
        raise ValueError("G-WD1 screen permits only folds 0 and 4")
    if device_name != "cuda":
        raise ValueError("G-WD1 uses normal CUDA execution")
    return replace(Task3BaselineConfig(target="gender"), weight_decay=spec.weight_decay)


def _pool(runs):
    return pd.concat([runs[f]["predictions"] for f in FOLDS], ignore_index=True)


def _robustness(runs):
    return robustness_changes(
        pd.concat([runs[f]["robustness"] for f in FOLDS], ignore_index=True),
        clean_by_fold={f: runs[f]["metrics"]["macro_f1"] for f in FOLDS},
        run_ids_by_fold={f: runs[f]["run_id"] for f in FOLDS},
    )


def evaluate_weight_decay_screen(child, sources, classes, *, repetitions=10_000):
    """A smaller gap alone cannot pass: validation must improve as well."""
    if set(child) != set(FOLDS):
        raise ValueError("G-WD1 screen needs exactly folds 0 and 4")
    g2, e6 = sources["G2"], sources["E6"]
    candidate = oof_metrics(_pool(child), classes)
    parent = oof_metrics(_pool(g2), classes)
    interval = paired_family_bootstrap(
        _pool(child), _pool(g2), classes=classes, repetitions=repetitions, seed=2753
    )
    checks = []

    def add(name, value, rule, passed):
        checks.append(check(name, value, rule, passed))

    gain = candidate["macro_f1"] - parent["macro_f1"]
    add("validation_gain", gain, ">= 0.010 versus matched G2", gain >= 0.010)
    add("validation_ci_lower", interval["lower_95"], "> 0", interval["lower_95"] > 0)
    folds = []
    for f in FOLDS:
        c, p = child[f]["metrics"], g2[f]["metrics"]
        # These are evaluation-mode clean-training scores from the final model,
        # never the online augmented-training epoch scores.
        for name, metrics in (("candidate", c), ("G2", p)):
            expected_gap = metrics["final_train_eval_macro_f1"] - metrics["macro_f1"]
            if not np.isclose(
                expected_gap, metrics["final_train_validation_macro_f1_gap"], atol=1e-8, rtol=0
            ):
                raise ValueError(f"{name} clean gap differs from its final checkpoint scores")
        delta = c["macro_f1"] - p["macro_f1"]
        gap_delta = (
            c["final_train_validation_macro_f1_gap"] - p["final_train_validation_macro_f1_gap"]
        )
        add(f"fold_{f}.validation", delta, ">= -0.005", delta >= -0.005)
        add(f"fold_{f}.gap_change", gap_delta, "<= 0", gap_delta <= 0)
        memory = c["peak_memory_bytes"]
        add(
            f"fold_{f}.gpu_memory_bytes",
            memory,
            "0 < value < 3000000000",
            np.isfinite(memory) and 0 < memory < MEMORY_LIMIT,
        )
        add(
            f"fold_{f}.parameters",
            c["parameter_count"],
            "== 390181",
            c["parameter_count"] == 390181,
        )
        folds.append(
            {
                "fold": f,
                "candidate_train_f1": c["final_train_eval_macro_f1"],
                "candidate_validation_f1": c["macro_f1"],
                "candidate_gap": c["final_train_validation_macro_f1_gap"],
                "parent_train_f1": p["final_train_eval_macro_f1"],
                "parent_validation_f1": p["macro_f1"],
                "parent_gap": p["final_train_validation_macro_f1_gap"],
                "train_seconds": c["train_seconds"],
                "latency_ms_batch_1": c["latency_ms_batch_1"],
                "peak_memory_bytes": memory,
            }
        )
    mean_gap = float(np.mean([r["candidate_gap"] for r in folds]))
    parent_gap = float(np.mean([r["parent_gap"] for r in folds]))
    reduction = parent_gap - mean_gap
    add("mean_gap_reduction", reduction, ">= 0.030 versus matched G2", reduction >= 0.030)
    by_class = {r["class_name"]: r["f1"] for r in candidate["per_class"]}
    differences = {
        r["class_name"]: by_class[r["class_name"]] - r["f1"] for r in parent["per_class"]
    }
    add("class_f1", differences, "every class >= -0.020", min(differences.values()) >= -0.020)
    for name, limit in (("nll", 0.020), ("ece_15", 0.010)):
        value = candidate[name] - parent[name]
        add(name, value, f"<= {limit} versus matched G2", value <= limit)
    robust = _robustness(child)
    against_e6 = robust - _robustness(e6)
    for name, value in against_e6.items():
        bound = 0.030 if name == "translation_003" else -0.020
        add(f"robustness_vs_e6.{name}", float(value), f">= {bound}", value >= bound)
    return {
        "status": decision(checks),
        "phase": "screen",
        "rule_version": RULE_VERSION,
        "checks": checks,
        "candidate": candidate,
        "comparison": parent,
        "bootstrap": interval,
        "folds": folds,
        "mean_clean_gap": mean_gap,
        "parent_mean_clean_gap": parent_gap,
        "required_validation_f1": parent["macro_f1"] + 0.010,
        "maximum_mean_clean_gap": parent_gap - 0.030,
        "robustness_change_vs_e6": against_e6.to_dict(),
        "speed_cap": None,
        "independent_test_evidence": False,
        "next_step": "Only a pass permits a separately planned confirmation; never auto-run it.",
    }


def _verify_baseline_controls(config, expected):
    for field in fields(Task3BaselineConfig):
        actual = config[field.name]
        wanted = expected.to_dict()[field.name]
        if actual != wanted:
            raise ValueError(f"G-WD1 baseline control changed: {field.name}")


def check_weight_decay_sources(*, g2_directory, e6_directory, source_registry_path, root=ROOT):
    """Zero-training source audit; reject missing or changed reference recipes."""
    sources, classes = load_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        registry_path=source_registry_path,
        root=root,
    )
    for run in sources["G2"].values():
        _verify_baseline_controls(run["config"], Task3BaselineConfig(target="gender"))
    spec = dataset_v2_spec(NAME, [sources["G2"][f]["run_id"] for f in range(5)])
    return sources, classes, spec


def run_gender_weight_decay_screen(
    *,
    g2_directory,
    e6_directory,
    source_registry_path,
    output_root,
    registry_path,
    root=ROOT,
    registry_mirrors=(),
    device_name="cuda",
):
    """Train only folds 0/4 from scratch, verify artifacts, then write the decision."""
    from fashion.train.task3_baseline import _json_dump, run_task3_baseline_fold
    from fashion.train.task3_dataset_v2 import _reusable_fold

    root, output_root = Path(root), Path(output_root)
    sources, classes, spec = check_weight_decay_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        source_registry_path=source_registry_path,
        root=root,
    )
    expected_config = weight_decay_config(spec, fold=0, device_name=device_name)
    splits = load_splits(root / "data/processed/splits.csv")
    destination = output_root / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(
        {
            "spec": spec.to_dict(),
            "baseline_controls": expected_config.to_dict(),
            "source_registry_sha256": compute_sha256(source_registry_path),
            "source_sha256": {
                run["run_id"]: run["sha256"] for group in sources.values() for run in group.values()
            },
            "folds": list(FOLDS),
            "optimizer_steps": 0,
            "rule_version": RULE_VERSION,
        },
        destination / "source_audit.json",
    )
    child = {}
    for f in FOLDS:
        result = _reusable_fold(spec, f, output_root=output_root)
        if result is None:
            result = run_task3_baseline_fold(
                "gender",
                f,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,
                parent_run_directory=sources["G2"][f]["directory"],
            )
        run = inspect_gender_run(
            Path(result["run_dir"]),
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
        )
        if run["config"]["child_experiment"] != spec.to_dict():
            raise ValueError("G-WD1 reused run has the wrong candidate recipe")
        _verify_baseline_controls(run["config"], expected_config)
        if run["config"]["parent_run_id"] != sources["G2"][f]["run_id"]:
            raise ValueError("G-WD1 candidate has the wrong matched parent")
        child[f] = run
        memory = run["metrics"]["peak_memory_bytes"]
        if not np.isfinite(memory) or not 0 < memory < MEMORY_LIMIT:
            stopped = {
                "status": "fail",
                "phase": "screen",
                "reason": "GPU memory must be strictly below 3 GB; later folds were not started.",
                "fold": f,
                "peak_memory_bytes": memory,
            }
            _json_dump(stopped, destination / "screen_decision.json")
            return stopped
    report = evaluate_weight_decay_screen(child, sources, classes)
    report["registry_and_artifact_integrity"] = True
    report["run_ids"] = {str(f): child[f]["run_id"] for f in FOLDS}
    _json_dump(report, destination / "screen_decision.json")
    pd.DataFrame(report["folds"]).to_csv(destination / "clean_gap_comparison.csv", index=False)
    _pool(child).sort_values("id").to_csv(destination / "oof_predictions.csv", index=False)
    return report
