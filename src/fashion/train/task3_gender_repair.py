"""G-D1: a gated, scratch Gender darkening screen and separate confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import load_label_maps, load_splits
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_decisions import (
    check,
    decision,
    oof_metrics,
    paired_family_bootstrap,
    robustness_changes,
)
from fashion.train.task3_g2_audit import inspect_gender_run

NAME = "gender_translation_mild_darkening"
MEMORY_LIMIT = 3_000_000_000
PARAMETERS = 390_181


def load_sources(*, g2_directory, e6_directory, registry_path, root=ROOT):
    """Read exactly the five saved G2 and E6 runs and verify their lineage."""
    root = Path(root)
    registry = pd.read_csv(registry_path, keep_default_na=False)
    splits = load_splits(root / "data/processed/splits.csv")
    classes = list(load_label_maps(root / "data/processed/label_maps.json")["gender"]["classes"])
    sources = {}
    for name, directory, prefix in (
        ("G2", g2_directory, "t3_gender_v2_g2_translation_gender_"),
        ("E6", e6_directory, "t3_gender_e6_gem_p3_gender_"),
    ):
        runs = {}
        for path in sorted(Path(directory).glob(prefix + "*")):
            if not path.is_dir():
                continue
            run = inspect_gender_run(
                path, registry=registry, splits=splits, classes=classes, root=root
            )
            if run["fold"] in runs:
                raise ValueError(f"duplicate {name} fold")
            run["directory"] = str(path.resolve())
            run["environment"] = json.loads(
                registry.loc[registry.run_id.eq(run["run_id"]), "environment_json"].iloc[0]
            )
            runs[run["fold"]] = run
        if set(runs) != set(range(5)):
            raise ValueError(f"{name} requires five verified source runs")
        sources[name] = runs
    e6_ids = [sources["E6"][f]["run_id"] for f in range(5)]
    expected = dataset_v2_spec("gender_v2_translation", e6_ids).to_dict()
    for fold, run in sources["G2"].items():
        if run["config"]["child_experiment"] != expected:
            raise ValueError("G2 is not the original frozen translation candidate")
        if run["config"]["parent_run_id"] != e6_ids[fold]:
            raise ValueError("G2/E6 fold lineage disagrees")
    return sources, classes


def _pool(runs, folds):
    return pd.concat([runs[f]["predictions"] for f in folds], ignore_index=True)


def _changes(runs, folds):
    return robustness_changes(
        pd.concat([runs[f]["robustness"] for f in folds], ignore_index=True),
        clean_by_fold={f: runs[f]["metrics"]["macro_f1"] for f in folds},
        run_ids_by_fold={f: runs[f]["run_id"] for f in folds},
    )


def _paired(child, parent, folds, classes, repetitions):
    return paired_family_bootstrap(
        _pool(child, folds), _pool(parent, folds), classes=classes, repetitions=repetitions
    )


def evaluate_gender_repair(
    child: Mapping[int, Any],
    sources: Mapping[str, Any],
    classes: Sequence[str],
    *,
    phase: str,
    repetitions: int = 10_000,
) -> dict[str, Any]:
    """Apply frozen rules. G2 is a comparison model, never an accepted parent."""
    folds = (0, 4) if phase == "screen" else tuple(range(5))
    if phase not in {"screen", "confirmation"} or set(child) != set(folds):
        raise ValueError("screen needs folds 0/4; confirmation needs all five folds")
    g2, e6 = sources["G2"], sources["E6"]
    checks, evidence = [], {}

    def add(name, value, rule, passed):
        checks.append(check(name, value, rule, passed))

    # Retain the screen rules on the selected folds, plus its pooled margins on
    # all five folds after confirmation. Fresh-fold E6 evidence stays separate.
    for scope, selected in [("screen", (0, 4))] + (
        [("pooled_five_vs_g2", folds)] if phase == "confirmation" else []
    ):
        c = oof_metrics(_pool(child, selected), classes)
        p = oof_metrics(_pool(g2, selected), classes)
        interval = _paired(child, g2, selected, classes, repetitions)
        delta = c["macro_f1"] - p["macro_f1"]
        add(scope + ".clean_gain", delta, ">= -0.005", delta >= -0.005)
        add(
            scope + ".clean_ci_lower",
            interval["lower_95"],
            ">= -0.005",
            interval["lower_95"] >= -0.005,
        )
        deltas = {r["class_name"]: r["f1"] for r in c["per_class"]}
        deltas = {r["class_name"]: deltas[r["class_name"]] - r["f1"] for r in p["per_class"]}
        add(scope + ".classes", deltas, "every class >= -0.020", min(deltas.values()) >= -0.020)
        robust = _changes(child, selected) - _changes(g2, selected)
        add(
            scope + ".dark_gain",
            float(robust["brightness_085"]),
            ">= 0.030",
            robust["brightness_085"] >= 0.030,
        )
        add(
            scope + ".translation",
            float(robust["translation_003"]),
            ">= -0.010",
            robust["translation_003"] >= -0.010,
        )
        evidence[scope] = {"candidate": c, "comparison": p, "bootstrap": interval}
    for f in (0, 4):
        delta = child[f]["metrics"]["macro_f1"] - g2[f]["metrics"]["macro_f1"]
        dark = (_changes(child, (f,)) - _changes(g2, (f,)))["brightness_085"]
        add(f"screen.fold_{f}.clean", delta, ">= -0.010", delta >= -0.010)
        add(f"screen.fold_{f}.dark", float(dark), "> 0", dark > 0)

    if phase == "confirmation":
        for scope, selected in (("fresh_vs_e6", (1, 2, 3)), ("five_vs_e6", folds)):
            c, p = [oof_metrics(_pool(r, selected), classes) for r in (child, e6)]
            ci = _paired(child, e6, selected, classes, repetitions)
            deltas = [
                child[f]["metrics"]["macro_f1"] - e6[f]["metrics"]["macro_f1"] for f in selected
            ]
            fresh = scope == "fresh_vs_e6"
            score = c["macro_f1"] - p["macro_f1"] if fresh else c["macro_f1"]
            add(
                scope + ".score",
                score,
                "> 0" if fresh else ">= 0.7462",
                score > 0 if fresh else score >= 0.7462,
            )
            add(scope + ".ci_lower", ci["lower_95"], "> 0", ci["lower_95"] > 0)
            count = sum(d > 0 for d in deltas)
            add(
                scope + ".improved_folds",
                count,
                ">= 2" if fresh else ">= 4",
                count >= (2 if fresh else 4),
            )
            add(scope + ".worst_fold", min(deltas), ">= -0.005", min(deltas) >= -0.005)
            evidence[scope] = {"candidate": c, "comparison": p, "bootstrap": ci}
        metrics = evidence["five_vs_e6"]["candidate"]
        values = {r["class_name"]: r["f1"] for r in metrics["per_class"]}
        gaps = np.array([child[f]["metrics"]["final_train_validation_macro_f1_gap"] for f in folds])
        parents = np.array([e6[f]["metrics"]["final_train_validation_macro_f1_gap"] for f in folds])
        numeric = [
            (
                "fold_sd",
                np.std([child[f]["metrics"]["macro_f1"] for f in folds], ddof=1),
                0.0209,
                False,
            ),
            ("clean_gap", gaps.mean(), 0.2466, False),
            ("gap_reduction", (parents - gaps).mean(), 0.020, True),
            ("worst_gap_worsening", (gaps - parents).max(), 0.005, False),
            (
                "minority_mean",
                np.mean([values[c] for c in ("Boys", "Girls", "Unisex")]),
                0.6013,
                True,
            ),
            ("Men_f1", values["Men"], 0.9188, True),
            ("Women_f1", values["Women"], 0.8950, True),
            ("nll", metrics["nll"], 0.4525, False),
            ("ece_15", metrics["ece_15"], 0.0842, False),
        ]
        for name, value, bound, minimum in numeric:
            add(
                name,
                float(value),
                f"{'>=' if minimum else '<='} {bound}",
                value >= bound if minimum else value <= bound,
            )
        for name, value in (_changes(child, folds) - _changes(e6, folds)).items():
            bound = 0.030 if name == "translation_003" else -0.020
            add("robustness_vs_e6." + name, float(value), f">= {bound}", value >= bound)
    for f in folds:
        m = child[f]["metrics"]
        memory = m["peak_memory_bytes"]
        add(
            f"fold_{f}.gpu_memory_bytes",
            memory,
            f"0 < value < {MEMORY_LIMIT}",
            np.isfinite(memory) and 0 < memory < MEMORY_LIMIT,
        )
        evidence[f"fold_{f}.resources"] = {
            "peak_memory_bytes": memory,
            "latency_ms_batch_1": m["latency_ms_batch_1"],
            "train_seconds": m["train_seconds"],
            "speed_cap": None,
        }
        add(
            f"fold_{f}.parameters",
            m["parameter_count"],
            "== 390181",
            m["parameter_count"] == PARAMETERS,
        )
    return {
        "status": decision(checks),
        "phase": phase,
        "checks": checks,
        "evidence": evidence,
        "independent_test_evidence": False,
        "selection": "pass opens seed confirmation; not final acceptance",
    }


def require_prerequisites(path, *, root, spec, device_name):
    from fashion.train.task3_gender_repair_preflight import verify_prerequisites

    return verify_prerequisites(path, root=Path(root), spec=spec, device_name=device_name)


def run_gender_repair(
    *,
    g2_directory,
    e6_directory,
    source_registry_path,
    output_root,
    registry_path,
    prerequisite_path,
    root=ROOT,
    registry_mirrors=(),
    phase="screen",
    device_name="cuda",
):
    """Run just the two-fold screen, or separately requested fresh confirmation."""
    from fashion.train.task3_baseline import _json_dump, run_task3_baseline_fold
    from fashion.train.task3_dataset_v2 import _reusable_fold

    if phase not in {"screen", "confirmation"}:
        raise ValueError("phase must be screen or confirmation")
    root, output_root = Path(root), Path(output_root)
    sources, classes = load_sources(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        registry_path=source_registry_path,
        root=root,
    )
    spec = dataset_v2_spec(NAME, [sources["G2"][f]["run_id"] for f in range(5)])
    require_prerequisites(prerequisite_path, root=root, spec=spec, device_name=device_name)
    splits = load_splits(root / "data/processed/splits.csv")
    runs = {}

    def inspect(result):
        run = inspect_gender_run(
            Path(result["run_dir"]),
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
        )
        if run["config"]["child_experiment"] != spec.to_dict():
            raise ValueError("candidate contract changed")
        from fashion.data.hashing import compute_sha256

        if run["metrics"].get("prerequisite_sha256") != compute_sha256(prerequisite_path):
            raise ValueError("candidate prerequisite evidence changed")
        return run

    for f in (0, 4):
        result = _reusable_fold(spec, f, output_root=output_root)
        if result is None:
            if phase == "confirmation":
                raise ValueError("confirmation requires the completed, passing two-fold screen")
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
                prerequisite_path=prerequisite_path,
            )
        runs[f] = inspect(result)
    screen = evaluate_gender_repair(runs, sources, classes, phase="screen")
    destination = output_root / spec.artifact_dir / "gender"
    _json_dump(screen, destination / "screen_decision.json")
    if phase == "screen" or screen["status"] != "pass":
        return screen
    for f in (1, 2, 3):
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
                prerequisite_path=prerequisite_path,
            )
        runs[f] = inspect(result)
    report = evaluate_gender_repair(runs, sources, classes, phase="confirmation")
    _json_dump(report, destination / "confirmation_decision.json")
    return report
