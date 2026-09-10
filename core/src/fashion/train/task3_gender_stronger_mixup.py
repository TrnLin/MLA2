"""A stronger MixUp trial must improve on 04af, not merely pass the older G2 gates."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import policy_for_alpha, training_contract
from fashion.train.task3_decisions import check, decision
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_dropout_darkening import _verify_training_evidence
from fashion.train.task3_gender_mixup import NAME as PARENT_NAME
from fashion.train.task3_gender_mixup import (
    MixUpSpec,
    check_gender_mixup_sources,
    verify_mixup_evidence,
)
from fashion.train.task3_gender_mixup import _source_identity as mixup20_source_identity
from fashion.train.task3_gender_name_truth import (
    NameTruthSpec,
    _run_verified_label_screen,
    label_contract,
)
from fashion.train.task3_gender_name_truth import _source_identity as label_source_identity
from fashion.train.task3_gender_narrow import FOLDS, require_narrow_prerequisites

NAME = "gender_name_truth_mixup_alpha040"
RULE_VERSION = "gname_mixup040_f1floor074_gap005_refine002_v2"
PARENT_COMMIT = "68fef49ab1d55d671531113a71a3e71400a0e3fc"
PARENT_RUN_IDS = (
    "t3_gender_name_truth_mixup_alpha020_gender_smallcnngem3_f0_s2753_"
    "4566a61e0e2d_20260906T070013Z8f7ca5",
    "t3_gender_name_truth_mixup_alpha020_gender_smallcnngem3_f4_s2753_"
    "4566a61e0e2d_20260906T071118Z37920b",
)
IMPROVEMENT_RULES = {
    "comparison": "completed MixUp alpha 0.2 on identical name-truth labels",
    "minimum_mean_clean_gap_reduction": 0.020,
    "each_fold_gap_must_shrink": True,
    "minimum_pooled_validation_f1": 0.74,
    "relative_f1_comparisons_are_diagnostic_only": True,
    "minimum_unisex_recall_delta": 0.0,
    "retain_non_f1_g2_e6_gates": True,
}
DIAGNOSTIC_GATES = frozenset(
    {"validation_delta", "validation_ci_lower", "class_f1"}
    | {f"fold_{fold}.validation_delta" for fold in FOLDS}
)


@dataclass(frozen=True)
class MixUp40Spec(MixUpSpec):
    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "mixup_alpha_020_to_040",
            "parent_artifact_dir": f"experiments/t3_{PARENT_NAME}",
            "parent_run_ids": PARENT_RUN_IDS,
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def parent_run_id_for_fold(self, fold):
        if fold not in FOLDS:
            raise ValueError("Stronger MixUp allows only folds 0 and 4")
        return PARENT_RUN_IDS[FOLDS.index(fold)]

    def to_dict(self):
        payload = super().to_dict()
        payload.update(
            parent_run_ids=list(PARENT_RUN_IDS),
            mixup_policy=policy_for_alpha(0.4),
            screen_rule_version=RULE_VERSION,
            improvement_rules=dict(IMPROVEMENT_RULES),
        )
        return payload


def mixup40_spec(root=ROOT):
    return MixUp40Spec(json.dumps(label_contract(root), sort_keys=True))


def mixup40_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, MixUp40Spec) or spec.to_dict() != mixup40_spec(root).to_dict():
        raise ValueError("Stronger MixUp differs from the frozen recipe")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("Stronger MixUp requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def training_splits(spec, *, root=ROOT):
    if spec.to_dict() != mixup40_spec(root).to_dict():
        raise ValueError("Stronger MixUp differs from the frozen label contract")
    return load_gender_name_truth_variant(root)


def recorded_mixup20_hashes(root):
    """The completed parent's code predates this trial; check its immutable commit."""
    return {
        name: hashlib.sha256(
            subprocess.check_output(
                ["git", "show", f"{PARENT_COMMIT}:src/fashion/train/{name}"], cwd=root
            )
        ).hexdigest()
        for name in ("mixup.py", "task3_gender_mixup.py", "task3_baseline.py")
    }


def check_gender_stronger_mixup_sources(*, mixup_directory, root=ROOT, **paths):
    sources, classes, old_spec, evidence = check_gender_mixup_sources(root=root, **paths)
    directory = Path(mixup_directory)
    audit = directory / "source_audit.json"
    previous = json.loads(audit.read_text())["identity"]
    expected = mixup20_source_identity(sources, old_spec, evidence, previous["paths"], root=root)
    expected["mixup_implementation_sha256"] = recorded_mixup20_hashes(root)
    if expected != previous:
        raise ValueError("Completed MixUp source audit differs from its frozen code or parents")
    saved_decision = json.loads((directory / "screen_decision.json").read_text())
    if saved_decision.get("status") != "pass" or saved_decision.get("run_ids") != dict(
        zip(map(str, FOLDS), PARENT_RUN_IDS)
    ):
        raise ValueError("Stronger MixUp requires the completed passing 04af screen")
    splits = load_gender_name_truth_variant(root)
    registry = pd.read_csv(paths["source_registry_path"], keep_default_na=False)
    direct = {}
    for fold, run_id in zip(FOLDS, PARENT_RUN_IDS, strict=True):
        run = inspect_gender_run(
            directory / run_id, registry=registry, splits=splits, classes=classes, root=root
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("MixUp parent is assigned to the wrong fold")
        _verify_training_evidence(
            run,
            old_spec,
            old_spec.parent_run_id_for_fold(fold),
            evidence,
            audit_sha256=compute_sha256(audit),
        )
        verify_mixup_evidence(run, fold=fold, splits=splits, directory=directory / run_id)
        if run["config"].get("gender_label_variant") != old_spec.to_dict()["gender_label_variant"]:
            raise ValueError("Completed MixUp parent uses different labels")
        run["directory"] = str(directory / run_id)
        direct[fold] = run
    sources["MixUp20"] = direct
    return sources, classes, mixup40_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    identity = label_source_identity(
        sources, NameTruthSpec(spec.label_contract_json), evidence, paths, root=root
    )
    splits = training_splits(spec, root=root)
    identity.update(
        spec=spec.to_dict(),
        rule_version=RULE_VERSION,
        mixup20_code_commit=PARENT_COMMIT,
        parent_screen_decision_sha256=compute_sha256(
            Path(paths["mixup_directory"]) / "screen_decision.json"
        ),
        mixup_contracts={
            str(f): training_contract(
                get_samples(get_cv_split(splits, f)[0], target="gender"),
                validation_fold=f,
                alpha=0.4,
            )
            for f in FOLDS
        },
        implementation_sha256={
            name: compute_sha256(Path(root) / "src/fashion/train" / name)
            for name in (
                "mixup.py",
                "task3_gender_stronger_mixup.py",
                "task3_gender_mixup.py",
                "task3_baseline.py",
                "task3_gender_name_truth.py",
            )
        },
    )
    return identity


def require_mixup40_prerequisites(
    path, *, spec, fold, parent_run_directory=None, root=ROOT, device_name="cuda"
):
    mixup40_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("Stronger MixUp source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked, evidence = check_gender_stronger_mixup_sources(**paths, root=root)
    if (
        checked.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("Stronger MixUp prerequisite evidence changed")
    parent = Path(sources["MixUp20"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("Stronger MixUp requires its verified 04af parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def verify_mixup40_evidence(run, **kwargs):
    verify_mixup_evidence(run, **kwargs, alpha=0.4)


def apply_improvement_rules(report):
    """A smaller training score alone cannot pass this refinement."""
    result = dict(
        report,
        checks=[c for c in report["checks"] if c["gate"] not in DIAGNOSTIC_GATES],
        diagnostic_checks=[c for c in report["checks"] if c["gate"] in DIAGNOSTIC_GATES],
        historical_screen_status=report["status"],
        improvement_rules=dict(IMPROVEMENT_RULES),
        required_validation_f1=IMPROVEMENT_RULES["minimum_pooled_validation_f1"],
    )
    comparison = report["incremental_comparison"]

    def add(name, value, threshold):
        result["checks"].append(
            check(
                "vs_mixup20." + name,
                value,
                f">= {threshold}",
                bool(np.isfinite(value) and value + 1e-12 >= threshold),
            )
        )

    for row in comparison["folds"]:
        reduction = row["gap_reduction"]
        result["checks"].append(
            check(
                f"vs_mixup20.fold_{row['fold']}.gap_reduction",
                reduction,
                "> 0",
                bool(np.isfinite(reduction) and reduction > 0),
            )
        )
    add(
        "mean_gap_reduction",
        float(np.mean([r["gap_reduction"] for r in comparison["folds"]])),
        0.020,
    )
    add(
        "pooled_validation_f1",
        comparison["candidate"]["macro_f1"],
        IMPROVEMENT_RULES["minimum_pooled_validation_f1"],
    )
    unisex = {
        k: next(r for r in comparison[k]["per_class"] if r["class_name"] == "Unisex")
        for k in ("candidate", "dropout")
    }
    add("unisex_recall_delta", unisex["candidate"]["recall"] - unisex["dropout"]["recall"], 0.0)
    result["status"] = decision(result["checks"])
    result["next_step"] = (
        "Stop after folds 0 and 4. Review clean gaps and validation before any further trial."
    )
    return result


def run_gender_stronger_mixup_screen(
    *,
    mixup_directory,
    output_root,
    registry_path,
    registry_mirrors=(),
    root=ROOT,
    device_name="cuda",
    **paths,
):
    paths = dict(paths, mixup_directory=mixup_directory)
    sources, classes, spec, evidence = check_gender_stronger_mixup_sources(**paths, root=root)
    mixup40_config(spec, fold=0, device_name=device_name, root=root)
    require_narrow_prerequisites(paths["precision_directory"], root=root)
    return _run_verified_label_screen(
        sources=sources,
        classes=classes,
        spec=spec,
        evidence=evidence,
        identity=_source_identity(sources, spec, evidence, paths, root=root),
        splits=training_splits(spec, root=root),
        source_registry_path=paths["source_registry_path"],
        output_root=output_root,
        registry_path=registry_path,
        registry_mirrors=registry_mirrors,
        root=root,
        device_name=device_name,
        parent_group="MixUp20",
        candidate_group="MixUp40",
        verify_candidate=verify_mixup40_evidence,
        refine_report=apply_improvement_rules,
    )
