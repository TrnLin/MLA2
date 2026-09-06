"""One frozen MixUp screen against the completed unweighted name-truth parents."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import POLICY, training_contract
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_dropout_darkening import _verify_training_evidence
from fashion.train.task3_gender_name_truth import NAME as PARENT_NAME
from fashion.train.task3_gender_name_truth import PARENT_RUN_IDS as GRAYSCALE_RUN_IDS
from fashion.train.task3_gender_name_truth import (
    NameTruthSpec,
    _run_verified_label_screen,
    check_gender_name_truth_sources,
    label_contract,
)
from fashion.train.task3_gender_name_truth import _source_identity as parent_source_identity
from fashion.train.task3_gender_narrow import FOLDS, require_narrow_prerequisites

NAME = "gender_name_truth_mixup_alpha020"
RULE_VERSION = "gname_mixup020_loss003_gap005_v1"
PARENT_RUN_IDS = (
    "t3_gender_name_truth_dropout_030_grayscale_010_gender_smallcnngem3_f0_s2753_"
    "416e64f58f57_20260905T160634Z0acdc0",
    "t3_gender_name_truth_dropout_030_grayscale_010_gender_smallcnngem3_f4_s2753_"
    "416e64f58f57_20260905T161722Za0a0f7",
)


@dataclass(frozen=True)
class MixUpSpec(NameTruthSpec):
    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "training_batch_mixup_alpha_020",
            "parent_artifact_dir": f"experiments/t3_{PARENT_NAME}",
            "parent_run_ids": PARENT_RUN_IDS,
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def parent_run_id_for_fold(self, fold):
        if fold not in FOLDS:
            raise ValueError("MixUp screen allows only folds 0 and 4")
        return PARENT_RUN_IDS[FOLDS.index(fold)]

    def to_dict(self):
        payload = super().to_dict()
        payload.update(
            parent_run_ids=list(PARENT_RUN_IDS),
            mixup_policy=dict(POLICY),
            screen_rule_version=RULE_VERSION,
        )
        return payload


def mixup_spec(root=ROOT):
    return MixUpSpec(json.dumps(label_contract(root), sort_keys=True))


def mixup_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, MixUpSpec) or spec.to_dict() != mixup_spec(root).to_dict():
        raise ValueError("MixUp policy or frozen name-truth recipe changed")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("MixUp requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def training_splits(spec, *, root=ROOT):
    if spec.to_dict() != mixup_spec(root).to_dict():
        raise ValueError("MixUp differs from the frozen label contract")
    return load_gender_name_truth_variant(root)


def check_gender_mixup_sources(*, name_truth_directory, root=ROOT, **paths):
    sources, classes, old_spec, evidence = check_gender_name_truth_sources(root=root, **paths)
    directory = Path(name_truth_directory)
    audit = directory / "source_audit.json"
    previous = json.loads(audit.read_text())["identity"]
    if (
        parent_source_identity(sources, old_spec, evidence, previous["paths"], root=root)
        != previous
    ):
        raise ValueError("Completed name-truth source audit differs from verified parents")
    splits = load_gender_name_truth_variant(root)
    registry = pd.read_csv(paths["source_registry_path"], keep_default_na=False)
    direct = {}
    for fold, run_id, gray_id in zip(FOLDS, PARENT_RUN_IDS, GRAYSCALE_RUN_IDS, strict=True):
        run = inspect_gender_run(
            directory / run_id, registry=registry, splits=splits, classes=classes, root=root
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("Completed name-truth parent is assigned to the wrong fold")
        _verify_training_evidence(
            run, old_spec, gray_id, evidence, audit_sha256=compute_sha256(audit)
        )
        if run["config"].get("gender_label_variant") != old_spec.to_dict()["gender_label_variant"]:
            raise ValueError("Direct parent gender labels differ from the verified variant")
        run["directory"] = str(directory / run_id)
        direct[fold] = run
    sources["NameTruth"] = direct
    return sources, classes, mixup_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    identity = parent_source_identity(
        sources, NameTruthSpec(spec.label_contract_json), evidence, paths, root=root
    )
    splits = training_splits(spec, root=root)
    identity.update(
        spec=spec.to_dict(),
        rule_version=RULE_VERSION,
        mixup_contracts={
            str(f): training_contract(
                get_samples(get_cv_split(splits, f)[0], target="gender"), validation_fold=f
            )
            for f in FOLDS
        },
        mixup_implementation_sha256={
            name: compute_sha256(Path(root) / "src/fashion/train" / name)
            for name in ("mixup.py", "task3_gender_mixup.py", "task3_baseline.py")
        },
    )
    return identity


def require_mixup_prerequisites(
    path, *, spec, fold, parent_run_directory=None, root=ROOT, device_name="cuda"
):
    mixup_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("MixUp source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked, evidence = check_gender_mixup_sources(**paths, root=root)
    if (
        checked.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("MixUp prerequisite evidence changed")
    parent = Path(sources["NameTruth"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("MixUp requires its verified name-truth parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def verify_mixup_evidence(run, *, fold, splits, directory, alpha=0.2):
    training = get_samples(get_cv_split(splits, fold)[0], target="gender")
    expected = training_contract(training, validation_fold=fold, alpha=alpha)
    config = Task3BaselineConfig(target="gender")
    path = Path(directory) / "mixup_training.json"
    receipt = json.loads(path.read_text())
    history = pd.read_csv(Path(directory) / "history.csv")
    if (
        run["config"].get("mixup_contract") != expected
        or run["metrics"].get("mixup_receipt_sha256") != compute_sha256(path)
        or receipt.get("contract") != expected
        or len(receipt.get("epochs", [])) != config.epochs
        or not history.train_macro_f1.isna().all()
        or not history.train_metric_scope.eq(POLICY["online_train_f1"]).all()
    ):
        raise ValueError("Saved MixUp training evidence differs from the frozen contract")
    for epoch, row in enumerate(receipt["epochs"], start=1):
        batches = math.ceil(len(training) / config.batch_size)
        if (
            row.get("epoch") != epoch
            or row.get("rows") != len(training)
            or row.get("batches") != batches
            or not 0
            <= row.get("self_pairs", -1)
            <= row.get("same_label_pairs", -1)
            <= len(training)
            or not np.isfinite(row.get("lambda_sum", np.nan))
            or not 0 <= row["lambda_sum"] <= batches
            or len(row.get("mix_plan_sha256", "")) != 64
        ):
            raise ValueError("Saved MixUp epoch coverage is incomplete or invalid")


def run_gender_mixup_screen(
    *,
    name_truth_directory,
    output_root,
    registry_path,
    registry_mirrors=(),
    root=ROOT,
    device_name="cuda",
    **paths,
):
    paths = dict(paths, name_truth_directory=name_truth_directory)
    sources, classes, spec, evidence = check_gender_mixup_sources(**paths, root=root)
    mixup_config(spec, fold=0, device_name=device_name, root=root)
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
        parent_group="NameTruth",
        candidate_group="MixUp",
        verify_candidate=verify_mixup_evidence,
    )
