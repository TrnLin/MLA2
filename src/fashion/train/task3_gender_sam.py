"""One SAM trial against the completed alpha 0.2 models, with frozen gap gates."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.mixup import POLICY as MIXUP_POLICY
from fashion.train.mixup import training_contract
from fashion.train.sam import POLICY, policy_for_epochs
from fashion.train.task3_gender_mixup import verify_mixup_evidence
from fashion.train.task3_gender_name_truth import (
    NameTruthSpec,
    _run_verified_label_screen,
    label_contract,
)
from fashion.train.task3_gender_name_truth import _source_identity as label_source_identity
from fashion.train.task3_gender_narrow import FOLDS, require_narrow_prerequisites
from fashion.train.task3_gender_stronger_mixup import (
    PARENT_COMMIT,
    MixUp40Spec,
    apply_improvement_rules,
    check_gender_stronger_mixup_sources,
)

NAME = "gender_name_truth_mixup_alpha020_sam005"
RULE_VERSION = "gname_sam005_f1floor074_gap005_refine002_v1"


@dataclass(frozen=True)
class SAMSpec(MixUp40Spec):
    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "adamw_to_sam_rho005_adamw",
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def to_dict(self):
        return dict(
            super().to_dict(),
            mixup_policy=dict(MIXUP_POLICY),
            sam_policy=dict(POLICY),
            screen_rule_version=RULE_VERSION,
        )


def sam_spec(root=ROOT):
    return SAMSpec(json.dumps(label_contract(root), sort_keys=True))


def sam_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, SAMSpec) or spec.to_dict() != sam_spec(root).to_dict():
        raise ValueError("SAM differs from the frozen recipe")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("SAM requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def training_splits(spec, *, root=ROOT):
    if spec.to_dict() != sam_spec(root).to_dict():
        raise ValueError("SAM differs from the frozen label contract")
    return load_gender_name_truth_variant(root)


def check_gender_sam_sources(*, root=ROOT, **paths):
    # This audits the completed 04af parents, not the failed 04ah candidates.
    sources, classes, _, evidence = check_gender_stronger_mixup_sources(root=root, **paths)
    return sources, classes, sam_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root, splits=None):
    identity = label_source_identity(
        sources, NameTruthSpec(spec.label_contract_json), evidence, paths, root=root
    )
    if splits is None:
        splits = training_splits(spec, root=root)
    identity.update(
        spec=spec.to_dict(),
        rule_version=spec.to_dict()["screen_rule_version"],
        mixup20_code_commit=PARENT_COMMIT,
        parent_screen_decision_sha256=compute_sha256(
            Path(paths["mixup_directory"]) / "screen_decision.json"
        ),
        mixup_contracts={
            str(f): training_contract(
                get_samples(get_cv_split(splits, f)[0], target="gender"), validation_fold=f
            )
            for f in FOLDS
        },
        implementation_sha256={
            name: compute_sha256(Path(root) / "src/fashion/train" / name)
            for name in (
                "sam.py",
                "task3_gender_sam.py",
                "mixup.py",
                "task3_gender_mixup.py",
                "task3_gender_stronger_mixup.py",
                "task3_baseline.py",
                "task3_gender_name_truth.py",
                "task3_gender_narrow.py",
            )
        },
    )
    return identity


def require_sam_prerequisites(
    path, *, spec, fold, parent_run_directory=None, root=ROOT, device_name="cuda"
):
    sam_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("SAM source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked, evidence = check_gender_sam_sources(**paths, root=root)
    if (
        checked.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("SAM prerequisite evidence changed")
    parent = Path(sources["MixUp20"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("SAM requires its verified 04af parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def verify_sam_evidence(run, *, fold, splits, directory, epochs=30):
    expected_policy = policy_for_epochs(epochs)
    verify_mixup_evidence(run, fold=fold, splits=splits, directory=directory, epochs=epochs)
    directory = Path(directory)
    path = directory / "sam_training.json"
    receipt = json.loads(path.read_text())
    mixup = json.loads((directory / "mixup_training.json").read_text())
    history = pd.read_csv(directory / "history.csv")
    diagnostics = json.loads((directory / "clean_epoch_diagnostics.json").read_text())
    if (
        receipt.get("policy") != expected_policy
        or run["config"].get("sam_policy") != expected_policy
        or run["metrics"].get("sam_receipt_sha256") != compute_sha256(path)
        or run["metrics"].get("clean_epoch_diagnostics_sha256")
        != compute_sha256(directory / "clean_epoch_diagnostics.json")
        or len(receipt.get("epochs", [])) != epochs
        or [row["epoch"] for row in diagnostics] != expected_policy["diagnostic_epochs"]
        or history.epoch.tolist() != list(range(1, epochs + 1))
    ):
        raise ValueError("Saved SAM training evidence differs from the frozen contract")
    for row, mixed, (_, epoch) in zip(
        receipt["epochs"], mixup["epochs"], history.iterrows(), strict=True
    ):
        if (
            any(row.get(k) != mixed[k] for k in ("epoch", "rows", "batches"))
            or row.get("forward_backward_passes") != 2 * mixed["batches"]
            or row.get("optimizer_steps") != mixed["batches"]
            or not all(
                np.isfinite(row.get(k, np.nan))
                for k in ("first_loss", "second_loss", "gradient_norm_min", "gradient_norm_max")
            )
            or not 0 <= row["gradient_norm_min"] <= row["gradient_norm_max"]
            or not np.isclose(row["first_loss"], epoch.train_loss, rtol=0, atol=1e-12)
            or not np.isclose(row["second_loss"], epoch.sam_second_loss, rtol=0, atol=1e-12)
        ):
            raise ValueError("Saved SAM step accounting or loss trace is invalid")


def run_gender_sam_screen(
    *, output_root, registry_path, registry_mirrors=(), root=ROOT, device_name="cuda", **paths
):
    sources, classes, spec, evidence = check_gender_sam_sources(root=root, **paths)
    sam_config(spec, fold=0, device_name=device_name, root=root)
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
        candidate_group="SAM005",
        verify_candidate=verify_sam_evidence,
        refine_report=apply_improvement_rules,
    )
