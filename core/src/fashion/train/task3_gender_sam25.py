"""Fixed epoch-25 SAM checkpoint with the original 30-epoch cosine trajectory."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.sam import policy_for_epochs
from fashion.train.task3_gender_name_truth import _run_verified_label_screen, label_contract
from fashion.train.task3_gender_narrow import FOLDS, require_narrow_prerequisites
from fashion.train.task3_gender_sam import SAMSpec, check_gender_sam_sources, verify_sam_evidence
from fashion.train.task3_gender_sam import _source_identity as sam_source_identity
from fashion.train.task3_gender_stronger_mixup import apply_improvement_rules

NAME = "gender_name_truth_mixup_alpha020_sam005_epoch25"
RULE_VERSION = "gname_sam005epoch25_f1floor074_gap005_refine002_v1"
EPOCHS = 25
COSINE_T_MAX = 30


@dataclass(frozen=True)
class SAM25Spec(SAMSpec):
    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "sam005_fixed_epoch25_preserve_cosine30",
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def to_dict(self):
        return dict(
            super().to_dict(),
            sam_policy=policy_for_epochs(EPOCHS),
            screen_rule_version=RULE_VERSION,
            training_epochs=EPOCHS,
            cosine_t_max=COSINE_T_MAX,
            selection_basis="04ai_epoch_diagnostics_post_selection",
        )


def sam25_spec(root=ROOT):
    return SAM25Spec(json.dumps(label_contract(root), sort_keys=True))


def sam25_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, SAM25Spec) or spec.to_dict() != sam25_spec(root).to_dict():
        raise ValueError("SAM epoch-25 differs from its frozen recipe")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("SAM epoch-25 requires CUDA and only folds 0 and 4")
    return replace(Task3BaselineConfig(target="gender"), epochs=EPOCHS)


def training_splits(spec, *, root=ROOT):
    if spec.to_dict() != sam25_spec(root).to_dict():
        raise ValueError("SAM epoch-25 differs from its frozen label contract")
    return load_gender_name_truth_variant(root)


def check_gender_sam25_sources(*, root=ROOT, **paths):
    sources, classes, _, evidence = check_gender_sam_sources(root=root, **paths)
    return sources, classes, sam25_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    identity = sam_source_identity(
        sources, spec, evidence, paths, root=root, splits=training_splits(spec, root=root)
    )
    identity["baseline_controls"] = sam25_config(
        spec, fold=0, device_name="cuda", root=root
    ).to_dict()
    identity["cosine_t_max"] = COSINE_T_MAX
    identity["implementation_sha256"].update(
        {
            name: compute_sha256(Path(root) / "src/fashion/train" / name)
            for name in (
                "task3_gender_sam25.py",
                "task3_g2_audit.py",
                "task3_gender_dropout_darkening.py",
            )
        }
    )
    return identity


def require_sam25_prerequisites(
    path, *, spec, fold, parent_run_directory=None, root=ROOT, device_name="cuda"
):
    sam25_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("SAM epoch-25 source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked, evidence = check_gender_sam25_sources(**paths, root=root)
    if (
        checked.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("SAM epoch-25 prerequisite evidence changed")
    parent = Path(sources["MixUp20"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("SAM epoch-25 requires its verified 04af parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def verify_sam25_evidence(run, *, fold, splits, directory):
    verify_sam_evidence(run, fold=fold, splits=splits, directory=directory, epochs=EPOCHS)
    config, metrics = run["config"], run["metrics"]
    history = pd.read_csv(Path(directory) / "history.csv")
    expected_lr = 0.00001 + 0.5 * (0.001 - 0.00001) * (
        1 + np.cos(np.pi * np.arange(EPOCHS) / COSINE_T_MAX)
    )
    if (
        config.get("epochs") != EPOCHS
        or config.get("cosine_t_max") != COSINE_T_MAX
        or metrics.get("epochs_completed") != EPOCHS
        or metrics.get("selected_epoch") != EPOCHS
        or metrics.get("checkpoint_policy") != "final_epoch"
        or metrics.get("early_stopped") is not False
        or not np.allclose(history.learning_rate, expected_lr, rtol=0, atol=1e-12)
        or history.selected_checkpoint.tolist() != [False] * (EPOCHS - 1) + [True]
    ):
        raise ValueError("SAM epoch-25 history, checkpoint or 30-epoch cosine schedule changed")


def run_gender_sam25_screen(
    *, output_root, registry_path, registry_mirrors=(), root=ROOT, device_name="cuda", **paths
):
    sources, classes, spec, evidence = check_gender_sam25_sources(root=root, **paths)
    sam25_config(spec, fold=0, device_name=device_name, root=root)
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
        candidate_group="SAM005Epoch25",
        verify_candidate=verify_sam25_evidence,
        refine_report=apply_improvement_rules,
        candidate_epochs=EPOCHS,
    )
