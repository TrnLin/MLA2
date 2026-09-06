"""Training-only MixUp with a separate random stream and auditable row coverage."""

import hashlib
import json

import numpy as np

POLICY = {
    "version": "gender_mixup_alpha020_v1",
    "alpha": 0.2,
    "probability": 1.0,
    "lambda": "one Beta(alpha, alpha) draw per batch",
    "partners": "random permutation within the current training batch; self-pairs allowed",
    "placement": "after existing image augmentation and RGB normalization",
    "loss": "lambda * CE(logits, y) + (1-lambda) * CE(logits, partner_y)",
    "rng": "dedicated numpy PCG64; seed XOR 0x4D495855; persists across epochs",
    "evaluation": "unmixed images, plain unweighted cross-entropy",
    "online_train_f1": "not_applicable_mixed_inputs",
}


def training_contract(training, *, validation_fold, seed=2753):
    columns = ["id", "cv_fold", "product_family_group", "gender"]
    required = {*columns, "partition"}
    if validation_fold not in (0, 4) or not required.issubset(training.columns):
        raise ValueError("MixUp needs fold 0 or 4 and complete training metadata")
    if (
        training.empty
        or training[list(required)].isna().any().any()
        or training.id.duplicated().any()
        or not training.partition.eq("development").all()
        or not training.cv_fold.isin(set(range(5)) - {validation_fold}).all()
        or not training.gender.isin(["Boys", "Girls", "Men", "Unisex", "Women"]).all()
    ):
        raise ValueError("MixUp may only use unique valid fold-training rows")
    return {
        "policy": dict(POLICY),
        "seed": seed,
        "validation_fold": validation_fold,
        "training_rows": len(training),
        "training_rows_sha256": hashlib.sha256(
            training[columns].to_csv(index=False).encode()
        ).hexdigest(),
    }


class TrainingMixUp:
    """Keep every row once per epoch and reject any out-of-fold batch before mixing."""

    def __init__(self, training, *, validation_fold, label_to_index, seed=2753):
        self.contract = training_contract(training, validation_fold=validation_fold, seed=seed)
        self.allowed = dict(zip(training.id.astype(int), training.gender.map(label_to_index)))
        if any(not np.isfinite(value) for value in self.allowed.values()):
            raise ValueError("MixUp labels are absent from the class map")
        self.rng = np.random.Generator(np.random.PCG64(seed ^ 0x4D495855))
        self.epochs = []
        self.seen = None

    def begin_epoch(self, epoch):
        if self.seen is not None or epoch != len(self.epochs) + 1:
            raise ValueError("MixUp epochs must be complete and consecutive")
        self.seen = set()
        self.digest = hashlib.sha256()
        self.stats = {
            "epoch": epoch,
            "rows": 0,
            "batches": 0,
            "self_pairs": 0,
            "same_label_pairs": 0,
            "lambda_sum": 0.0,
        }

    def plan(self, ids, labels):
        ids, labels = [int(i) for i in ids], [int(y) for y in labels]
        if (
            self.seen is None
            or not ids
            or len(ids) != len(labels)
            or len(set(ids)) != len(ids)
            or self.seen.intersection(ids)
            or any(self.allowed.get(i) != y for i, y in zip(ids, labels, strict=True))
        ):
            raise ValueError("MixUp batch contains missing, repeated or non-training rows/labels")
        lam = float(self.rng.beta(POLICY["alpha"], POLICY["alpha"]))
        order = self.rng.permutation(len(ids))
        self.seen.update(ids)
        self.stats["rows"] += len(ids)
        self.stats["batches"] += 1
        self.stats["lambda_sum"] += lam
        self.stats["self_pairs"] += int(np.count_nonzero(order == np.arange(len(ids))))
        self.stats["same_label_pairs"] += sum(labels[i] == labels[j] for i, j in enumerate(order))
        self.digest.update(json.dumps([ids, order.tolist(), lam], separators=(",", ":")).encode())
        self.digest.update(b"\n")
        return lam, order

    def apply(self, images, target, ids):
        import torch

        lam, order = self.plan(ids.tolist(), target.detach().cpu().tolist())
        index = torch.as_tensor(order, device=images.device)
        return lam * images + (1.0 - lam) * images[index], target[index], lam

    def end_epoch(self):
        if self.seen != set(self.allowed):
            raise ValueError("MixUp epoch did not cover each training row exactly once")
        self.epochs.append({**self.stats, "mix_plan_sha256": self.digest.hexdigest()})
        self.seen = None

    def receipt(self):
        if self.seen is not None:
            raise ValueError("Cannot save an unfinished MixUp epoch")
        return {"contract": self.contract, "epochs": list(self.epochs)}
