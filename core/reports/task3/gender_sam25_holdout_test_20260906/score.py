"""Score frozen SAM25 predictions against the requested external references."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fashion.config import ROOT, TEACHER_TRAIN_CSV
from fashion.data.families import normalize_product_name
from fashion.data.gender_name_truth import product_name_gender_cues
from fashion.data.hashing import compute_sha256
from fashion.train.metrics import classification_metrics
from fashion.train.task3_decisions import probability_columns

REPORT = Path(__file__).resolve().parent
HIGHRES = ROOT / "data/raw/external/fashion_product_images_v1_legacy_extras/fashion-dataset"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def references(path, requested_ids):
    records, extra_fields = {}, []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header[:10] == [
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
            "productDisplayName",
        ]
        for row in reader:
            if not row or int(row[0]) not in requested_ids:
                continue
            product_id = int(row[0])
            assert product_id not in records and len(row) >= 10
            name_fields = row[9:]
            while len(name_fields) > 1 and name_fields[-1] == "":
                name_fields.pop()
            records[product_id] = {
                "gender": row[1].strip(),
                "productDisplayName": ",".join(name_fields).strip(),
                "articleType": row[4].strip(),
            }
            if len(row) > len(header):
                extra_fields.append(product_id)
    assert set(records) == requested_ids
    return records, extra_fields


def probabilities(scope, classes):
    frame = pd.read_csv(
        REPORT / f"gender_{scope}_probabilities.csv",
        keep_default_na=False,
        float_precision="round_trip",
    )
    labels = pd.read_csv(REPORT / f"gender_{scope}_predictions.csv", keep_default_na=False)
    manifest = pd.read_csv(REPORT / f"{scope}_image_manifest.csv", keep_default_na=False)
    assert frame.id.tolist() == labels.id.tolist() == manifest.id.tolist()
    assert frame.id.is_unique
    values = frame[probability_columns(classes)].to_numpy(dtype=float)
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)
    assert frame.gender.tolist() == labels.gender.tolist()
    assert np.asarray(classes)[values.argmax(axis=1)].tolist() == labels.gender.tolist()
    folds = []
    for fold in range(5):
        saved = pd.read_csv(
            REPORT / f"gender_{scope}_fold_{fold}.csv", float_precision="round_trip"
        )
        assert saved.id.tolist() == frame.id.tolist()
        folds.append(saved[probability_columns(classes)].to_numpy(dtype=float))
    np.testing.assert_array_equal(np.mean(np.stack(folds), axis=0), values)
    return frame, values, folds


def calculate(truth, values, classes):
    valid = truth.isin(classes).to_numpy()
    assert valid.all(), "Missing or unknown reference gender; inspect before scoring."
    labels = truth.map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
    metrics = classification_metrics(labels, values, classes)
    predicted = np.asarray(classes)[values.argmax(axis=1)]
    np.testing.assert_allclose(metrics["accuracy"], accuracy_score(truth, predicted))
    np.testing.assert_allclose(
        metrics["macro_f1"],
        f1_score(truth, predicted, labels=classes, average="macro", zero_division=0),
    )
    metrics["correct"] = int(np.sum(truth.to_numpy() == predicted))
    metrics["incorrect"] = len(truth) - metrics["correct"]
    return metrics


def main():
    if (REPORT / "evaluation_metrics.json").exists():
        raise RuntimeError("This evaluation is complete; preserve the first saved scores.")
    freeze = read(REPORT / "prediction_freeze.json")
    assert freeze["holdout_and_test_reference_labels_opened"] is False
    for name, digest in freeze["outputs"].items():
        assert compute_sha256(REPORT / name) == digest, name
    recipe = read(REPORT / "inference_recipe.json")
    for name, digest in recipe["input_sha256"].items():
        assert compute_sha256(resolve_task3_path(name, root=ROOT)) == digest, name
    model_root = (resolve_task3_path(recipe["model_manifest"], root=ROOT)).parent
    for fold in recipe["folds"]:
        for artifact in fold["files"].values():
            assert compute_sha256(model_root / artifact["path"]) == artifact["sha256"]
    classes = recipe["classes"]
    frames = {scope: probabilities(scope, classes) for scope in ("holdout", "test")}
    holdout_ids, test_ids = (set(frames[scope][0].id) for scope in ("holdout", "test"))
    opened_at = now()
    assert opened_at > freeze["frozen_at_utc"]
    teacher, teacher_extra = references(TEACHER_TRAIN_CSV, holdout_ids)
    highres, highres_extra = references(HIGHRES / "styles.csv", holdout_ids | test_ids)
    json_disagreements = []
    json_hashes = {}
    for product_id, reference in highres.items():
        path = HIGHRES / "styles" / f"{product_id}.json"
        payload = read(path)
        product = payload.get("data", payload)
        assert int(product["id"]) == product_id
        actual = str(product["gender"]).strip()
        reference["json_gender"] = actual
        if actual != reference["gender"]:
            json_disagreements.append(product_id)
        image = HIGHRES / "images" / f"{product_id}.jpg"
        assert image.is_file(), product_id
        reference["highres_image_path"] = str(image.relative_to(ROOT))
        json_hashes[str(path.relative_to(ROOT))] = compute_sha256(path)
    assert not json_disagreements, json_disagreements
    splits = pd.read_csv(
        ROOT / "data/processed/splits.csv",
        usecols=["id", "partition", "product_name_key"],
        keep_default_na=False,
    )
    development_names = set(splits.loc[splits.partition.eq("development"), "product_name_key"]) - {
        ""
    }
    rows, class_rows, fold_rows, comparison_frames = [], [], [], []
    result = {}
    for scope, (predictions, values, fold_values) in frames.items():
        comparison = predictions[["id", "gender", "confidence"]].rename(
            columns={"gender": "predicted_gender"}
        )
        comparison["scope"] = scope
        comparison["highres_gender"] = comparison.id.map(lambda i: highres[i]["gender"])
        comparison["highres_json_gender"] = comparison.id.map(lambda i: highres[i]["json_gender"])
        comparison["highres_image_path"] = comparison.id.map(
            lambda i: highres[i]["highres_image_path"]
        )
        source = teacher if scope == "holdout" else highres
        comparison["reference_gender"] = comparison.id.map(lambda i: source[i]["gender"])
        comparison["productDisplayName"] = comparison.id.map(
            lambda i: source[i]["productDisplayName"]
        )
        comparison["articleType"] = comparison.id.map(lambda i: source[i]["articleType"])
        cues = product_name_gender_cues(comparison.productDisplayName)
        comparison["name_gender"] = cues.name_gender
        comparison["name_cue_count"] = cues.cue_count
        comparison["name_rule_gender"] = cues.name_gender.where(
            cues.cue_count.eq(1), comparison.reference_gender
        )
        comparison["name_rule_changes_reference"] = comparison.name_rule_gender.ne(
            comparison.reference_gender
        )
        comparison["correct"] = comparison.predicted_gender.eq(comparison.reference_gender)
        comparison["normalized_name_in_development"] = comparison.productDisplayName.map(
            normalize_product_name
        ).isin(development_names)
        comparisons = {
            "primary": comparison.reference_gender,
            "highres_raw": comparison.highres_gender,
            "fixed_name_rule_diagnostic": comparison.name_rule_gender,
        }
        for label_basis, truth in comparisons.items():
            metrics = calculate(truth, values, classes)
            key = f"{scope}/{label_basis}"
            result[key] = metrics
            rows.append(
                {
                    "scope": scope,
                    "label_basis": label_basis,
                    "images": len(comparison),
                    "correct": metrics["correct"],
                    "incorrect": metrics["incorrect"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "nll": metrics["nll"],
                    "ece_15": metrics["ece_15"],
                }
            )
            for row in metrics["per_class"]:
                class_rows.append({"scope": scope, "label_basis": label_basis, **row})
            pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
                REPORT / f"{scope}_{label_basis}_confusion_matrix.csv", index_label="actual"
            )
        for fold, single in enumerate(fold_values):
            metrics = calculate(comparison.reference_gender, single, classes)
            fold_rows.append(
                {
                    "scope": scope,
                    "fold": fold,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                }
            )
        comparison.to_csv(REPORT / f"{scope}_predictions_and_labels.csv", index=False)
        comparison.loc[~comparison.correct].to_csv(REPORT / f"{scope}_errors.csv", index=False)
        comparison_frames.append(comparison)
    all_comparisons = pd.concat(comparison_frames, ignore_index=True)
    all_comparisons.to_csv(REPORT / "reference_label_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORT / "evaluation_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(REPORT / "per_class.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REPORT / "individual_fold_scores.csv", index=False)
    audit = {
        "reference_labels_opened_at_utc": opened_at,
        "prediction_freeze_at_utc": freeze["frozen_at_utc"],
        "teacher_reference": str(TEACHER_TRAIN_CSV.relative_to(ROOT)),
        "teacher_reference_sha256": compute_sha256(TEACHER_TRAIN_CSV),
        "highres_reference": str((HIGHRES / "styles.csv").relative_to(ROOT)),
        "highres_reference_sha256": compute_sha256(HIGHRES / "styles.csv"),
        "highres_csv_json_gender_disagreements": json_disagreements,
        "highres_image_files_present": len(highres),
        "highres_json_sha256": json_hashes,
        "teacher_extra_csv_fields": teacher_extra,
        "highres_extra_csv_fields": highres_extra,
        "holdout_teacher_highres_label_differences": int(
            comparison_frames[0].reference_gender.ne(comparison_frames[0].highres_gender).sum()
        ),
        "name_rule_label_changes": {
            scope: int(frame.name_rule_changes_reference.sum())
            for scope, frame in zip(("holdout", "test"), comparison_frames, strict=True)
        },
        "rows_with_normalized_name_in_development": {
            scope: int(frame.normalized_name_in_development.sum())
            for scope, frame in zip(("holdout", "test"), comparison_frames, strict=True)
        },
        "note": (
            "Teacher test family IDs are not present in the canonical family audit. "
            "Exact IDs and files are disjoint; normalized-name overlaps are reported, "
            "not removed from the assigned test set."
        ),
        "score_code_sha256": compute_sha256(Path(__file__)),
    }
    save_json(audit, REPORT / "reference_audit.json")
    save_json({"completed_at_utc": now(), "metrics": result}, REPORT / "evaluation_metrics.json")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("Label changes:", audit["name_rule_label_changes"], flush=True)
    print(
        "Teacher vs high-res holdout differences:",
        audit["holdout_teacher_highres_label_differences"],
        flush=True,
    )
    print(
        "Development name overlaps:", audit["rows_with_normalized_name_in_development"], flush=True
    )


if __name__ == "__main__":
    main()
