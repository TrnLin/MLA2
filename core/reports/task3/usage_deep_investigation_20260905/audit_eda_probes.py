"""Read back every saved Usage view probe; do not fit or certify stale contracts."""

import json

import pandas as pd
from analyse_evidence import CLASSES, OUT, ROOT, sha
from sklearn.metrics import f1_score


def main():
    source = ROOT / "results/evidence/task3/clean_slate_eda/view_probes"
    contract = json.loads((source / "view_probe_contract.json").read_text())
    split = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)
    dev = split.loc[split.partition.eq("development")].set_index("id")
    rows = []
    for path in sorted(source.glob("usage_*_oof.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        expected = dev.loc[frame.id]
        assert not frame.id.duplicated().any()
        assert (frame.cv_fold.astype(int).to_numpy() == expected.cv_fold.astype(int)).all()
        target = frame.true_label.replace("", "NA")
        assert (target.to_numpy() == expected.usage).all()
        digest = sha(path)
        current = path.name in contract["artifact_sha256"]
        if current:
            assert contract["artifact_sha256"][path.name] == digest
        rows.append(
            {
                "view": path.name.removeprefix("usage_").removesuffix("_oof.csv"),
                "rows": len(frame),
                "pooled_macro_f1_nine": f1_score(
                    target, frame.predicted_label, labels=CLASSES, average="macro", zero_division=0
                ),
                "accuracy": target.eq(frame.predicted_label).mean(),
                "listed_in_current_view_contract": current,
                "sha256": digest,
                "scope": "selected diagnostic sample, not a full candidate score",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "eda_view_comparison.csv", index=False)
    registry = pd.read_csv(ROOT / "results/runs.csv", keep_default_na=False)
    registry = registry.loc[
        registry.target.eq("usage") & registry.experiment_id.eq("t3_clean_slate_eda_view_probe")
    ].copy()
    for key in ["macro_f1", "accuracy", "support"]:
        registry[key] = registry.metrics_json.map(lambda v, field=key: json.loads(v)[field])
    registry[
        [
            "run_id",
            "validation_fold",
            "config_hash",
            "config_path",
            "prediction_path",
            "prediction_sha256",
            "macro_f1",
            "accuracy",
            "support",
            "train_seconds",
        ]
    ].to_csv(OUT / "eda_probe_run_ledger.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["sha256", "scope"]).to_string(index=False))
    print(f"Read {len(registry)} historical probe run records; no new fits.")


if __name__ == "__main__":
    main()
