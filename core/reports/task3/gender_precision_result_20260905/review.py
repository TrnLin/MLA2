"""Independently recompute the saved precision comparisons from probability arrays."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data.hashing import compute_sha256

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / "reports/task3/gender_diagnostic_result_20260905"
status = json.loads((OUT / "precision_status.json").read_text())
old = json.loads((PRIOR / "diagnostic_status.json").read_text())
for field, path in {
    "module_sha256": ROOT / "src/fashion/train/task3_gender_precision.py",
    "split_sha256": ROOT / "data/processed/splits.csv",
    "source_diagnostic_sha256": PRIOR / "diagnostic_status.json",
    "source_stability_sha256": PRIOR / "batch_order_stability.csv",
}.items():
    assert status[field] == compute_sha256(path), field
assert status["runtime"] == old["runtime"]
assert status["status"] == "complete_for_review" and not status["training_performed"]
assert status["settings_restored"]
assert set(status["ieee_settings"].values()) == {"ieee"}
assert len(status["runs"]) == 4
assert {r["run_id"] for r in status["runs"]} == {r["run_id"] for r in old["runs"]}
saved = pd.read_csv(OUT / "precision_comparisons.csv")
earlier = pd.read_csv(PRIOR / "batch_order_stability.csv")
assert len(saved) == 144 and not saved.duplicated(["run_id", "partition", "mode", "batch_size", "order"]).any()
verified, cross = [], []
for run in status["runs"]:
    source = next(r for r in old["runs"] if r["run_id"] == run["run_id"])
    assert run["source_sha256"] == source["source_sha256"]
    assert run["weights_and_buffers_unchanged"]
    assert 0 < run["peak_allocated_gpu_bytes"] < 3_000_000_000
    assert {s["partition"] for s in run["samples"]} == {"train", "validation"}
    for sample in run["samples"]:
        partition = sample["partition"]
        ids_path = PRIOR / run["run_id"] / f"stability_{partition}_ids.csv"
        assert compute_sha256(ids_path) == sample["ids_sha256"]
        ids = pd.read_csv(ids_path).id.to_numpy()
        with np.load(OUT / run["run_id"] / f"{partition}_probabilities.npz", allow_pickle=False) as arrays:
            assert np.array_equal(ids, arrays["ids"]) and len(ids) == sample["rows"] == 160
            for mode in ("runtime_default", "ieee"):
                reference = arrays[f"{mode}_reference"]
                for batch in (1, 32, 128):
                    for order in ("forward", "reverse", "shuffle"):
                        actual = arrays[f"{mode}_b{batch}_{order}"]
                        for value in (reference, actual):
                            assert value.shape == (160, 5)
                            assert np.isfinite(value).all() and (value >= 0).all()
                            assert np.allclose(value.sum(1), 1, atol=1e-6)
                        delta = float(np.max(np.abs(reference - actual)))
                        flips = int(np.count_nonzero(reference.argmax(1) != actual.argmax(1)))
                        row = saved[(saved.run_id == run["run_id"]) & (saved.partition == partition) &
                                    (saved["mode"] == mode) & (saved.batch_size == batch) & (saved.order == order)].iloc[0]
                        assert abs(delta - row.max_abs_difference) < 1e-15
                        assert flips == row.prediction_flips
                        assert (delta <= 1e-5 and flips == 0) == row["pass"]
                        if mode == "runtime_default":
                            prior_row = earlier[(earlier.run_id == run["run_id"]) & (earlier.partition == partition) &
                                                (earlier.batch_size == batch) & (earlier.order == order)].iloc[0]
                            assert abs(delta - prior_row.max_abs_difference) <= 1e-7
                            assert flips == prior_row.prediction_flips
                            assert row.earlier_summary_reproduced
                        verified.append(dict(model=run["model"], fold=run["fold"], partition=partition,
                                             mode=mode, batch_size=batch, order=order, delta=delta, flips=flips,
                                             passed=delta <= 1e-5 and flips == 0))
            a, b = arrays["runtime_default_reference"], arrays["ieee_reference"]
            delta = float(np.max(np.abs(a - b)))
            flips = int(np.count_nonzero(a.argmax(1) != b.argmax(1)))
            assert delta == sample["cross_mode_reference"]["max_abs_difference"]
            assert flips == sample["cross_mode_reference"]["prediction_flips"]
            cross.append(dict(model=run["model"], fold=run["fold"], partition=partition, delta=delta, flips=flips))
verified = pd.DataFrame(verified)
verified.to_csv(OUT / "verified_comparisons.csv", index=False)
pd.DataFrame(cross).to_csv(OUT / "verified_cross_mode.csv", index=False)
summary = verified.groupby("mode").agg(comparisons=("passed", "size"), passed=("passed", "sum"),
                                       maximum_probability_difference=("delta", "max"), label_flips=("flips", "sum"))
summary.to_csv(OUT / "verified_summary.csv")
assert status["ieee_all_comparisons_pass"] == bool(verified[verified["mode"].eq("ieee")].passed.all())
assert status["earlier_summaries_reproduced"]
result = {"verified":True, "comparisons":len(verified), "summary":summary.to_dict(orient="index"),
          "cross_mode_maximum":max(r["delta"] for r in cross), "cross_mode_label_flips":sum(r["flips"] for r in cross),
          "limitations":"Sampled inference only. Input preparation and unchanged state are attested by the GPU run; no local GPU rerun."}
(OUT / "verified_review.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plot = verified.groupby(["model", "fold", "partition", "mode"]).delta.max().unstack()
labels = [f"{'G2' if m == 'G2' else 'Compact'} f{f}\n{p}" for m,f,p in plot.index]
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(plot))
ax.scatter(x, plot.runtime_default, color="#dc8330", s=65, label="Default precision", zorder=3)
ax.scatter(x, plot.ieee, color="#267baa", s=65, label="Full FP32", zorder=3)
ax.axhline(1e-5, color="#b73341", linestyle="--", label="Frozen limit: 0.00001")
ax.set(yscale="log", ylabel="Largest probability difference (log scale)",
       title="Full FP32 removes the sampled stability warning", xticks=x, xticklabels=labels,
       ylim=(1e-8, 1e-2))
ax.legend(loc="lower left", ncol=3, fontsize=9)
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(OUT / "precision_review.png", dpi=160)
