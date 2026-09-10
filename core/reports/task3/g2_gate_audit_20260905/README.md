# Task 3: G2 decision and U2 repair

G2 is rejected under the unchanged confirmation rules. E6 stays the Gender performance benchmark.

Fresh folds 1–3 improve by 0.012947 macro-F1. Their paired family 95% interval is −0.003554 to +0.028650. The required positive lower bound fails.

All five folds score 0.750217 for G2 and 0.733521 for E6. Their difference is +0.016697, with interval +0.004105 to +0.029213. This does not replace the separate fresh-fold check.

Dark-image induced loss worsens by 0.031993 versus E6. The allowed worsening is 0.020. This is the second failed rule. Every other frozen numeric check passes.

The review checks all ten G2/E6 bundles: exact registered runs, split and labels, probabilities, configuration, parent lineage, checkpoint SHA-256 and archive CRC, complete histories, saved clean-gap arithmetic, and all five corruption summaries. No checkpoints were unpickled or run. Corruption scores and clean-training scores are verified saved evidence, not newly measured inference. Bootstrap: 10,000 paired whole-family draws within canonical folds, seed 2753, fixed class set. These results cover one seed and development folds used in repeated selection; they are not independent final-test evidence.

## Evidence

- [Full decision, run IDs, and hashes](g2_decision.json)
- [Every frozen check](g2_gates.csv)
- [Fold comparison](g2_fold_comparison.csv)
- [Downloaded file manifest](download_manifest.json)
- [U2 zero-fit setup check](u2_preflight.json)

The G2 evidence was copied with rclone from `gdrive:MLA2/task3/experiments/t3_gender_v2_g2_translation/gender`. The registry snapshot came from `gdrive:MLA2/task3/results/runs.csv`. The local working registry was not replaced.

From the repository root, reproduce only this saved-result review:

```bash
./.venv/bin/python reports/task3/g2_gate_audit_20260905/reproduce.py
```

## U2 repair

Each inner training subset now computes its own effective-number weights. Only its scaler and SVM use them. Sigmoid calibration uses unweighted held-out data. All eight supported Usage classes must appear in every inner fitting/calibration subset. Home is excluded from fitting/calibration, receives zero probability, and stays in the official nine-class score, including its NLL penalty.

The screen now evaluates the frozen Route A or Route B, with class, calibration, rare prediction count, corruption, runtime, memory, and artifact guards. Missing evidence cannot pass. Old calibration contracts cannot be reused. All five corruptions are measured and saved per image before a fold completes. The 90-minute fold budget includes diagnostics; peak host memory is capped at 7 GiB.

E2 has no comparable saved clean finished-model training score. That Route B branch stays unavailable; the NLL/Brier branch remains usable. The eight-class and common/rare scores are diagnostics only.

The real-data setup check is ready: all parent evidence verified, 1,944 HOG columns, estimated cache 243 MiB, zero fits. No U2 training was started. To run the first screen later, select the repository `.venv` kernel in `notebooks/task3_training/usage_v2_u2_full_rgb_hog_svm.ipynb` and Run All.

## Initial repair validation

`./.venv/bin/pytest -q -rs`: 149 passed, 3 skipped because PyTorch is absent from this local environment. Ruff and `git diff --check` pass. Both edited notebooks' code cells compile. The U2 setup cells ran against real data without fitting. Tests use small synthetic fits to check calibration weight routing and inner class support; they also check family resampling, canonical predictions, missing evidence, corruption failures, resource limits, and rejection when neither route passes.

The report text, changed decision row, and U2 rules were rendered in a browser and visually inspected. All 14 pre-existing notebook output digests are unchanged; the other 13 pre-existing dirty notebooks are byte-for-byte unchanged from the start of this work. Human review forms were left intact. No experiment training, commits, or pushes were made.

## Enforced resource-stop repair

Each U2 fold now runs in a fresh spawned worker. The supervising process terminates and reaps the worker at the 90-minute deadline, including startup, fitting, diagnostics, and worker artifact writes. Linux `RLIMIT_AS` imposes a hard 7 GiB worker address-space ceiling during native solver code. It counts mappings as well as resident memory, so this is intentionally stricter than the original 7 GiB peak-RAM guard. The saved metrics and setup check state this scope. Preparation and aggregate checks are outside the per-fold budget.

Only the supervisor marks a run complete, after successful worker exit. Timeouts, allocation failures, and unexpected exits leave failed registry rows. Each fold has a fresh peak-memory counter. Solver iteration counts, convergence flags, and warnings are retained in `solver_history.csv`. The resource configuration changes the reuse hash and older unsupervised runs are rejected. Decision checks require total supervised fold time rather than falling back to fit time.

Validation: 156 tests passed; three skipped because PyTorch is absent. Tests kill an unresponsive worker, block an oversized allocation, detect an abrupt exit, check the failed registry row, and complete/reuse a tiny synthetic-image fold with all five corruption artifacts. Warning capture and iteration counts also pass. The notebook setup check passes against the real parent evidence with zero fits. Ruff and `git diff --check` pass. No real-data training, commits, or pushes were performed.
