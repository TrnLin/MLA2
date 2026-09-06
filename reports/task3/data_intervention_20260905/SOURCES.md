# Primary-source evidence ledger

Read 5 September 2026. The report uses local experiment evidence to choose interventions. Research papers supply mechanisms, not expected scores for this dataset. No literature result is presented as a Task 3 result.

| Source | Supported point | Strength for this decision |
|---|---|---|
| [Zhang, ICML 2019: Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html) | Small translations can change CNN outputs; downsampling can contribute | Strong general mechanism; strong local corroboration from G2. Does not identify which layer caused Task 3 errors |
| [Cui et al., CVPR 2019: Class-Balanced Loss](https://arxiv.org/abs/1901.05555) | Reweighting by effective number accounts for diminishing returns of examples | Mechanistic support. Local E2/E9/U1 comparisons limit claims of benefit; family counts are not the paper's fitted effective number |
| [Zhang et al.: Mixup](https://arxiv.org/abs/1710.09412) | Interpolation of inputs/labels can regularise memorisation | Positive benchmark evidence; low direct evidence for tiny fashion images. Proposed weak variant is untested |
| [Yun et al., ICCV 2019: CutMix](https://arxiv.org/abs/1905.04899) | Patch mixing uses area-based label mixing and has benchmark gains | Direct method definition; applicability is uncertain when most canvas area is background |
| [Zhong et al.: Random Erasing](https://arxiv.org/abs/1708.04896) | Occlusion-based training augmentation | General proposal; local preview shows cue-loss risk, not performance of a trained erasing model |
| [Hendrycks et al.: AugMix](https://arxiv.org/abs/1912.02781) | Augmentation mixtures can improve corruption robustness and uncertainty | Benchmark support only; no local AugMix run, so neither accepted nor ruled out |
| [Northcutt et al., JAIR 2021: Confident Learning](https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/) | Label-error ranking under assumptions about noise | Useful review method; subjective usage labels may violate its assumptions. Does not authorise relabelling |
| [Kang et al., ICLR 2020: Decoupling Representation and Classifier](https://arxiv.org/abs/1910.09217) | Representation and classifier rebalancing can be treated separately | Alternative long-tail mechanism, not evidence of a local gain or priority over the planned small audit |
| [scikit-learn 1.9 calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Calibration data must be separated from base fitting; probability scores and class decisions measure different things | High implementation relevance. Actual forwarding of sample weights was checked against installed source, saved locally |
| [Assignment spec](<../../../docs/COSC2753_2026B_Assignment 2.pdf>) and [Decision 0015](../../../docs/decisions/0015-teacher-only-shared-image-preparation.md) | Extra collection is possible in the assignment generally; current Tasks 1–3 policy is teacher-only; submitted models are scratch-trained | Binding task constraints, with the broader spec distinguished from the narrower project decision |

Local evidence records: `source_snapshot.json`, `provenance_verdict.json`, `g2_frozen_gate_readback.csv`, `fit_ledger_recheck.csv`, `oof_coverage_checks.json`, and the report's linked CSVs. Main limitations: one-seed comparisons, many repeated development decisions, sparse tails, incomplete blind human review, and incomplete G2 full-artifact/paired-bootstrap certification.
