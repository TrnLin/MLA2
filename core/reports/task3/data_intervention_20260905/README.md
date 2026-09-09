# Task 3 data intervention research

**The data plan should be small and targeted. More copies will not fix the rare Usage classes.**

Gender G2 is complete. It improves all five folds, but still overfits and fails its dark-image rule. A small mild-darkening test on top of G2 is the best next data experiment, after the evidence checks are complete.

Usage has a different problem. Party has 12 images from 10 families. Home has one image from one family. New independent labelled products could help, but public images are outside the current Task 3 scope. Edited copies do not create new families. Finish the blind human review before promising that rare occasions can be learned from these tiny pictures.

The analysis checked all 32,773 development image hashes, rebuilt independent support, checked saved model scores, traced Usage errors and inspected image/augmentation figures. Existing work is unchanged. No training, new predictions, collection, relabelling, split changes, commits or pushes were done.

- [Full findings and exact proposed experiments](REPORT.md)
- [G2 frozen-gate readback](g2_frozen_gate_readback.csv)
- [Image and family counts](class_independent_support.csv)
- [Fold and inner-calibration support](fold_class_support.csv)
- [Usage error trade-offs](usage_error_tradeoffs.png)
- [Augmentation previews](augmentation_preview.png)
- [Provenance checks](provenance_verdict.json)

U2 is still unrun. Its gate and calibration code need repair before a user-run screen. The full report gives the exact repair and stop rules. No old acceptance gate was changed.
