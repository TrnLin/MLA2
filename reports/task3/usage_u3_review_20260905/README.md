# Usage U3 review — 5 September 2026

**U3 fails. Do not run more U3 folds.**

I downloaded the saved files with rclone and checked all 13,110 validation rows. U3 scores **0.4034 F1**, below E2's **0.4073**. Its probability errors are worse: NLL rises **37.8%** and Brier rises **7.1%**.

Stage B learned the small training classes better. That gain barely carried over to new products. It made **23 Party guesses with no correct answers**. Travel and the brightening test fail their agreed limits.

There is also a record issue. Fold 4 has finished files, but the current Drive `runs.csv` still says **running**. Fold 0's completion record checks out. Both runs' artifact hashes, saved predictions and frozen feature weights check out.

**Next:** recover fold 4's completion row from the original Colab copy before another run. Keep E2 as the reference. The next research direction is to check independent data with existing Usage or occasion labels: label fit, usable rights and overlap with the teacher data first. More data is a hypothesis to test, not a promised fix.

The E2/E3/E8 average is off the next-test plan. Its small measured gain remains in the results, but it does not show that the models' overfitting or weak rare-class predictions are solved.

[Detailed checks and exact run IDs](REVIEW.md) · [Recomputed decision](recomputed_decision.json) · [Verification](verification.json) · [Test notes](checks.json)

No training, recipe changes, commits or pushes were made. Existing work was preserved.
