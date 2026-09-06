# Stronger MixUp screen

Use `notebooks/04ah_task3_gender_stronger_mixup_screen.ipynb` on a fresh Colab L4.
Push its code first, then Run All. It needs the completed 04af alpha 0.2 runs
and their original parent and precision evidence on Drive.

The new trial changes **only MixUp alpha, from 0.2 to 0.4**. This puts more weight
on the second image on average. Keep every other training control fixed,
including corrected labels, dropout 0.30, grayscale probability 0.10,
batch 128, seed 2753, 30 epochs and the final checkpoint. Train from scratch
on folds 0 and 4 only. It does not continue training the saved models.

The case for trying it comes from the last result: mean clean gap fell from
0.173793 to 0.129133 while pooled validation F1 rose to 0.808909. Unisex,
Boys and Girls explain about 87% of the remaining summed class gaps.
The [MixUp paper](https://arxiv.org/abs/1710.09412) motivates blending as a
way to reduce memorization. Alpha 0.4 is a new hypothesis, not a proven gain.
Stronger blending can also hide useful details in these small images.

Keep the 14 non-F1 G2/E6 gates. Replace relative F1 limits with the user's
74% pooled validation macro-F1 floor. Add five checks:

- Mean clean gap falls by at least 0.020, to about 0.109133 or less.
- Both folds' clean gaps shrink.
- Pooled validation macro-F1 is at least 0.74 (74%).
- Unisex recall does not fall, staying at about 0.510608 or more.

All 19 checks must pass. Fold, class and pooled F1 changes and the paired
family-bootstrap interval remain diagnostics, not pass/fail requirements.
The five replaced G2 F1 checks are saved separately as `diagnostic_checks`.
Targets use exact matched scores, not the rounded
values shown above. This does not change 04af's earlier pass. The gap uses
real, clean training and validation images in evaluation mode. Online training
F1 remains blank because MixUp labels are blended. A drop from the parent's
80.89% validation F1 is allowed down to 74%; Unisex recall must still not fall.

The completed parent's implementation hashes are checked against commit
`68fef49ab1d55d671531113a71a3e71400a0e3fc`, so this new implementation can evolve
without pretending the parent used it. New audits bind this trial's own code,
alpha 0.4 policy, canonical training rows, source runs and parent decision.
Training receipts retain per-epoch row coverage and mixing-plan fingerprints.
The two strengths use separate artifact directories and cannot reuse each
other's receipt. All fits are registered before their first optimizer step.

Output: `MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha040/gender`.
Read `screen_decision.json` and `incremental_comparison.json` first. The latter's
legacy `dropout_*` fields refer to the completed alpha 0.2 models. The report
also keeps matched G2/E6/Gray10 scores and original teacher-label diagnostics.

A zero gap cannot be promised and is not proof that overfitting is gone.
An overly weak model can also have a small gap. Stop after this one trial,
keep the current candidate unless the new rules pass, and keep the held-out
test sealed. Do not automatically run stronger settings or confirmation folds.
