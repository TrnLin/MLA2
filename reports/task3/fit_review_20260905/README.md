# Task 3: independent fitting review

Reviewed 5 September 2026. No training or inference was started. Existing notebooks and reports were preserved.

## Main judgement

Overfitting is confirmed across the main learned-model search. Gender models usually learn the training rows almost perfectly, then lose a large amount on unseen product families. Usage has the same problem, often more sharply, plus very weak rare-class learning.

Underfitting is not an equally sound label for every weak run. The type cascade is the clearest example. TinyConvNeXt fits training data poorly compared with SmallCNN, but also clearly overfits later in training. Calling it only “underfit” misses half the evidence. A low training score does not establish that the architecture is too small; the fixed optimiser, budget, objective and representation could also limit learning.

G2 improves Gender, but does not cure overfitting. Its full five-fold evidence now supports that statement directly.

## What was checked

The 23-entry old ledger was rebuilt from saved fold metrics or epoch histories. All 23 train/validation means agree to within 1e-8. The four Dataset V2 screens were added, plus the completed G2 confirmation as a separate scope: 27 configurations, 28 rows. U2 has not run, so it has no fitting diagnosis.

I read the main notebook's recorded decisions, screen notebooks, class results, robustness tables and training/evaluation code. I rebuilt and visually inspected all available mean learning curves and loss curves. G2 folds 1–3 and both aggregate metrics were read from Drive; folds 0/4 were already local. The small readbacks and their source links are saved beside this report. This is not a full checkpoint/registry integrity audit or a new paired-bootstrap acceptance audit.

Diagnostic EDA probes on balanced samples are not additional full-development models. Their scores do not establish overfitting without matching training scores. Unrun proposals and lower bounds are not counted as completed trained candidates.

## Read the numbers correctly

All scores below are macro-F1, not accuracy. Train and validation columns are arithmetic fold means; pooled OOF combines each unseen-fold prediction before scoring. The gap is train mean minus validation mean. A two-fold screen must be compared with its parent's same two folds, not the parent's five-fold total.

An asterisk marks an online training score: Gender E1/E2 and Usage E1/E2 did not save the same finished-model clean-training check. Online scores mix changing model weights, training mode and sometimes augmentation. They support a warning, but are not exactly comparable with later clean evaluation gaps. In G2, online training ends around 0.963, whereas clean finished-model training is 0.9944. Using the former would hide some of the remaining gap.

Loss charts show trends, not a common scale across methods. Weighted CE, focal loss, smoothing and E10's helper loss are different objectives. Even within a run, train and validation loss definitions can differ. Use unweighted saved NLL for cross-model probability comparisons. E8's last logged epoch is not necessarily its selected checkpoint; its table uses selected-model metrics.

## Every completed configuration

### Gender

| Experiment | Folds | Train | Validation | Gap | Pooled OOF | Judgement |
|---|---:|---:|---:|---:|---:|---|
| E1 SmallCNN | 5 | 0.9988* | 0.7117 | 0.2871 | 0.7118 | Overfit; near-perfect online train score. |
| E2 brightness | 5 | 0.9920* | 0.6988 | 0.2931 | 0.6989 | Overfit; brightness did not help. |
| E3 class weights | 5 | 0.9994 | 0.7079 | 0.2916 | 0.7081 | Overfit; weighting did not fix the class boundary. |
| E4 TinyResNet | 5 | 1.0000 | 0.7120 | 0.2879 | 0.7121 | Overfit; almost perfect train, worse confidence. |
| E5 CompactBlur | 5 | 0.8723 | 0.7069 | 0.1654 | 0.7069 | Less overfit, but weaker fit and no score gain. |
| E6 GeM | 5 | 1.0000 | 0.7333 | 0.2666 | 0.7335 | Overfit despite a useful score gain. |
| E7 TinyHRNet | 5 | 1.0000 | 0.7018 | 0.2982 | 0.7025 | Strong overfit; perfect train in every fold. |
| E8 early stopping | 5 | 0.9993 | 0.7411 | 0.2582 | 0.7412 | Overfit; checkpoint choice only trims the gap. |
| E9 label filter | 5 | 1.0000 | 0.7451 | 0.2548 | 0.7451 | Overfit; removing suspect rows does not stop memorisation. |
| E10 helper head | 5 | 1.0000 | 0.7268 | 0.2731 | 0.7269 | Overfit; helper target does not fix generalisation. |
| S1 HOG–SVM | 2 | 0.8535 | 0.6606 | 0.1929 | 0.6608 | Residual overfit plus weaker representation. |
| S2 Micro-Swin | 2 | 0.9999 | 0.6848 | 0.3150 | 0.6847 | Strong overfit; extra epochs mostly raise confidence in errors. |
| G1 foreground mask | 2 | 1.0000 | 0.7297 | 0.2703 | 0.7295 | Overfit; clean training still near perfect. |
| G2 translation screen | 2 | 0.9939 | 0.7455 | 0.2484 | 0.7457 | Reduced gap; still overfit (screen only). |
| G3 component weights | 2 | 1.0000 | 0.7244 | 0.2756 | 0.7245 | Overfit; duplicate weights do not help. |
| G2 completed confirmation | 5 | 0.9944 | 0.7502 | 0.2442 | 0.7502 | Real improvement; still overfit; dark-image guard fails. |

### Usage

| Experiment | Folds | Train | Validation | Gap | Pooled OOF | Judgement |
|---|---:|---:|---:|---:|---:|---|
| E1 SmallCNN | 5 | 0.6546* | 0.3741 | 0.2805 | 0.3738 | Mixed: poor minority fit plus a large train–validation gap. |
| E2 class weights | 5 | 0.9385* | 0.4043 | 0.5342 | 0.4082 | Strong overfit; weighting greatly raises train fit. |
| E3 dropout | 5 | 0.9291 | 0.4140 | 0.5151 | 0.4161 | Strong overfit; modest, uncertain validation gain. |
| E4 TinyResNet | 5 | 0.9762 | 0.3858 | 0.5904 | 0.3878 | Strong overfit; largest Usage gap in this ledger. |
| E5 label smoothing | 5 | 0.9420 | 0.4033 | 0.5387 | 0.4049 | Strong overfit; softer confidence does not fix class errors. |
| E6 focal loss | 5 | 0.9373 | 0.4056 | 0.5317 | 0.4094 | Strong overfit; focal loss barely changes score. |
| E7 TinyConvNeXt | 5 | 0.6838 | 0.3474 | 0.3364 | 0.3451 | Mixed: poor train fit AND clear later overfitting. |
| E8 translation | 5 | 0.7905 | 0.4155 | 0.3750 | 0.4194 | Less overfit; real shift benefit, unresolved class errors. |
| E9 exception weights | 5 | 0.7775 | 0.3947 | 0.3828 | 0.3955 | Overfit plus a harmful change in class trade-offs. |
| S1 type cascade | 2 | 0.3815 | 0.3194 | 0.0621 | 0.3193 | Clearest underfit pattern: both scores low and close. |
| S2 Micro-Swin | 2 | 0.9435 | 0.3962 | 0.5473 | 0.3973 | Strong overfit; validation loss rises while train loss falls. |
| U1 component weights | 2 | 0.9086 | 0.4099 | 0.4987 | 0.4138 | Strong overfit; weighting does not solve generalisation. |

## What the curves change in the diagnosis

**1. This is not just “we trained a few epochs too long.”** Gender E6, TinyResNet and TinyHRNet already reach near-perfect training fit while validation stays much lower. Their end scores are only about 0.006–0.008 below their own best logged fold scores on average. Those hindsight peaks are optimistic, yet even they cannot explain gaps near 0.27–0.30. E8's checkpoint experiment improves Gender by 0.0077 but leaves a 0.2582 gap. Early stopping treats a small part of the problem.

**2. The early Gender runs also have training instability.** E1–E3 validation curves jump sharply in the middle, then settle as the learning rate falls. Their loss minimum often occurs much earlier than their best macro-F1. This is not a clean monotonic “validation got worse after epoch 5” story. The logs alone do not identify whether the jumps come from batch statistics, optimisation or sensitive minority predictions. They do show that a simple loss-minimum cutoff could throw away later useful class learning.

**3. Micro-Swin shows the clearest harmful extra training.** From roughly the middle to the end, train F1 rises while mean validation F1 slightly falls in both targets. Validation loss rises strongly. Its 60 epochs did not turn training fit into useful unseen-family rules.

**4. Smaller gaps can come from giving up useful fit.** CompactBlur lowers Gender's gap to 0.1654, but training falls to 0.8723 and pooled OOF falls to 0.7069. This is a real reduction in memorisation, with a cost in predictive fit; not a success under the frozen rule. The Usage type cascade has only a 0.0621 gap because its train/validation scores are both poor. TinyConvNeXt's later training score rises much faster than validation while validation loss worsens: limited fit and overfitting coexist.

**5. I would not call Usage E8 simply underfit.** Its lower clean training score comes with improved pooled validation and a large gain under shifts. That is useful regularisation. It still has a 0.3750 gap, weak rare classes and a dark-image regression. E9 is different again: exception weighting changes which mistakes the model is rewarded for avoiding. Its lower train score is not proof of insufficient capacity.

## G2: what the completed run proves

Against matched five-fold E6:

- Pooled macro-F1: 0.7335 → 0.7502. All five fold changes are positive.
- Clean train mean: 1.0000 → 0.9944. It still nearly memorises the training images.
- Clean gap: 0.2666 → 0.2442. This is a 0.0224 reduction, about 8.4%, not removal of the problem.
- Fold score SD: 0.0128. Fresh folds score 0.7532 pooled; the result is not only folds 0/4.
- NLL: 0.4325 → 0.2904; ECE: 0.0642 → 0.0303. Probability quality improves alongside class score.
- All five class F1 scores improve, but Girls remains 0.6012 and Unisex 0.6137, far below Men 0.9363 and Women 0.9177.
- Mean shift-induced score change: −0.1465 → −0.0273, an improvement of 0.1192.
- Mean darkening-induced score change: −0.2042 → −0.2362. The 0.0320 regression exceeds the frozen 0.020 no-harm guard.

G2's mean final validation loss is only about 0.0014 above each fold's minimum, compared with 0.0841 for E6. Thus its probability learning is much healthier, even though its clean train–validation F1 gap is still large. This is a useful distinction: persistent generalisation error does not mean every part of training is getting worse.

The completed dark-image failure alone prevents an all-gates pass. I have not recomputed its paired-bootstrap gates or silently changed the acceptance rule.

## Why Usage remains hard

The saved E2 class table has 25,151 Casual rows, but only 61 NA, 47 Smart Casual, 22 Travel, 12 Party and one Home row. E2 reaches F1 0.9318 for Casual, 0.8540 for Ethnic and 0.7759 for Formal, while Party, Smart Casual and Home are zero. An overall accuracy near 0.89 therefore does not mean the whole task is solved.

Home cannot establish generalisation: when its one row is validation, training has no Home example. In other folds training may contain it while validation has no Home support. Fixed-class macro averages therefore contain some unavoidable support mismatch. This explains part of a gap, not the widespread 0.50-plus gaps or failures on other classes. Low rare-class validation F1 alone does not prove underfitting; it can reflect sparse support, memorisation, label ambiguity, or a wrong decision boundary.

The recorded E9 error audit separates common type/usage pairs from exceptions. Strong weighting reduces exception errors but more than doubles usual-row error. That is a harmful trade-off, not simply “more overfitting.” E5 smoothing and E6 focal loss reduce extreme confidence errors without a useful macro-F1 gain. Confidence quality and class separation are separate problems.

## What is measured, and what remains a hypothesis

Measured: large clean gaps; near-perfect training in many models; small/zero rare-class recall; high shift/dark sensitivity; useful G2 translation effects; failed background/component-weight screens; poor fit in the type cascade; several weak models with small ECE.

Likely but not established: some catalogue labels are not visible in a tiny product-only image; some label conflicts cap image-only performance; object layout is used as a shortcut. The incomplete blind human review means there is no measured human ceiling. Nuisance correlations do not prove a model uses a feature, and a failed mask does not prove background cues never matter.

The EDA contract mismatch remains: audit hash starts 92b53b, while completion/view contracts start da1880. Treat the EDA package's “current and complete” claim as unverified until reconciled. It does not erase the directly observed train/validation gaps. The saved boundary table reports no family/accepted-duplicate fold crossings; this review did not redo image hashing or the full boundary audit.

## Recommended interpretation

Keep “overfitting persists” as the main Task 3 finding. Describe underfitting precisely, by run, rather than applying it to every low score. G2 is a successful position-tolerance intervention with remaining memorisation and a failed dark-image guard. Usage combines memorisation, support scarcity and weak class separation. No single gap, accuracy or calibration number captures all three.

The next useful work is evidence repair and G2's complete frozen evaluation, followed by U2's code repair before any user-run training. Do not start another architecture sweep from these results. Keep final holdout evaluation sealed and avoid declaring a winner from repeated development-fold comparisons.

## Supporting files

- [Exact ledger](all_experiments.csv)
- [Gender F1 curves](gender_curves.png) and [loss curves](gender_loss.png)
- [Usage F1 curves](usage_curves.png) and [loss curves](usage_loss.png)
- [Fold metrics and mean robustness](detail.json)
- [G2 Drive readback manifest](g2_readback_manifest.json)

The older pooled scores in the ledger retain their published precision. Most have local aggregates; Usage E1's pooled score remains notebook-ledger evidence because its aggregate is not local. No new uncertainty interval, seed replication, final-holdout score or registry certification is claimed here.
