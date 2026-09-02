# Task 3 clean-slate model research

Status: research and experiment design only. No model result is claimed by this document.

Date: 2026-09-02

## 1. Purpose and clean-slate decision

Task 3 predicts `gender` and `usage` separately from fashion product images. This report replaces
the architectural direction of the E1–E10 sequence. Those experiments remain valuable failure
evidence and comparison anchors. They are not parents for another CNN change.

The clean-slate decision is:

1. Stop incremental work on CNN width, depth, pooling, loss, augmentation, checkpointing, filtering,
   and auxiliary heads.
2. Treat the current image preparation as provisional rather than proven.
3. Test genuinely different representation and decision families.
4. Give `gender` and `usage` different first choices because their failure mechanisms differ.
5. Keep the protected holdout closed until one complete method per target has been frozen.

The recommended first model for `gender` is fixed HOG, colour, and shape features with a calibrated
SVM. The recommended first model for `usage` is an image-to-predicted-product-type
model followed by a probabilistic type-to-usage mapping. A small scratch transformer is the main
learned non-CNN comparison for both targets.

This report uses three evidence labels:

- **Repository evidence** means an observed result or artifact already present in this repository.
- **External evidence** means a claim supported by a linked paper or official technical source.
- **Recommendation/inference** means a proposed decision that has not yet been tested here.

## 2. Non-negotiable assignment and repository constraints

All future work described here must obey the following rules.

1. Use [`data/processed/splits.csv`](../data/processed/splits.csv) as the only split. Do not create a
   new `train_test_split` or new fold file.
2. Keep every `product_family_group` inside one fold. Fit on the four-fold training complement and
   predict only the held-out family-safe fold.
3. Keep `gender` and `usage` as separate outputs. The official prediction columns remain
   `id,gender,articleType,season,usage`.
4. Train every submission-eligible model from random initialisation. Pretrained weights are allowed
   only in an explicitly ineligible comparison benchmark.
5. Fit normalisation, feature transforms, class weights, samplers, codebooks, conditional tables,
   calibration, stacking, and learned preprocessing without outer-validation, holdout, quarantine,
   or official-prediction information.
6. Preserve all classes. Literal `usage="NA"` is a real class. `Home` remains in the output even
   though it has one development example.
7. Register every future training execution through `fashion.train.registry` in
   [`results/runs.csv`](../results/runs.csv).
8. Do not inspect holdout targets, tune after the holdout is opened, or use holdout or official
   prediction images for self-supervised training.
9. Refit the frozen eligible method on all development rows before the one-time holdout evaluation.
10. Keep pretrained super-resolution, pretrained segmentation, pretrained detectors, OCR, external
    text labels, and external style JSON out of an eligible image-only pipeline.

These constraints are grounded in the [assignment specification](<COSC2753_2026B_Assignment 2.pdf>),
the [rubric](../rubrics/RUBRIC.md), [decision 0014](decisions/0014-development-holdout-cv-boundary.md),
[decision 0015](decisions/0015-teacher-only-shared-image-preparation.md),
[decision 0016](decisions/0016-development-label-scope.md), and
[decision 0017](decisions/0017-product-name-na-and-cv-refreeze.md).

## 3. What E1–E10 actually establish

### 3.1 Evidence sources and provenance warning

The main sources are:

- [Task 3 narrative notebook](../notebooks/04_task3_gender_usage.ipynb);
- [main run registry](../results/runs.csv);
- [Task 3 evidence registry](../results/evidence/task3/results/runs.csv);
- [E9 pre-run audit](../results/evidence/task3/e9_prerun/e9_prerun_summary.json);
- [E10 experiment notebook](../notebooks/04j_task3_audience_aux_e10_experiment.ipynb);
- [E10 evidence directory](../results/evidence/task3/experiments/t3_gender_e10_audience_aux/).

**Repository evidence.** The evidence chain is not yet canonical:

- the main registry contains E1–E8 but not E9 or E10;
- the evidence copy contains E9 but not E10;
- E10 is supported by notebook output and artifact files rather than a canonical registry row;
- E1 gender fold 0 has an older T4 result, a newer canonical L4 result, and stale `running` rows;
- early text sometimes uses population fold standard deviation while later text uses sample standard
  deviation;
- all canonical E1–E10 runs use seed 2753 only;
- no three-seed confirmation or independent holdout result exists.

Before new runs, create one read-only reconciliation table that identifies the canonical run ID for
each target, experiment, fold, and seed. Do not delete old rows. Mark duplicates and stale rows as
non-canonical in the reconciliation artifact.

### 3.2 Exact historical score ledger

Pooled OOF macro-F1 below was reproduced by summing the five saved confusion matrices. Fold SD is
sample standard deviation.

| Gender experiment | Pooled macro-F1 | Fold range | Fold SD | Decision value |
|---|---:|---:|---:|---|
| E1 SmallCNN | 0.7118 | 0.6982–0.7279 | 0.0122 | clean baseline reference |
| E2 brightness | 0.6989 | 0.6740–0.7279 | 0.0263 | worse |
| E3 class balance | 0.7081 | 0.6833–0.7238 | 0.0149 | no reliable gain |
| E4 TinyResNet | 0.7121 | 0.7030–0.7235 | 0.0086 | no reliable gain |
| E5 CompactBlur | 0.7069 | 0.6809–0.7268 | 0.0175 | smaller gap, lower score |
| E6 GeM | 0.7335 | 0.7196–0.7495 | 0.0123 | clearest reliable gain |
| E7 TinyHRNet | 0.7025 | 0.6639–0.7222 | 0.0223 | worse and slower |
| E8 best checkpoint | 0.7412 | 0.7240–0.7586 | 0.0130 | small gain |
| E9 semantic filter | **0.7451** | 0.7334–0.7803 | 0.0200 | best observed, failed frozen gates |
| E10 audience helper | 0.7269 | 0.7129–0.7556 | 0.0187 | intended mechanism failed |

E9 is a comparison anchor, not an accepted parent. It missed its frozen score gate, lost too much on
fold 1, retained a mean train–validation gap of about 0.2548, and used 305 name-selected training
exclusions without human label confirmation.

| Usage experiment | Pooled macro-F1 | Without `Home` | Fold range | Fold SD | Decision value |
|---|---:|---:|---:|---:|---|
| E1 SmallCNN | 0.3738 | 0.4205 | 0.3609–0.3960 | 0.0135 | clean baseline reference |
| E2 class balance | 0.4082 | 0.4592 | 0.3818–0.4362 | 0.0203 | accepted historical reference |
| E3 dropout | 0.4161 | 0.4681 | 0.3867–0.4608 | 0.0298 | interval crossed zero |
| E4 TinyResNet | 0.3878 | 0.4363 | 0.3474–0.4426 | 0.0373 | worse |
| E5 label smoothing | 0.4049 | 0.4555 | 0.3806–0.4136 | 0.0131 | confidence gain only |
| E6 focal loss | 0.4094 | 0.4606 | 0.3866–0.4422 | 0.0239 | no reliable separation gain |
| E7 TinyConvNeXt | 0.3451 | 0.3882 | 0.3334–0.3571 | 0.0102 | underfit and expensive |
| E8 translation | **0.4194** | **0.4718** | 0.3868–0.4440 | 0.0271 | best observed, unstable |
| E9 exception balance | 0.3955 | 0.4450 | 0.3485–0.4163 | 0.0271 | worse |

There is no usage E10. E8 is the observed score anchor, while E2 remains the accepted historical
reference because E8's paired interval crossed zero and its minority/robustness evidence was mixed.

### 3.3 Firm findings

**Repository evidence.** The following conclusions are supported:

1. Gender models memorise. Finished training macro-F1 is close to 1.0. The E1 gap is about 0.2871;
   E6 is 0.2666; E8 is 0.2582; E9 is 0.2548; E10 is 0.2731.
2. Smaller capacity alone is not enough. E5 cut the gap to about 0.1654 and improved cost and some
   robustness measures, but pooled macro-F1 fell to 0.7069.
3. GeM was the clearest gender improvement. Its paired family-bootstrap interval against E1 was
   fully positive, approximately +0.0084 to +0.0349.
4. E9 found a real conflict pattern. Child-to-adult errors fell from 229 to 133, and Boys/Girls F1
   improved. It did not prove those 305 filtered labels were wrong.
5. E10's helper task did not improve its intended three-way audience mechanism and reduced five-way
   gender macro-F1.
6. Usage class weighting helped score and probability quality. E2 moved from 0.3738 to 0.4082 and
   improved NLL and ECE.
7. Calibration and class separation are different. Label smoothing and focal loss greatly reduced
   extreme-confidence errors without a reliable macro-F1 gain.
8. TinyConvNeXt usage underfit and cost about twice as much as the small-CNN family.
9. Synthetic corruptions show shortcut risk. E1 gender lost about 0.256 under darkening and 0.155
   under a small translation. E8 usage improved translation robustness but harmed dark-image
   robustness relative to E2.
10. All completed learned candidates are convolution models or small changes around them. The
    planned HOG+HSV reference was described but never completed and registered.

### 3.4 What the evidence does not prove

The history does not prove that:

- all CNNs are unsuitable;
- every augmentation, imbalance method, or structured objective is useless;
- architecture is irrelevant;
- native 60×80 RGB is the best representation;
- image-only performance has reached its ceiling;
- fixed descriptors, transformers, metric learning, probabilistic cascades, or instance-based
  methods will fail;
- the best development model will transfer to the protected holdout or real catalogue inputs.

Repeatedly adapting to the same five folds makes late OOF scores model-selection evidence. A family
bootstrap handles dependence between rows. It does not remove repeated-selection bias. Only the
still-closed holdout can provide the final independent check.

## 4. Data, label, and shortcut evidence

### 4.1 Development population

**Repository evidence.** The canonical development partition contains 32,773 rows in 22,905
families. Gender is present on all 32,773 rows. Usage is present on 32,772 rows; product 28319 has a
missing usage value and remains excluded from usage metrics.

| Gender | Support | Approximate share |
|---|---:|---:|
| Men | 17,753 | 54.2% |
| Women | 12,027 | 36.7% |
| Unisex | 1,766 | 5.4% |
| Boys | 684 | 2.1% |
| Girls | 543 | 1.7% |

| Usage | Support | Learnability warning |
|---|---:|---|
| Casual | 25,151 | dominates accuracy |
| Sports | 3,346 | supported |
| Ethnic | 2,183 | supported, acquisition shift risk |
| Formal | 1,949 | supported |
| NA | 61 | extremely rare |
| Smart Casual | 47 | extremely rare and semantically close to Casual/Formal |
| Travel | 22 | extremely rare |
| Party | 12 | extremely rare |
| Home | 1 | absent from fold-4 training complement |

If `Home` F1 is zero, even perfect F1 on the other eight classes produces an all-nine macro-F1 of
8/9 = 0.8889. `Home` must remain in the official taxonomy, but it cannot be a model-selection gate.
Report both all-nine and without-`Home` usage macro-F1.

Counts and masks are available in [development class summary](../data/processed/development_class_summary.csv)
and the [Task 3 evaluation contract](task3/research/05_evaluation_framework.md).

### 4.2 Image properties and catalogue artifacts

**Repository evidence.** Of the development files, 32,761 are exactly 60×80, 12 have unusual
dimensions, and 294 decode as grayscale. The images are mostly catalogue items on white backgrounds,
but some contain people, mannequins, scenes, or more than one object.

Read-only measurements found strong class-conditional image differences:

- median near-white area is about 65.1%, and median border whiteness is 100%;
- Boys and Girls have about 79% white canvas, compared with about 61% for Men;
- median visible-object height is about half the canvas for Boys/Girls and about 90% for Men;
- median file sizes differ strongly by gender class;
- Ethnic usage images have a median file size near 2.4 KB, compared with about 14.8 KB for Casual.

These may be object cues, acquisition-batch shortcuts, or both. They must be rechecked within the
same `articleType` before being interpreted as label signal. Current global image-quality sampling
is not enough for that decision. Relevant artifacts include the
[data-preparation evidence directory](../results/evidence/data_preparation/) and
[development contact sheet](../results/figures/data_preparation/development_contact_sheet.png).

### 4.3 Family and label ambiguity

**Repository evidence.** Family groups are leakage boundaries, not guaranteed semantic identities.

- 2 gender families have mixed gender labels, covering 14 rows.
- 238 usage families have mixed usage labels, covering 1,232 rows.
- accepted near-identical images still include one Men/Unisex sunglasses conflict and two
  Casual/NA cosmetics conflicts.
- development image ID `44998`, fold 4, Men/Casual/Watches, is completely white.

Some rows therefore have no recoverable pixel evidence, and some nearly identical pixels map to
different labels. This supports a label/signal ceiling. It does not estimate that ceiling and does
not justify automatic relabelling.

ID 44998 must remain in natural validation. A frozen label-free blank-image rule may exclude it from
training complements and trigger an input-quality fallback at inference. Validation rows must not be
deleted because the model cannot read them.

### 4.4 Metadata shortcut diagnostic

**Repository evidence.** A cross-fitted lookup fitted on each outer training complement produced:

| Diagnostic input | Gender accuracy | Gender macro-F1 | Usage accuracy | Usage macro-F1 |
|---|---:|---:|---:|---:|
| ground-truth `articleType` | 0.7825 | 0.4423 | 0.8973 | 0.3974 |
| ground-truth `articleType` + `baseColour` | 0.8017 | 0.4757 | 0.8980 | 0.4187 |

This is a diagnostic, not a legal Task 3 predictor. Ground-truth article type and colour are not
available as Task 3 inference inputs. The result shows that common usage labels are strongly tied to
product type, while long-tail classes remain unsolved. It motivates an image-derived type-posterior
cascade rather than use of hidden metadata.

## 5. EDA that must happen before model selection

The next EDA must be allowed to change the design. It is not another descriptive chart pack.

### 5.1 Audit A — visual observability and label ceiling

Review all 143 rows in `NA`, `Smart Casual`, `Travel`, `Party`, and `Home`, plus a family-stratified
sample from common usage and all gender classes.

Use two reviewers who first see only the image. For each image, record:

- clear, uncertain, or not visually knowable;
- plausible alternative labels;
- whether a person or mannequin is present;
- one item or several items;
- object size, clipping, and visibility;
- only after the blind judgement, agreement with the teacher label.

Report raw agreement, reviewer agreement, Cohen's kappa where meaningful, and the fraction marked
“not visually knowable.” Write plain working definitions for the five rare usage labels. Keep the
canonical labels unchanged.

Use high-confidence OOF errors and Confident Learning only to rank further review cases. Scores must
come from predictions that are out-of-fold for the reviewed row. Never auto-relabel from a model.
[Confident Learning](https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/)
supports this ranking-and-review role.

### 5.2 Audit B — foreground, background, and nuisance signal

Create one deterministic, label-blind foreground proposal based on border-connected near-white
pixels, with a safe full-frame fallback for failed or very large masks. Do not tune its threshold on
OOF scores.

For HOG or another cheap linear probe, compare:

1. full image;
2. foreground-only image;
3. background-only or border-only image;
4. silhouette-only image;
5. grayscale image;
6. colour-summary features without shape;
7. simple nuisance features only: file size, dimensions, white fraction, JPEG indicators, and
   acquisition year where available.

Fit every target-aware probe on an outer training complement. Background-only or nuisance-only
success is a shortcut warning, not a candidate production result. Repeat the analysis within common
`articleType` groups so product mix does not explain every difference.

### 5.3 Audit C — duplicates, families, and sample unit

1. Reconfirm no exact or accepted near-duplicate component crosses folds.
2. Measure family size versus error and confidence.
3. Report product-weighted primary metrics and family-cluster bootstrap intervals.
4. Compare ordinary row-uniform training with exact/accepted-visual-component weighting.
5. Do not equal-weight the broad normalised-name family by default; one name can join many real
   products.
6. Keep the official split and primary validation population unchanged.

### 5.4 Audit D — representation neighbourhoods

For raw pixels, HOG, scattering, and Fisher features:

- remove all same-family neighbours;
- measure top-1 and top-k label purity;
- show purity by class and object-size band;
- inspect nearest neighbours for rare classes and persistent errors.

High non-family purity supports kNN, prototypes, or metric learning. Low purity warns that a single
class prototype will be too simple.

### 5.5 Audit E — fold artifacts

Report per-fold support and Jensen–Shannon shifts for label-free image properties. Keep folds fixed
even if a shift is found. The purpose is to explain fold spread, not repair the split after seeing
model scores.

## 6. Preprocessing and representation comparisons

### 6.1 Model-independent candidates

These may be shared across model families only after a fair test:

- native teacher resolution versus a fixed resize/recompression path;
- full canvas versus a frozen foreground-normalised crop with full-frame fallback;
- RGB versus grayscale and silhouette diagnostic views;
- original decoded pixels versus a fixed resize/recompression path;
- row-uniform versus exact-visual-component-weighted training;
- blank/nearblank label-free training exclusion;
- aspect-preserving letterbox versus any existing stretch or crop path.

Upscaling the teacher image alone adds no information. Unconditional stretch or centre crop should
not be accepted because they can deform or remove small products. A preprocessing effect may be
called model-independent only if it moves both a fixed-feature sentinel and a learned non-CNN
sentinel in the same direction. Otherwise keep it family-specific.

### 6.2 Family-specific choices

These belong inside the named model configuration:

- HOG cell size, block size, luminance/gamma handling, and colour-grid resolution;
- scattering scale/order and any PCA applied afterwards;
- SIFT/DAISY density, PCA, vocabulary, GMM, and spatial pyramid;
- transformer patch size, padding, width, window size, depth, and positional treatment;
- per-image versus fold-fitted dataset normalisation;
- self-supervised transformations and projection heads;
- class prototypes, metric objective, and class/family-balanced batch construction;
- learned foreground or saliency models;
- cascade conditional priors, stacker, or blend.

### 6.3 Safe before fitting versus fold-fitted

| Safe only when fixed without target-score feedback | Must be fitted inside each outer training complement |
|---|---|
| load only `splits.csv` and verify its digest | RGB mean and standard deviation |
| EXIF orientation, decode, and deterministic RGB conversion | feature scaling, PCA, whitening, feature selection |
| fixed aspect-preserving resize/pad | SIFT/Fisher vocabulary, PCA, and GMM |
| frozen per-image foreground rule and fallback | learned segmentation, crop, saliency, or background model |
| fixed label-free blank/nearblank flag | scratch autoencoder or self-supervised encoder |
| fixed HOG/scattering extraction | label-quality scores and any training-row filter |
| fixed class order and corruption transforms | type-to-usage priors or tables |
| split/family/duplicate integrity checks | prototypes, stackers, blend weights, and calibration |

If a supposedly safe threshold is chosen after looking at a scored result, it becomes a model
parameter and must be selected without the outer validation rows.

Self-supervised training is still fitting. It must be repeated on the four-fold training complement
for every outer OOF round. Unlabelled outer-validation pixels are not allowed.

## 7. Broad clean-slate candidate pool

| Family | Core idea | Why it is different from E1–E10 | Main risk | Initial role |
|---|---|---|---|---|
| HOG + Lab/HSV + silhouette SVM/logistic model | fixed gradients, colour and shape | no learned visual hierarchy | may miss texture and context | mandatory cheap classical reference |
| wavelet scattering + SVM | fixed multi-scale wavelets | fixed representation with controlled deformation stability | invariance may remove small cues | later fixed-feature comparison after HOG |
| dense SIFT/DAISY + spatial pyramid/Fisher vector | local invariant descriptors and fold-fitted codebook | classical local-part representation | tiny images and CPU cost | second classical family |
| kNN or multi-prototype classifier | instance-based decisions in a fixed embedding | no parametric softmax head | class shapes may be multi-modal | diagnostic/ensemble candidate |
| micro-Swin from scratch | linear patches and local/global self-attention | token attention rather than a convolution feature stack | data hunger and overfit | main learned non-CNN family |
| small GFNet from scratch | learned global frequency filters | global FFT mixing | less established on this data scale | alternative learned non-CNN |
| supervised contrastive embedding | class/family-balanced metric geometry | representation objective changes from direct softmax | rare batches, augmentation risk, cost | later exploratory family |
| SimSiam self-supervision | learn from paired views without labels or negative batches | fold-local representation pretraining | no extra sample count; long five-fold cost | later exploratory family |
| probabilistic gender hierarchy | final five classes built from meaningful binary probabilities | changes the final decision structure | routing/error propagation | gender-specific candidate |
| predicted-type usage cascade | integrate image-derived type uncertainty with `P(usage|type)` | task-shaped probabilistic model | type errors and exception classes | first usage candidate |
| coarse-to-fine usage cascade | supported classes versus rare branch, then specialist | explicit rare-branch routing | only 143 ultra-rare rows | usage diagnostic |
| privileged-information distillation | metadata teacher during fold training, image-only student at inference | training-only extra information | product names leak labels; policy/ethics concern | high-risk research only |
| diverse late ensemble | average accepted, distinct model probabilities | combines independent error patterns | leakage if weights use outer validation | final step only |

The HOG reference should use the locally normalised gradient design introduced by
[Dalal and Triggs](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). Scattering is
grounded in [Bruna and Mallat](https://arxiv.org/abs/1203.1513) and can be implemented with the
[Kymatio user guide](https://www.kymat.io/userguide.html). Dense local descriptors and layout are
supported by [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf),
[spatial pyramids](https://slazebni.cs.illinois.edu/publications/cvpr06b.pdf), and
[improved Fisher vectors](https://europe.naverlabs.com/wp-content/uploads/2010/09/PSM10_0766.pdf).

Micro-Swin is based on the [Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html).
The original evidence is from much larger datasets, so a small scratch version is a hypothesis, not
an evidence-backed winner here. Plain ViT is lower priority because its strongest results rely on
large-scale pretraining; see the [ViT paper](https://openreview.net/pdf?id=YicbFdNTTy).

Later metric options are supported by [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html),
[SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html),
and [Prototypical Networks](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html).
Privileged-information work is described by [Vapnik and Vashist](https://www.jmlr.org/papers/volume16/vapnik15b/vapnik15b.pdf)
and [generalised distillation](https://arxiv.org/abs/1511.03643), but current image-only policy makes
this a non-priority, approval-gated lane.

## 8. Ranked shortlist for gender

### Rank 1 — HOG + colour/shape + calibrated SVM

**Recommendation.** Extract fixed HOG features, Lab/HSV grid summaries, foreground bounding-box
geometry, and coarse spatial shape. Fit scaling, optional PCA, SVM regularisation, and calibration
inside the outer training complement.

Why first:

- it is genuinely different from every completed learned model;
- fixed filters directly test whether lower variance reduces the large gender gap;
- the teacher-only neighbourhood audit found HOG stronger than the compact scattering summary;
- colour and shape retain signals that grayscale HOG may weaken;
- fixed label-free features can be cached before fold fitting.

Main risks:

- fixed cells may miss small child/adult or accessory cues;
- the foreground rule may fail on people, scenes, or white products;
- an RBF SVM may be too large, making linear SVM the safer first fit.

### Rank 2 — pure-patch micro-Swin trained from scratch

**Recommendation.** Start with a linear 4×4 patchifier, approximately 20×15 tokens for 80×60
input, small shifted windows, 2–3 million parameters, random weights, and no convolution tokenizer.
Use aspect-safe padding and one fixed training recipe.

Why second:

- attention provides a materially different token-mixing family;
- local windows respect the small image and control memory;
- the hierarchy can combine object parts with broader catalogue context.

Main risks are scratch-data hunger, custom rectangular padding, and the same memorisation pattern at
higher cost. Screen two fixed folds before full five-fold training.

### Rank 3 — probabilistic gender hierarchy on fixed features

Fit and combine:

1. `Unisex` versus gendered;
2. child versus adult, conditional on gendered;
3. male versus female, conditional on gendered.

Use calibrated probabilities to recover `Boys`, `Girls`, `Men`, `Women`, and `Unisex`. This changes
the inference structure, unlike E10's training-only auxiliary head. Screen it on cached HOG
features. Reject it if routing harms Unisex or child recall.

### Rank 4 — dense SIFT/Fisher spatial pyramid + colour + linear SVM

This is the second true classical family. It may capture local garment parts and layout better than
HOG. Run it after the cheaper HOG reference. The tested compact scattering summary remains a
configuration-specific negative diagnostic, not a rejection of every scattering design.

### Rank 5 — metric embedding with several prototypes per class

A single Men or Women prototype is too simple. If non-family nearest-neighbour purity is promising,
test several class prototypes or calibrated kNN after supervised contrastive or SimSiam training.
This is exploratory because balanced rare-class batches and two-view training are expensive.

## 9. Ranked shortlist for usage

### Rank 1 — image-derived article-type posterior to usage

**Recommendation.** Inside each outer training complement:

1. train an eligible image-to-`articleType` model from scratch;
2. obtain calibrated `P(articleType | image)`;
3. estimate smoothed `P(usage | articleType)` from the same training complement;
4. predict

   `P(usage=u | image) = sum_t P(usage=u | articleType=t) * P(articleType=t | image)`;
5. report usual type–usage rows and exception rows separately.

The first version should use the probability mixture above, not a learned stack. A learned stack is
allowed only with inner-fold OOF type predictions. Never train a stacker on in-sample type
probabilities or true validation metadata.

Why first:

- the cross-fitted diagnostic shows about 89.7% type-to-usage accuracy;
- it represents type uncertainty rather than committing to one hard type;
- it matches the semantic shape of usage better than another flat visual head;
- it provides an interpretable failure path: image-to-type error versus type-to-usage exception.

Risks:

- common `articleType` rules can erase Party, Smart Casual, Travel, and NA;
- article-type mistakes propagate;
- a type-only mixture may underuse visual context;
- calibration must be fold-safe.

If it passes as a standalone model, a direct visual residual or an unweighted mean with an accepted
direct model may be tested later. Blend weights require nested cross-fitting.

### Rank 2 — scattering or SIFT + calibrated ECOC/one-vs-rest SVM

This is the low-risk direct visual reference. ECOC uses several binary code decisions and can pool
evidence for hard classes. It will not create signal for labels that are not visible. The method is
grounded in [error-correcting output codes](https://arxiv.org/abs/cs/9501101). Use fold-safe
calibration and inspect predicted counts for every rare class.

### Rank 3 — direct micro-Swin or small GFNet

Use the same frozen source and crop choice as the gender screen, but train a separate usage model.
Swin is the preferred first learned family. A small [GFNet](https://proceedings.neurips.cc/paper_files/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html)
is a second non-CNN option if the transformer fails because global frequency mixing may capture broad
garment context. Both remain at risk from nonvisual labels.

### Rank 4 — coarse-to-fine usage cascade

Route first to Casual, Ethnic, Formal, Sports, or rare/other, then run a specialist on the rare
branch. Treat this as a diagnostic, not an assumed finalist. The rare branch has only 143 rows and
routing errors compound.

### Rank 5 — fold-local self-supervised or supervised-contrastive embedding

Prefer SimSiam or a small supervised-contrastive model over DINO multi-crop. There are no extra
allowed training images, so self-supervision changes regularisation rather than sample count. It is
expensive across five outer folds and its transformations may erase real usage cues.

## 10. Approaches not recommended first

- Another small CNN, ResNet, ConvNeXt, pooling change, dropout change, loss change, augmentation
  change, filter, or auxiliary head continues the failed E1–E10 direction.
- A plain ViT is more data-hungry than the proposed local-window micro-Swin.
- DINO multi-crop is likely to exceed the practical RTX 3070 time/memory budget.
- One prototype per class ignores the many visual modes inside Men, Women, and Casual.
- A deep ensemble of the same architecture adds cost and seed averaging, not rubric breadth.
- A learned ensemble or stack fitted on outer OOF rows leaks model-selection information unless an
  inner cross-fitting layer is used.
- Product names, ground-truth article type, base colour, file size, or IDs must not be Task 3
  inference features.
- A pretrained ResNet, CLIP, or DINOv2 can be a final `submission_eligible=false` comparison only.
  It cannot produce the submitted predictions or become a parent.

## 11. Compute and implementation risk

These are planning estimates, not measurements. Record a smoke-test wall time, peak GPU allocation,
host RAM, feature size, and batch-one latency before committing to five folds.

| Family | Expected time | Expected memory | Main implementation risk |
|---|---|---|---|
| HOG + colour + shape | 10–30 min total feature work; seconds/minutes per linear fit | about 1–4 GB host RAM, no GPU | feature dimensionality and foreground rule |
| scattering + linear SVM | 30–90 min fixed extraction; minutes per fold fit | about 1–2 GB VRAM, 2–8 GB host RAM | coefficient size and safe caching |
| dense SIFT spatial pyramid | 2–6 h CPU extraction | about 1–4 GB host RAM | few stable keypoints at 60×80 |
| Fisher vectors | about 0.5–2 h per fold CPU | about 2–6 GB host RAM | fold-fitted PCA/GMM cost |
| cached-feature gender hierarchy | under 30 min per fold | under 2 GB host/GPU memory | calibration and routing |
| type-posterior usage model | minutes beyond article-type probabilities for the table | small table overhead | article-type model is the main cost |
| micro-Swin, 60–100 epochs | L4 0.5–2 h/fold; RTX 3070 1–3 h/fold | about 2–6 GB VRAM | rectangular attention and overfit |
| small GFNet | L4 0.5–1.5 h/fold; RTX 3070 1–2.5 h/fold | about 1–4 GB VRAM | less local inductive bias |
| SimSiam/MoCo, 100–300 epochs | 2–6 h/fold | about 4–8 GB VRAM | two views and fivefold retraining |
| supervised-contrastive/proxy model | 1–3 h/fold | about 3–7 GB VRAM | rare-class batch construction |
| privileged-information distillation | 0.5–1.5 h/fold plus teacher cost | about 2–6 GB VRAM | policy and metadata leakage |

The hard deployment-development gate is 7 GiB peak on an RTX 3070. The planning limit is 90 minutes
per fold on RTX 3070 or 60 minutes on L4, unless a two-fold result justifies an explicit exception.

## 12. Evaluation contract for every candidate

Use the same evaluation unit and metrics for all families:

1. pooled family-safe five-fold OOF macro-F1 as the primary development score;
2. usage macro-F1 both with and without `Home`;
3. per-class precision, recall, F1, support, and predicted count;
4. fold scores, sample SD, minimum fold, and range;
5. family-cluster paired bootstrap delta against the historical anchor;
6. train and validation macro-F1 where a training prediction exists;
7. NLL, multiclass Brier score, ECE15, reliability plots, and high-confidence errors;
8. the existing fixed mild-corruption suite with the same transforms and severity;
9. object-size, background, usual-type, exception, family-size, blank/nearblank, and human-ambiguity
   slices;
10. parameters, feature/checkpoint size, wall time, host RAM, peak GPU allocation, and batch-one
    latency;
11. exact split, label-map, source-manifest, transform, code, configuration, and artifact digests.

Calibration must not use the outer validation labels. Use inner cross-fitting or a calibration subset
drawn only from the outer training complement. See [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)
and the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
The fixed-corruption principle follows [ImageNet-C](https://arxiv.org/abs/1903.12261); this project
must keep its own low-resolution, task-relevant corruption set frozen.

## 13. Staged experiment matrix

### Stage 0 — repair and freeze evidence, no training

1. Reconcile the canonical E1–E10 run IDs without deleting history.
2. Freeze sample SD, pooled confusion-matrix aggregation, class order, usage masks, and all-nine plus
   without-`Home` reporting.
3. Freeze matched historical anchor predictions for paired comparison.
4. Freeze artifact completeness checks and resource measurement.

Exit: one canonical evidence index exists and reproduces the score tables in Section 3.

### Stage 1 — design-changing audits, no model selection

Complete the five teacher-only audits in Section 5. Freeze the crop rule, blank rule, human-review
protocol, diagnostic views, and compute limits.

Exit: every proposed input can be rebuilt without target labels or protected data, and label
knowability limits are documented.

### Stage 2 — fold-0 smoke tests, not rankings

For each implementation family, run one short fixed-budget fold-0 job. Check:

- class order and output probability shape;
- split/family assertions;
- no missing or duplicate validation IDs;
- normalisation/codebook/calibration fit scope;
- convergence or solver completion;
- memory, wall time, and artifacts;
- registry write and failure-state handling.

Do not promote or reject a model from the smoke score.

### Stage 3 — two-fold representation and model screen

Use folds 0 and 4 for every screened candidate. Fold 4 exposes the unlearnable `Home` case. Use one
fixed seed and predeclared small hyperparameter grids fitted only through inner training folds.

First screen teacher-only preprocessing on HOG and scattering:

| Teacher input comparison | Full canvas | Frozen foreground crop |
|---|---:|---:|
| native teacher 80×60 | required | mask-only and aspect-preserving letterbox required |
| fixed teacher resize/recompression | required | optional after full-canvas result |

Then screen models on the winning family-specific input:

| Gender screen | Usage screen |
|---|---|
| completed HOG+colour+shape reference | completed HOG+colour+shape reference |
| scattering+colour+shape SVM | scattering ECOC/one-vs-rest SVM |
| probabilistic gender hierarchy | predicted-type probability mixture |
| micro-Swin | direct micro-Swin |
| SIFT/Fisher only if fixed-feature evidence warrants | SIFT/Fisher only if fixed-feature evidence warrants |

Diagnostics such as background-only, silhouette-only, and nuisance-only do not compete for final
selection.

### Stage 4 — full five-fold screen

Advance at most three model×input pairs per target. Run all five folds with seed 2753. Produce one
pooled OOF table and full diagnostic bundle per pair. Do not create a child chain; each pair is a
separate clean-slate family comparison.

Exit: each target has at most two candidates that pass the frozen full acceptance gate.

### Stage 5 — three-seed confirmation

Freeze the best one or two methods per target. Run seeds 2753, 2754, and 2755 over all five folds.
Do not alter source, model, hyperparameters, epochs, checkpoint rule, calibration plan, or gates
between seeds.

Exit: the method-level seed mean passes an acceptance route, no seed falls more than 0.010 below its
historical score anchor, and training-randomness spread is reported separately from family-bootstrap
uncertainty.

### Stage 6 — final refit and one holdout opening

Refit the frozen eligible method on all development data, write its immutable evidence bundle, and
unlock the protected holdout once. The holdout judges the chosen method. It cannot select a new one.
An ineligible pretrained comparison may be reported separately. No image outside the teacher
dataset enters Task 3 at any stage.

## 14. Frozen advancement and acceptance gates

Historical anchors are comparison references, not architectural parents:

| Target anchor | Macro-F1 | Companion | NLL | Brier | ECE15 | Main warning |
|---|---:|---:|---:|---:|---:|---|
| gender E9 | 0.7451 | — | about 0.440 | about 0.173 | about 0.065 | gap 0.2548; failed old gates |
| usage E8 | 0.4194 | 0.4718 without `Home` | about 0.329 | about 0.172 | about 0.008 | unstable/minority trade-off |

Before Stage 3, derive and save matched-fold anchor values for folds 0 and 4. Do not compare a
two-fold candidate with a five-fold aggregate.

### 14.1 Stage-3 screen gate

A candidate advances when all applicable hard checks pass:

1. matched-row pooled macro-F1 is no more than 0.020 below its matched anchor;
2. no supported class loses more than 0.030 F1;
3. no NaN, missing OOF row, duplicate OOF row, class-order mismatch, split-digest mismatch, family
   crossing, or out-of-scope fit is present;
4. runtime and memory remain within the screen budget;
5. the model produces non-degenerate probabilities and predicted class counts.

To preserve comparison breadth, one materially different family may advance when it is within 0.030
of the two-fold screen leader and passes all integrity/resource gates.

### 14.2 Full finalist acceptance routes

A candidate must pass either the performance route or the Pareto route, plus every hard safety gate.

**Performance route**

- improve pooled macro-F1 by at least 0.010 over the full historical anchor; and
- have a paired family-bootstrap delta whose 95% interval lower bound is above zero.

This means gender needs at least 0.7551. Usage needs at least 0.4294 and must also improve or preserve
the 0.4718 without-`Home` companion within 0.005.

**Pareto route**

- remain within 0.005 macro-F1 of the full historical anchor;
- achieve at least one reliability improvement: reduce the train–validation gap by at least 25%,
  improve mean corruption drop by at least 0.030, or improve both NLL and Brier by at least 10%;
- improve either runtime or memory by at least 20%;
- satisfy every class and non-inferiority gate below.

The Pareto route allows a fixed-feature method to win by being nearly as accurate but materially more
stable, robust, or practical.

### 14.3 Hard gates for every finalist

**Integrity**

- exact canonical split, family, label-map, mask, and source digests;
- complete five-fold OOF coverage with one prediction per eligible product;
- every run registered, reproducible, and marked `scratch=true` and `submission_eligible=true`;
- no outer-validation, holdout, quarantine, or prediction fitting;
- complete configuration, metrics, history/solver, predictions, robustness, cost, and environment
  artifacts.

**Stability**

- gender fold macro-F1 sample SD ≤ 0.030;
- usage fold macro-F1 sample SD ≤ 0.040;
- no gender fold more than 0.050 below pooled gender F1;
- no usage fold more than 0.070 below pooled usage F1.

**Gender class safety**

- no zero-recall class;
- mean F1 across Boys, Girls, and Unisex ≥ 0.630;
- no gender class more than 0.030 below the matched anchor.

**Usage class safety**

- Casual, Ethnic, Formal, and Sports F1 may not fall more than 0.030 below the matched anchor;
- at least three of NA, Party, Smart Casual, and Travel must have nonzero pooled recall;
- mean F1 over those four rare classes ≥ 0.130;
- predicted count for a rare class must not exceed five times its support;
- `Home` is descriptive only, but a large false-`Home` prediction burst rejects the model;
- both all-nine and without-`Home` comparisons must be reported.

**Confidence and robustness**

- raw ECE15 ≤ 0.050;
- NLL and Brier may be no more than 5% worse than the matched anchor;
- mean macro-F1 drop across the fixed mild-corruption suite ≤ 0.050;
- worst corruption drop ≤ 0.100;
- no supported class recall may collapse to zero under a mild corruption.

**Generalisation and cost**

- mean train–validation macro-F1 gap ≤ 0.200 for gender and ≤ 0.250 for usage;
- a larger gap passes only when it is at least 25% smaller than the anchor gap and accompanies at
  least a 0.015 OOF macro-F1 gain;
- peak RTX 3070 allocation ≤ 7 GiB;
- soft training limit ≤ 90 min/fold on RTX 3070 or ≤ 60 min/fold on L4;
- suggested deployment limits: ≤ 25 ms batch-one latency and ≤ 500 MiB checkpoint or complete
  fixed-feature bundle.

If no candidate passes, report that honestly. Do not relax a gate after seeing results. A failed
clean-slate comparison still adds method breadth and evidence about the data ceiling.

## 15. Recommended first, second, and third tests

### First — observability and shortcut gate

Before model training:

1. reconcile E1–E10 evidence;
2. review all 143 ultra-rare usage images and the stratified gender/common sample;
3. freeze the blank-image and foreground fallback rules;
4. run foreground/background/silhouette/colour/nuisance probes.

This decides whether teacher-image preprocessing discards useful pixels, whether catalogue artifacts
drive labels, and whether the rare labels are visually knowable.

### Second — complete the missing classical references

Run the full five-fold HOG+colour+shape reference, then scattering+colour+shape SVM, under the same
input policy and artifact contract. These are cheap, genuinely different, and directly test whether
fixed representations reduce memorisation.

### Third — run the target-specific structured models

- `usage`: predicted article-type posterior to smoothed type-to-usage probability mixture;
- `gender`: probabilistic Unisex/gendered, child/adult, and male/female hierarchy on the best fixed
  representation.

After these three tests, screen micro-Swin on folds 0 and 4. Do not start with micro-Swin, metric
learning, an ensemble, or another CNN because the cheaper experiments answer the largest open
questions first.

## 16. Required artifacts for implementation handoff

Every future candidate should leave:

- one hypothesis/configuration record written before training;
- exact run IDs for five folds and each confirmation seed;
- split, label-map, teacher-image, and code/config digests;
- fold-safe normalisation/codebook/calibration records;
- one OOF file with ID, family, fold, truth, class probabilities, prediction, and slice flags;
- aggregate metrics, confusion matrix, per-class table, fold table, and family-bootstrap comparison;
- human-observability, usual-type/exception, foreground/background, family-size, and blank slices;
- corruption, calibration, confidence-error, latency, memory, and runtime evidence;
- an accept/reject decision that names the frozen gate and evidence path;
- a note that the holdout remains closed or the exact one-time unlock record.

## 17. External comparison context

External fashion results are useful only when their data and eligibility differences are explicit.
[DeepFashion](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html)
is much larger and has richer annotations, so it is not a direct score comparison. Work using
pretrained ResNet/BERT components or undersampling is also not directly comparable to this
assignment's scratch, fixed-split Task 3 system.

A pretrained ImageNet or [DINOv2](https://arxiv.org/abs/2304.07193) linear probe may be run after the
eligible method is frozen. Mark it `submission_eligible=false`, exclude it from official predictions,
and use it only to discuss how much representation pretraining changes the result.

## 18. Final recommendation

The next Task 3 phase should not ask, “Which CNN knob is next?” It should ask:

1. Is the label visible in the image?
2. Which teacher-only preprocessing keeps the most useful evidence?
3. Is the model learning the product or the catalogue background?
4. Can a fixed low-variance representation match the CNNs without memorising?
5. Can usage be modelled through predicted product type without destroying exceptions?
6. Can a small scratch transformer add useful global context after the input questions are settled?

This plan creates real comparison breadth, keeps leakage protections intact, stays practical on an
L4 or RTX 3070, and turns failures into reportable evidence rather than another unstructured tuning
chain.
