# Task 3 references

[Research index](README.md)

External sources were checked on 26 August 2026. Primary papers, official standards pages, and
official technical documentation are preferred. A source listed here motivates a method or control;
it does not override the assignment or accepted repository decisions.

## 1. Project sources

| Source | What it supports |
|---|---|
| [Assignment specification](<../../COSC2753_2026B_Assignment 2.pdf>) | Task deliverables, scratch-training rule, prediction format, comparison and independent-evaluation expectations |
| [Rubric](../../../rubrics/RUBRIC.md) | Marks reward approach breadth, justified judgement, and report quality rather than a raw accuracy criterion |
| [Project README](../../../README.md) | Project scope and hard constraints |
| [Problem-definition notebook](../../../notebooks/00_problem_definition.ipynb) | Task framing and project roles |
| [Data-preparation notebook](../../../notebooks/01_data_preparation.ipynb) | Official EDA, data quality, class, family, transform, and shortcut evidence |
| [Task 3 notebook](../../../notebooks/04_task3_gender_usage.ipynb) | Narrative home for Task 3 experiments and decisions |
| [Final-evaluation notebook](../../../notebooks/06_final_evaluation.ipynb) | Locked method-freeze, development refit, holdout unlock, and prediction workflow |
| [`splits.csv`](../../../data/processed/splits.csv) | Canonical partitions, folds, families, paths, labels, and masks |
| [`cv_fold_summary.json`](../../../data/processed/cv_fold_summary.json) | Five-fold sizes, class coverage, and fold-4 `Home` limitation |
| [`development_class_summary.csv`](../../../data/processed/development_class_summary.csv) | Product/family/fold support for every development class |
| [`label_maps.json`](../../../data/processed/label_maps.json) | Stable official class order |
| [Data-preparation evidence](../../../results/evidence/data_preparation/) | Digests, class imbalance, NMI, family boundary, image-quality, and provenance evidence |
| [Decision 0014](../../decisions/0014-development-holdout-cv-boundary.md) | Development/holdout boundary and saved family-safe folds |
| [Decision 0015](../../decisions/0015-teacher-only-shared-image-preparation.md) | Teacher-only Task 1–3 image scope and task-owned transforms |
| [Decision 0016](../../decisions/0016-development-label-scope.md) | All development labels kept; literal `NA` valid; no rare-class deletion/masking |
| [Decision 0017](../../decisions/0017-product-name-na-and-cv-refreeze.md) | Final development target and fold refreeze evidence |
| [`dataset.py`](../../../src/fashion/data/dataset.py) | Runtime split, target, and protected-loader contracts |
| [`images.py`](../../../src/fashion/data/images.py) | Deterministic RGB/letterbox transforms and streaming content-pixel statistics |

## 2. AI lifecycle and risk standards

### [ISO/IEC 5338:2023 — AI system life cycle processes](https://www.iso.org/standard/81118.html)

Supports the use of controlled life-cycle processes covering definition, execution, management,
control, and improvement of AI systems.

### [ISO/IEC 23894:2023 — Guidance on AI risk management](https://www.iso.org/standard/77304.html)

Supports adapting AI risk-management processes to an organisation and use context.

### [NIST AI Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)

Supports trustworthy and responsible AI risk management across design, development, use, and
evaluation.

### [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)

Supports the Govern, Map, Measure, and Manage mapping used in the lifecycle plan. The page currently
notes that AI RMF 1.0 is being updated.

### [NIST AI RMF Playbook](https://www.nist.gov/itl/ai-risk-management-framework/nist-ai-rmf-playbook)

Provides practical actions and documentation ideas for applying AI RMF outcomes.

### [NIST SP 1270 — Identifying and managing bias in AI](https://www.nist.gov/publications/towards-standard-identifying-and-managing-bias-artificial-intelligence)

Supports treating bias as a socio-technical risk that can enter throughout the AI life cycle.

## 3. Core image models and features

### [Dalal and Triggs — Histograms of Oriented Gradients](https://doi.org/10.1109/CVPR.2005.177)

Supports HOG as an established hand-built gradient/shape descriptor for the classical image
baseline.

### [He et al. — Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

Supports ResNet residual connections as an optimisation-friendly CNN family. The low-resolution
stem is a Task 3 adaptation, not a claim made by the paper.

### [Howard et al. — Searching for MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)

Supports MobileNetV3 as a hardware-aware compact architecture family. Task 3 still measures real
latency on its own hardware.

### [Tan and Le — EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)

Supports EfficientNet’s compound depth/width/resolution scaling and its use as a conditional compact
capacity comparison.

### [TorchVision models and pretrained weights](https://docs.pytorch.org/vision/master/models.html)

Official documentation for random initialisation with `weights=None`, pretrained weight enums, and
weight-specific preprocessing.

## 4. Fashion classification context

### [Parekh et al. — Fine-Grained Visual Attribute Extraction From Fashion Wear](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/html/Parekh_Fine-Grained_Visual_Attribute_Extraction_From_Fashion_Wear_CVPRW_2021_paper.html)

Supports the relevance of learning related fashion attributes and motivates testing, rather than
assuming, a shared representation.

### [Seo, Lee, and Jang — Fashion e-commerce classification using ResNet-BERT and transfer learning](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324621)

Uses the broad Fashion Product Images dataset and discusses imbalance, noisy product photos,
multimodal inputs, undersampling, and transfer learning. Its filtered, high-resolution, pretrained,
multimodal setting is not directly comparable to this project’s full-label, low-resolution,
image-only, scratch setting.

### [Fashionpedia](https://research.google/pubs/fashionpedia-ontology-segmentation-and-an-attribute-localization-dataset/)

Provides wider context on fine-grained fashion ontology and attribute/localisation tasks. It is not
Task 3 training data.

## 5. Class imbalance and loss design

### [Cui et al. — Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)

Supports the effective-number class-weight experiment. Task 3 adds fold-train fitting and a cap to
avoid extreme one-example weights.

### [Lin et al. — Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

Supports focal loss as a way to reduce easy-example contribution. Its original detection context is
why this plan treats it as conditional rather than default.

## 6. Multitask learning and negative transfer

### [Standley et al. — Which Tasks Should Be Learned Together?](https://proceedings.mlr.press/v119/standley20a.html)

Supports both sides of the shared-backbone decision: multitask learning can reduce inference cost,
but competing objectives can make performance worse.

### [Kendall, Gal, and Cipolla — Multi-Task Learning Using Uncertainty to Weigh Losses](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)

Supports uncertainty-based task weighting as one conditional response to unequal task losses.

### [Yu et al. — Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

Supports PCGrad as a conditional method for projecting conflicting task gradients.

## 7. Augmentation and regularisation

### [Zhang et al. — mixup](https://openreview.net/pdf?id=r1Ddp1-Rb)

Supports convex image/label mixing as a regularisation method. Task 3 requires extra mask handling
because one usage label is missing.

### [Yun et al. — CutMix](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf)

Supports region replacement and area-weighted label mixing. Tiny images and multi-product scenes make
it an optional rather than default Task 3 method.

### [Cubuk et al. — RandAugment](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)

Provides context for reduced-search automated augmentation. This plan prefers a simpler light policy
unless evidence justifies added search.

### [Zhong et al. — Random Erasing](https://ojs.aaai.org/index.php/AAAI/article/view/7000)

Supports random erasing as an occlusion regulariser. Task 3 uses it cautiously because the product is
small.

## 8. Evaluation and statistical uncertainty

### [Scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

Official metric definitions for precision, recall, F1, balanced accuracy, MCC, confusion matrices,
and probability metrics.

### [Bengio and Grandvalet — No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/v5/grandvalet04a.html)

Supports the warning that overlapping CV training sets make naive fold-based variance estimates
unreliable.

### [Bouthillier et al. — Accounting for Variance in Machine Learning Benchmarks](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)

Supports measuring several sources of experimental variance rather than relying on one run.

### [Ojala and Garriga — Permutation Tests for Studying Classifier Performance](https://www.jmlr.org/papers/v11/ojala10a.html)

Supports a paired permutation test as an optional sensitivity analysis. The plan’s main comparison
uses paired family-bootstrap effect intervals.

## 9. Calibration and selective review

### [Guo et al. — On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

Supports measuring neural-network miscalibration and testing temperature scaling.

### [Scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)

Official practical guidance for reliability diagrams and probability calibration.

## 10. Robustness and visual diagnosis

### [Hendrycks and Dietterich — Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://openreview.net/pdf?id=HJz6tiCqYm)

Supports a fixed corruption protocol and the rule against training on the exact held-out robustness
suite after seeing its results.

### [Selvaraju et al. — Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)

Supports coarse gradient-based localisation as a model and dataset-bias diagnostic. It does not make
the heatmap a causal proof.

## 11. Ethics, documentation, and reproducibility

### [Buolamwini and Gebru — Gender Shades](https://proceedings.mlr.press/v81/buolamwini18a.html)

Shows large intersectional disparities in commercial facial gender classification. Task 3 is not a
face classifier; the source supports the strict boundary against presenting catalogue labels as
personal identity inference.

### [Keyes — The Misgendering Machines](https://ironholds.org/resources/papers/agr_paper.pdf)

Discusses conceptual and social harms in automated gender recognition. Supports careful language and
the non-use boundary.

### [Mitchell et al. — Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596)

Supports documenting intended use, evaluation, limits, and group performance in a model card.

### [Gebru et al. — Datasheets for Datasets](https://doi.org/10.1145/3458723)

Supports structured dataset documentation covering motivation, composition, collection, use, and
limitations.

### [PyTorch reproducibility guide](https://docs.pytorch.org/docs/stable/notes/randomness.html)

Official guidance on seeds, deterministic operations, DataLoader workers, cuDNN behaviour, and the
limits of reproducibility across releases and platforms.
