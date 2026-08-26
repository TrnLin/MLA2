# Task 3 problem framing and lifecycle

[Research index](README.md) · [Next: data and validation](02_data_and_validation_design.md) ·
[References](references.md)

## 1. Purpose

Task 3 learns two image classifiers for the supplied fashion catalogue:

- a five-class `gender` output;
- a nine-class `usage` output.

The official prediction file keeps these as separate columns:

```text
id,gender,articleType,season,usage
```

This plan covers development, evaluation, final refit, independent holdout use, official
predictions, application checks, and monitoring. It does not claim that the catalogue labels are
objective facts about a person or an item.

## 2. Problem statement

**Repository fact.** The input is a teacher product image, usually only 60×80 pixels. The outputs
are the teacher metadata values in the project label maps. The assignment allows comparison of
methods but requires the submitted model to be trained from scratch. See the
[assignment specification](<../../COSC2753_2026B_Assignment 2.pdf>),
[problem-definition notebook](../../../notebooks/00_problem_definition.ipynb), and
[label maps](../../../data/processed/label_maps.json).

**Recommendation.** Define the operational problem as:

> Given one teacher catalogue image, suggest the supplied catalogue target-audience label and the
> supplied usage label, with confidence and a human-review path.

The word “suggest” is deliberate. These labels are merchant metadata. Some are subjective, weakly
visual, or socially sensitive.

## 3. Intended users and decisions

### Intended users

- The student team comparing machine-learning approaches.
- A catalogue editor reviewing a proposed label.
- The assignment application, which needs deterministic Task 3 predictions.

### Decisions supported

| Output | Supported decision | Required caution |
|---|---|---|
| `gender` label | Suggest which supplied catalogue audience bucket best matches the image | It is not a statement about a visible person or customer identity |
| `usage` label | Suggest which supplied merchant usage bucket best matches the image | Usage may be an occasion or business label that is not fully visible |
| Confidence | Decide whether to accept or review the suggestion | Raw softmax confidence is not automatically trustworthy |
| Top alternatives | Help a reviewer correct uncertain cases | Alternatives remain within the fixed supplied taxonomy |

### Decisions not supported

Do not use this model to:

- infer a customer’s or photographed person’s gender identity;
- recognise a face or person;
- infer age, sexuality, body type, ethnicity, or another personal trait;
- decide access, pricing, employment, credit, safety, or another high-impact outcome;
- restrict which products a person may view or buy;
- claim that the supplied catalogue taxonomy is complete or socially neutral.

## 4. Label meaning

### `gender`

The development labels are `Men`, `Women`, `Unisex`, `Boys`, and `Girls`.

**Repository fact.** They are teacher catalogue values. The dataset does not provide evidence that
they describe the identity of any person visible in an image.

**Recommendation.** In the report and application, call this output a “catalogue target-audience
label.” Avoid phrases such as “detect gender,” “recognise male/female,” or “infer identity.”

Important interpretations:

- `Unisex` is a real catalogue class, not an abstention state.
- `Boys` and `Girls` mix age and marketing categories with gendered language.
- The five labels do not represent all identities.
- A visually stereotyped pattern may reflect catalogue practice, not an inherent product property.

### `usage`

The development labels are `Casual`, `Sports`, `Ethnic`, `Formal`, `NA`, `Smart Casual`, `Travel`,
`Party`, and `Home`.

**Repository fact.** Literal `NA` is a valid teacher label. Only a truly blank value is missing.
`Home` has one development product.

**Recommendation.** Treat usage as a supplied merchant context label. Do not silently reinterpret
`NA` as missing or “not applicable.” Document that its business meaning is uncertain unless better
dataset provenance becomes available.

Usage is likely harder than many visual object labels because:

- `Casual` versus `Smart Casual` may depend on merchant judgement;
- `Travel`, `Party`, and `Home` can describe context rather than visible construction;
- several labels may be plausible for the same product;
- one image may contain a person or several products that distract from the labelled item.

## 5. Why the outputs stay separate

**Repository facts.** There are five gender classes and nine usage classes, but only 26 of the 45
possible pairs appear in development. One row has a valid gender label and a missing usage label.
The official CSV needs two separate columns.

**Recommendation.** Reject a combined 45-way class because it would:

- make rare pairs even rarer;
- discard valid gender supervision on the missing-usage row;
- be unable to express unseen combinations cleanly;
- make per-output error, calibration, and uncertainty hard to explain;
- add conversion work just to recreate the required separate columns.

Separate heads preserve the output contract while still allowing a shared visual backbone to be
tested.

## 6. Industry lifecycle standards

### Sourced practice

ISO/IEC 5338 defines processes for controlling, executing, managing, and improving an AI system over
its life cycle. [ISO/IEC 5338:2023](https://www.iso.org/standard/81118.html).

ISO/IEC 23894 describes AI-specific risk management that organisations can adapt to their context.
[ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html).

The NIST AI Risk Management Framework organises work into Govern, Map, Measure, and Manage. It says
risk work should be continuous across the life cycle. As of August 2026, NIST states that AI RMF 1.0
is being updated, so this project should use the published 1.0 framework while noting that status.
[NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/).

### Task 3 mapping

| Lifecycle phase | NIST emphasis | Task 3 activity | Exit condition |
|---|---|---|---|
| Govern | Ownership, rules, accountability | Name task owner, data owner, holdout approver, report owner, and application owner | Roles and protected-data rules are written |
| Map | Context, users, impacts, limits | Define catalogue-only use, label meaning, non-use, affected users, and known ambiguity | Problem statement and risk register are approved |
| Understand data | Data suitability and limits | Audit counts, families, image properties, masks, shortcuts, and class support | Data contract and five-fold plan are fixed |
| Measure | Tests and evidence | Run controlled baselines, OOF metrics, calibration, robustness, uncertainty, and cost tests | Comparable registered evidence exists |
| Manage | Accept, reduce, transfer, or reject risk | Choose separate/shared model, define review, reject unsafe claims, and freeze the method | Selection gates pass or limitations are accepted explicitly |
| Independently evaluate | Unseen evidence | Refit on development and open the holdout once | Frozen holdout report exists |
| Release | Correct operation | Validate schema, checkpoints, app behaviour, latency, and human review | Handoff checklist passes |
| Monitor | Detect change and failure | Track drift, overrides, reviewed errors, calibration, and operational faults | Trigger and owner are documented |
| Retire or retrain | Controlled change | Stop an unsafe model or repeat the whole evidence process | No silent online learning |

## 7. Ethics and risk framing

### Sourced practice

NIST SP 1270 explains that bias can enter across the full AI life cycle and can cause harm even when
intent is good. [NIST SP 1270](https://www.nist.gov/publications/towards-standard-identifying-and-managing-bias-artificial-intelligence).

Research on facial gender classification found large error differences across demographic groups.
This project is not a facial-analysis task, but that work shows why it must not be presented as
personal gender inference. [Gender Shades](https://proceedings.mlr.press/v81/buolamwini18a.html).

### Task 3 ethical risks

| Risk | Example | Control |
|---|---|---|
| Identity overclaim | Saying the model identifies a person’s gender | Use catalogue-audience wording and forbid person inference |
| Stereotype learning | Colour or article type becomes a proxy for gender | Per-class slices, shortcut probes, explanations, and human review |
| Taxonomy exclusion | Five audience labels are treated as complete human categories | State that the taxonomy is supplied and limited |
| Weak visual truth | A product is labelled `Travel` although nothing visual proves it | Describe usage as merchant metadata and allow override |
| Rare-class overclaim | One correct `Home` guess is reported as generalisation | Report support and make `Home` review-only |
| Unequal product exposure | Wrong catalogue label hides an item | Provide alternatives, confidence, and an editor override |
| Automation bias | Reviewer accepts a confident wrong label | Calibrate confidence and show a review warning |

The dataset does not provide demographic attributes needed for a proper demographic fairness audit.
The team must not claim demographic fairness from class-level gender results.

## 8. Success definition

Task 3 succeeds only when all four layers are credible.

### 8.1 Scientific success

- Every claimed comparison uses the same split and validation scope.
- The selected model beats simple baselines on predeclared primary metrics.
- Per-class failures, fold variation, seed variation, and uncertainty are shown.
- The report explains why rejected models lost.

### 8.2 Project-contract success

- The output columns and label maps remain fixed.
- Missing-label masks are correct.
- Submitted models start from random weights.
- Every run is recorded through the required registry.
- Holdout labels stay sealed until the final gate.

### 8.3 Operational success

- Predictions contain no invalid labels or missing required values.
- Preprocessing is identical in batch and application use.
- Latency, memory, and checkpoint size fit the named device.
- Low-confidence and rare predictions have a review path.

### 8.4 Ethical success

- The system is described as a catalogue-label assistant.
- It is not used for person inference or high-impact decisions.
- Uncertainty and limitations are visible to the reviewer.
- `Home` and other low-support labels are not presented as well-validated capabilities.

Detailed numeric gates are defined in the
[evaluation framework](05_evaluation_framework.md). The freeze and holdout procedure is defined in
[final selection and deployment](07_final_selection_and_deployment.md).

## 9. Decisions that are closed

The following choices are already fixed by project decisions and must not be reopened in Task 3:

- one canonical split file;
- existing development, holdout, and quarantine membership;
- five saved family-safe fold assignments;
- teacher images only for Tasks 1–3;
- all development-observed labels stay in stable maps;
- literal `NA` stays valid;
- low-support classes are not deleted, merged, or masked;
- separate `gender` and `usage` official columns;
- holdout access occurs only after final method freeze.

See [development/holdout boundary](../../decisions/0014-development-holdout-cv-boundary.md),
[teacher-only preparation](../../decisions/0015-teacher-only-shared-image-preparation.md), and
[development label scope](../../decisions/0016-development-label-scope.md).

## 10. Decisions that experiments must answer

- Best scratch backbone for each target.
- Native-like versus upsampled input.
- Useful augmentation strength.
- Whether class-balanced loss helps.
- Whether one shared backbone is non-inferior to separate models.
- Whether a larger CNN is justified.
- Whether post-training calibration improves confidence.
- Which model gives the best performance-cost-risk tradeoff.

These are answered by [03_model_choice.md](03_model_choice.md),
[04_experiment_plan.md](04_experiment_plan.md), and
[05_evaluation_framework.md](05_evaluation_framework.md), not by preference alone.
