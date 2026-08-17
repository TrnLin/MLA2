# COSC2753 Machine Learning — Assignment 2 Marking Rubric

> Transcribed from `rubrics/Screenshot 2026-08-10 at 19.17.04.png` and
> `rubrics/Screenshot 2026-08-10 at 19.17.31.png`.
> Canvas title: **"COSC2753 Machine Learning Project_2024"**. Total: **100 points**.

## Summary of weightings

| Criterion | Points | Share |
|---|---|---|
| Approach | 50 | 50% |
| Ultimate Judgement and Evaluation | 30 | 30% |
| Report Presentation | 20 | 20% |
| **Total** | **100** | **100%** |

**Critical implication:** there is **no criterion for raw model accuracy**. Marks come from
the breadth and justification of the *investigation*, the quality of the *ultimate judgement*
and its *independent* evaluation, and the *readability* of the report. A model with mediocre
macro-F1 that is thoroughly investigated and honestly analysed outscores a high-accuracy model
presented without rationale.

---

## Criterion 1 — Approach (50 pts)

| Grade | Band |
|---|---|
| HD | 50 to >39.0 pts |
| DI | 39 to >30.0 pts |
| CR | 30 to >25.0 pts |
| PA | 25 to >17.0 pts |
| NN | 17 to >0 pts |

### HD (50 to >39.0)
Outstanding across the course. The approach is an excellent and extremely thorough investigation
of the chosen ML problem. It explored **multiple algorithms and techniques** for solving the
problem. It **goes beyond using the tools provided in class**. There are **no gaps** in what the
investigation has considered. The approach makes careful consideration of the **unique aspects of
the chosen ML problem**.
The choices in the investigation are **highly justified** and have an excellent explanation. The
reader is left with **no questions**.
The design of the approach includes excellent choice of **training data selection, preprocessing,
model training, parameter tuning, and evaluation**. Be able to showcase your understanding not only
of building the ML model but also **of how to integrate the ML model into a real-world application**.

### DI (39 to >30.0)
The approach is a good and reasonably thorough investigation of the chosen ML problem. It is
**mostly limited to the techniques and algorithms discussed in class**. There are **small gaps**
between what could have been explored. The approach considers, but not fully, the unique aspects
of the chosen ML problem.
The choices in the investigation are justified and have a good explanation. The reader is left
with **minor questions**.
The design of the approach includes all necessary steps including choice of training data
selection, pre-processing, model training, parameter tuning, and evaluation. Be able to showcase
understanding of both building the ML model and integrating it into a real-world application.

### CR (30 to >25.0)
The approach is sufficient, but **not a thorough** investigation. It is limited to the techniques
and algorithms discussed in class, **such as a pure reproduction of lab code**. There are gaps in
the investigation and **alternative algorithms or techniques are better** than the ones in the
approach. The approach has a **limited consideration** of the unique aspects of the problem.
The choices have sufficient justification, however there are **unexplained choices** and the reader
is left with important open questions.
The design includes **most** of the necessary steps (training data selection, preprocessing, model
training, parameter tuning, evaluation).

### PA (25 to >17.0)
The approach is a minimally sufficient investigation. It only examines the **bare minimum**
requirements of suitable techniques and algorithms. There are **many gaps** and there are algorithms
or techniques that are clearly **more suited** to the problem. The approach has limited or minimal
consideration of the unique aspects of the problem.
The choices have the **bare minimal justification**. The reader is left with important unanswered
questions that are not considered.
The design includes most of the necessary steps.

### NN (17 to >0)
Poor, superficial, or incomplete approach that does not meet the minimum requirements for PA.
**Does not justify** the approach. **Does not explain** the approach. The design is **missing
significant steps** such as the choice of training data selection, pre-processing, model training,
parameter tuning, and evaluation.

---

## Criterion 2 — Ultimate Judgement and Evaluation (30 pts)

| Grade | Band |
|---|---|
| HD | 30 to >24.0 pts |
| DI | 24 to >20.0 pts |
| CR | 20 to >15.0 pts |
| PA | 15 to >10.0 pts |
| NN | 10 to >0 pts |

### HD (30 to >24.0)
Outstanding across the course. The Ultimate Judgement is established and **exceptionally justified**.
Evaluation of the Ultimate Judgement is **exceptional and clearly demonstrated (or proves) the
viability of the trained model in real-world practice**. The evaluation is **independent**.

### DI (24 to >20.0)
Ultimate Judgement is established and **suitably justified**. Evaluation is sound and suitably
explained, however the reader may **not be fully convinced** and may have minor questions.
The evaluation is independent.

### CR (20 to >15.0)
Ultimate Judgement is established, but there are **unexplained choices**, or the justification is
**hard to follow**. A sufficient attempt at evaluating the Ultimate Judgement is made.
The evaluation is independent.

### PA (15 to >10.0)
An Ultimate Judgement is made but **not justified**. The Ultimate Judgement is **not evaluated**, or
the evaluation is incomplete and insufficient. The evaluation is **not independent**.

### NN (10 to >0)
Not completed. An Ultimate Judgement is not made.

---

## Criterion 3 — Report Presentation (20 pts)

| Grade | Band |
|---|---|
| HD | 20 to >17.0 pts |
| DI | 17 to >12.0 pts |
| CR | 12 to >9.0 pts |
| PA | 9 to >6.0 pts |
| NN | 6 to >0 pts |

### HD (20 to >17.0)
Outstanding across the course. Report is **easy to read and flows well**. It is **structured well**,
leading the reader to **fully understand the rationale** for the final approach taken. Approaches are
**excellently described**. Tables, figures and other visualisation are **tailored to the descriptions
and justifications** made in the report's text.

### DI (17 to >12.0)
Report is reasonably easy to read and flows relatively well. Structured reasonably well, leading the
reader to reasonably understand the rationale for the final approach. Approaches are described well.
Tables, figures and visualisations are easy to read and interpret, but **may not be highly relevant
or appropriately tailored** to the descriptions and justifications in the text.

### CR (12 to >9.0)
Report can be followed but **does not flow well in places**. It is adequately structured, but the
reader may find it difficult to understand the rationale of the selected approach. Approaches are
described in sufficient detail, but aspects may be incomplete. Tables, figures and other
visualisation are sufficient but **difficult to interpret**.

### PA (9 to >6.0)
Report is **difficult to follow** and doesn't flow well. Readers find it difficult to understand the
rationale of the selected approach. Approaches are described to a minimal standard. Tables, figures
and other visualisation are provided to a **bare minimum** amount.

### NN (6 to >0)
Not completed. Incomplete or error-ridden report.

---

## Checklist distilled from the rubric

Use this as the acceptance test for the project.

### To score HD on Approach (needs ≥39.1/50)
- [ ] **Multiple algorithms compared per task** — not one model per task. Classic ML baseline + CNN + tuned CNN, minimum.
- [ ] **Goes beyond class material** — e.g. metric learning / triplet loss, multi-task heads, Grad-CAM, focal loss, calibration. At least one technique not covered in labs.
- [ ] **No gaps** — every stage documented: data selection, cleaning, splitting, preprocessing, augmentation, architecture, hyper-parameter tuning, evaluation.
- [ ] **Unique aspects of THIS problem addressed explicitly** — 60×80 tiny images, 125-class long tail, `"NA"` string vs blank, missing images, grayscale images, weak season/usage signal.
- [ ] **Every choice justified with evidence**, not opinion. "X because Y, and Y is shown in Figure Z."
- [ ] **Real-world integration demonstrated** — a GUI/API layer, not terminal-only. Explicitly named in both the HD and DI bands.

### To score HD on Ultimate Judgement (needs ≥24.1/30)
- [ ] A clearly stated **ultimate judgement** per task: which model would you deploy and why.
- [ ] Justification that **goes beyond a single metric** — error analysis, per-class breakdown, calibration, latency, model size, failure modes, cost of mistakes.
- [ ] Evidence that **proves viability in real-world practice** — robustness checks, inference cost, behaviour on degraded inputs.
- [ ] **Independent evaluation** — a held-out set never used for tuning, AND comparison against external published results / other works on the same dataset.

### To score HD on Report Presentation (needs ≥17.1/20)
- [ ] Max 5 pages of text + 2 pages appendix, 11pt, single column.
- [ ] Structure that walks the reader to the conclusion; no orphaned sections.
- [ ] **Every figure/table is referenced in the text and supports a specific claim.** Decorative plots cost marks.
- [ ] Names and student IDs of all group members included.
- [ ] Over-length content is not marked — only the first 5 pages of text are read.

---

## Notes

- The rubric header says "Project_2024" but it is the rubric attached to the 2026B Assignment 2 Canvas page.
- The rubric contains **no bonus and no penalty for prediction accuracy on the test set**. The
  prediction CSV is a submission requirement, not a scored criterion in this rubric.
- Spec-level requirements that are *not* in this rubric but still mandatory (name conventions,
  zip contents, README, ≥4 models, from-scratch training) are in
  `COSC2753_2026B_Assignment 2.pdf` and summarised in `CLAUDE.md`.
