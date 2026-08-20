# MLA2 — COSC2753 Assignment 2

This is a clean starter repository for four fashion classification targets
(`articleType`, `season`, `gender`, and `usage`) plus Top-K visual search.

## Before working

1. Read `docs/COSC2753_2026B_Assignment 2.pdf`.
2. Read `rubrics/RUBRIC.md`.
3. Read `docs/decisions/README.md` and all accepted decisions.
4. Record a new decision when a choice will constrain later tasks.

Do not treat old Git history or backup branches as current project decisions.

## Hard assignment rules

- Train submitted models from scratch. Pretrained weights are benchmarks only.
- Use `data/processed/splits.csv` as the single shared split once it exists.
- Record every training run in `results/runs.csv`.
- Keep prediction columns in this order:
  `id,gender,articleType,season,usage`.
- Never edit or commit supplied files under `data/raw/`.

## Folder rules

- Keep notebooks narrative. Put reusable logic in `src/fashion/`.
- Put small command-line entry points in `scripts/`.
- Mirror reusable code with tests in `tests/`.
- Put report evidence in `results/figures/`.
- Put rebuildable manifests and the shared split in `data/processed/`.
- Do not add a folder before it is needed.

## Working rules

- Use `./.venv/bin/python`.
- Test new behaviour before calling it complete.
- Make one decision record per important choice.
- Do not silently rewrite an accepted decision. Supersede it with a new record.

## How to talk to me

Talk to me like I'm 5. Small words, short sentences, short paragraphs. If a big word is
needed, explain it right after. Only return what's actually necessary.

Just tell me what you did, did it work, what do I do now.

If I have to decide something: 2 options max, the context I need to pick fast, and which
one you'd go with.

Keep paths and commands exact.

## Where to look

- `rubrics/RUBRIC.md` — marking bands and HD checklists. Read when scoping work or
  deciding what to cut.
- `docs/COSC2753_2026B_Assignment 2.pdf` — the spec. Read for deliverables, submission
  format, and naming conventions.
- `docs/decisions/` — accepted project choices and the template for recording new ones.
  Read every accepted record before proposing changes to structure or workflow.
