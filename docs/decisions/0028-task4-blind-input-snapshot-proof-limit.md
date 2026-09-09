# 0028 — Task 4 blind-input snapshot proof limit

- Status: Accepted
- Date: 2026-09-09

## Context

The real blind ranking run used source commit
`bf64d2b8e350f29d658eb1496eaec40a3048f836`. That code read the evaluation
config, split CSV, compressed variant index, gallery manifest, model manifest,
and run registry from their live paths. It used those six non-image inputs and
then reopened the paths to make the receipt hashes and byte counts.

The tree was clean, the source was committed, and the final files match the
receipt hashes. These facts reduce concern, but they do not prove that no file
changed between use and the later hash. Query and gallery images were checked
from the same bytes used for encoding. The published rankings were also
byte-bound before labels were opened.

The holdout was opened once. The scores stand and the blind run cannot be
repeated to improve its historical proof.

## Decision

Keep the scored result and disclose this proof limit beside the final
judgement. Future blind runs capture each of the six receipt-listed non-image
inputs exactly once before its first parse or use. Parsers, validators, model
and gallery identity checks, ranking work, and receipt records all use those
captured bytes or private snapshots made from them.

The public audit also requires exactly the five blind output records nested in
`prediction_receipt.json` and checks each fixed path, SHA-256, and byte count.

## Consequences

The historical result is not described as proof against a mid-run mutation.
Its clean commit and matching final hashes remain useful supporting evidence.
Its image-byte and ranking-byte bindings remain valid.

Future receipt identity and ranking computation cannot diverge because of a
live change to the six non-image inputs after capture. No model, config, split,
ranking, metric, approval tag, score, or frozen final-evaluation artifact is
changed by this decision.

## Evidence

- `src/fashion/task4_evaluation/blind.py`
- `src/fashion/task4_evaluation/audit.py`
- `tests/task4_evaluation/test_blind.py`
- `tests/task4_evaluation/test_score.py`
- `tests/task4_evaluation/test_notebook.py`
- `notebooks/task-4/10_task4_part3_final_evaluation.ipynb`
