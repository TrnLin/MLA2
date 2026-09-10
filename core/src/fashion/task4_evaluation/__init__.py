"""Task 4 one-shot sealed-holdout evaluation.

This package is deliberately separate from :mod:`fashion.task4`. The retrieval and
training package is AST-scanned by ``tests/test_documentation.py`` and must never
mention the protected-label unlock; this package is the single audited place that may.
"""

from __future__ import annotations

from fashion.task4_evaluation.audit import load_verified_holdout_evaluation
from fashion.task4_evaluation.blind import (
    EVALUATION_MANIFEST_PATH,
    FINAL_EVALUATION_DIR,
    FINAL_EVALUATION_FIGURE_DIR,
    PREDICTION_RECEIPT_PATH,
    UNLOCK_RECEIPT_PATH,
    build_blind_holdout_evidence,
)
from fashion.task4_evaluation.score import score_holdout
from fashion.task4_evaluation.spec import (
    HoldoutEvaluationSpec,
    load_holdout_evaluation_spec,
)
from fashion.task4_evaluation.views import (
    DEVELOPMENT_ROWS_EXPECTED,
    HOLDOUT_ROWS_EXPECTED,
    QUARANTINE_ROWS_EXPECTED,
    build_holdout_views,
)

__all__ = [
    "DEVELOPMENT_ROWS_EXPECTED",
    "EVALUATION_MANIFEST_PATH",
    "FINAL_EVALUATION_DIR",
    "FINAL_EVALUATION_FIGURE_DIR",
    "HOLDOUT_ROWS_EXPECTED",
    "PREDICTION_RECEIPT_PATH",
    "QUARANTINE_ROWS_EXPECTED",
    "UNLOCK_RECEIPT_PATH",
    "HoldoutEvaluationSpec",
    "build_blind_holdout_evidence",
    "build_holdout_views",
    "load_verified_holdout_evaluation",
    "load_holdout_evaluation_spec",
    "score_holdout",
]
