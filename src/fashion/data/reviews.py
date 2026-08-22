"""Audit requirements for manual image-review ledgers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REVIEW_AUDIT_COLUMNS = {
    "reviewer_initials",
    "review_date",
    "review_method_or_tool",
    "blind_to_metrics",
    "independent_review",
    "second_reviewer_initials",
    "second_review_decision",
    "second_review_status",
    "disagreement_resolution",
    "signoff_status",
}
SIGNOFF_STATUSES = {"pending_team_signoff", "signed_off"}
SECOND_REVIEW_STATUSES = {"not_required", "pending", "agreed", "disagreed_resolved"}


def validate_review_ledger(review: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Validate audit fields without filling missing human provenance."""
    if "decision" not in review:
        raise ValueError(f"review file {path} is missing decision")
    if missing := REVIEW_AUDIT_COLUMNS.difference(review.columns):
        raise ValueError(f"review file {path} is missing audit columns: {sorted(missing)}")
    if unknown := set(review["signoff_status"]) - SIGNOFF_STATUSES:
        raise ValueError(f"review file {path} has unknown sign-off states: {sorted(unknown)}")

    signed = review[review["signoff_status"].eq("signed_off")]
    required_signed = (
        "reviewer_initials",
        "review_date",
        "review_method_or_tool",
        "blind_to_metrics",
        "independent_review",
        "second_review_status",
    )
    for column in required_signed:
        if signed[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"signed review file {path} has blank {column}")
    for column in ("blind_to_metrics", "independent_review"):
        values = set(signed[column].astype(str).str.lower())
        if not values.issubset({"true", "false"}):
            raise ValueError(f"signed review file {path} has invalid {column}")
    if not set(signed["second_review_status"]).issubset(SECOND_REVIEW_STATUSES):
        raise ValueError(f"signed review file {path} has invalid second-review status")
    if signed["second_review_status"].eq("pending").any():
        raise ValueError(f"signed review file {path} still has a pending second review")
    second_reviewed = signed["second_review_status"].isin({"agreed", "disagreed_resolved"})
    for column in ("second_reviewer_initials", "second_review_decision"):
        if signed.loc[second_reviewed, column].astype(str).str.strip().eq("").any():
            raise ValueError(f"signed review file {path} has blank {column}")
    decision_needs_second = signed["decision"].isin({"different", "uncertain"})
    if (~second_reviewed[decision_needs_second]).any():
        raise ValueError(
            f"signed review file {path} needs a second review for different/uncertain calls"
        )
    disagreements = signed["second_review_status"].eq("disagreed_resolved")
    if signed.loc[disagreements, "disagreement_resolution"].astype(str).str.strip().eq("").any():
        raise ValueError(f"signed review file {path} has an unresolved disagreement")

    pending = int(review["signoff_status"].eq("pending_team_signoff").sum())
    blind = signed["blind_to_metrics"].astype(str).str.lower().eq("true")
    independent = signed["independent_review"].astype(str).str.lower().eq("true")
    return {
        "status": "pending_team_signoff" if pending else "signed_off",
        "total_rows": len(review),
        "signed_off_rows": len(review) - pending,
        "pending_rows": pending,
        "human_provenance_complete": pending == 0,
        "blind_review_rows": int(blind.sum()),
        "independent_review_rows": int(independent.sum()),
        "second_reviewed_rows": int(second_reviewed.sum()),
        "hd_review_evidence_complete": bool(pending == 0 and blind.all() and independent.all()),
    }
