"""Regenerate every figure cited by the final Task 4 comparison report."""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib

from fashion.config import ROOT
from fashion.task4 import report_figures
from scripts.task4.run_model_comparisons import reject_sealed_image_rows


def _refuse_sealed_images(evidence: report_figures.WinnerRetrievalEvidence) -> None:
    """Re-apply the runner's protected-path refusal to every drawn image row."""

    panels = (
        *report_figures.select_retrieval_panels(evidence, kind="success"),
        *report_figures.select_retrieval_panels(evidence, kind="failure"),
        *report_figures.slice_retrieval_panels(evidence),
    )
    ids = sorted(
        {panel.query_id for panel in panels}
        | {result.candidate_id for panel in panels for result in panel.results}
    )
    reject_sealed_image_rows(
        report_figures.resolve_image_rows(evidence.catalogue, ids)
    )


def main() -> int:
    evidence = report_figures.load_winner_retrieval_evidence(root=ROOT)
    _refuse_sealed_images(evidence)
    outputs = {
        **report_figures.build_chart_figures(root=ROOT),
        **report_figures.build_retrieval_figures(root=ROOT, evidence=evidence),
    }
    if sorted(outputs) != sorted(report_figures.REPORT_FIGURE_NAMES):
        raise ValueError("generated figure set does not match the cited figure names")
    for name in report_figures.REPORT_FIGURE_NAMES:
        path = Path(outputs[name])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(ROOT)} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
