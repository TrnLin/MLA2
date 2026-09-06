from __future__ import annotations

import hashlib
import html
import importlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import nbformat
import pytest

from fashion.config import ROOT
from fashion.task4 import report_figures
from scripts.task4 import generate_model_comparison_report as report

BUNDLE_RELATIVE = Path(
    "results/evidence/task4/final/task4-final-comparison.json"
)
TEMPLATE_RELATIVE = Path(
    "scripts/task4/templates/task4-model-comparison-base.html"
)
NOTEBOOK = ROOT / "notebooks/task-4/05_task4_visual_search.ipynb"


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256", None)
    content = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _final_freeze_module():
    return importlib.import_module("fashion.task4.final_freeze")


def test_report_generation_preserves_broad_work_and_adds_complete_final_evidence() -> None:
    rendered = report.render_report(root=ROOT)

    required = (
        "HOG + HSV-edge fusion",
        "../figures/task4/hog_fusion/hog_fusion_examples.png",
        "../figures/task4/hog/hog_examples.png",
        "../figures/task4/baseline_examples.png",
        "What changed from the prior candidate",
        "Protocol A",
        "Protocol B",
        "Failure slices",
        "Qualitative examples",
        "Parameter count",
        "Checkpoint bytes",
        "Embedding bytes",
        "Index build time",
        "Peak RSS",
        "Encoding p50",
        "Encoding p95",
        "Search p50",
        "Search p95",
        "End-to-end p50",
        "End-to-end p95",
        "Fold 0",
        "Fold 4",
        "Pooled spread",
        "Teacher-only",
        "V1-only",
        "Two-view",
        "Holdout remains sealed",
    )
    assert all(value in rendered for value in required)
    assert len(rendered.splitlines()) > 1_877


def test_report_declares_one_final_decision_without_stale_progress_claims() -> None:
    rendered = report.render_report(root=ROOT)
    visible_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(rendered)))
    lowered = visible_text.lower()

    forbidden = (
        "provisional",
        "no winner is declared",
        "nothing here is frozen",
        "no final five-fold r5 number yet",
        "r5 — still running",
        "4 of 5",
        "4 / 5",
        "read this as unfinished",
        "these are four numbers, not five",
        "no valid five-fold r5 summary yet",
        "fold 4 had completed epoch 62",
        "started 63/100",
        "indicative only, not decided",
        "indicative, not a frozen final gallery policy",
        "has not been formally run",
        "not formally run",
        "treat the choice as open",
        "choice remains open",
        "gallery policy remains open",
        "gallery study has not run",
    )
    assert all(claim not in lowered for claim in forbidden)
    assert "scratch r5 + teacher-only: final learned decision" in lowered
    assert "final decision. scratch r5 wins." in lowered
    assert "canonical post-study decision is teacher-only" in lowered


def test_compact_bundle_is_trackable_and_supports_clean_checkout_generation(
    tmp_path: Path,
) -> None:
    for relative in (BUNDLE_RELATIVE, TEMPLATE_RELATIVE):
        assert (ROOT / relative).is_file()
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", str(relative)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert ignored.returncode == 1, ignored.stdout

    clean_root = tmp_path / "clean"
    for relative in (BUNDLE_RELATIVE, TEMPLATE_RELATIVE):
        destination = clean_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    rendered = report.render_report(root=clean_root)
    assert "HOG + HSV-edge fusion" in rendered
    assert "Scratch R5 + teacher-only" in rendered


def test_bundle_validator_rejects_hash_schema_and_protected_scope_mutations(
    tmp_path: Path,
) -> None:
    module = _final_freeze_module()
    source = ROOT / BUNDLE_RELATIVE
    validated = module.validate_final_comparison_bundle(source, root=ROOT)
    assert validated["decision"]["method"] == "R5"
    assert validated["decision"]["gallery_policy"] == "teacher"

    destination = tmp_path / BUNDLE_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))

    changed = json.loads(json.dumps(payload))
    changed["source_artifacts"][0]["sha256"] = "0" * 64
    destination.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256|hash"):
        module.validate_final_comparison_bundle(destination, root=tmp_path)

    changed = json.loads(json.dumps(payload))
    changed["unexpected"] = True
    changed["bundle_sha256"] = _canonical_sha256(changed)
    destination.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        module.validate_final_comparison_bundle(destination, root=tmp_path)

    changed = json.loads(json.dumps(payload))
    changed["safety"]["holdout_opened"] = True
    changed["bundle_sha256"] = _canonical_sha256(changed)
    destination.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed|development"):
        module.validate_final_comparison_bundle(destination, root=tmp_path)


def test_bundle_and_notebook_refuse_outside_repository_or_protected_paths(
    tmp_path: Path,
) -> None:
    module = _final_freeze_module()
    outside = tmp_path.parent / "outside-final-comparison.json"
    with pytest.raises(ValueError, match="repository"):
        module.validate_final_comparison_bundle(outside, root=tmp_path)

    protected = tmp_path / "data/raw/teacher/test/task4-final-comparison.json"
    with pytest.raises(ValueError, match="protected"):
        module.validate_final_comparison_bundle(protected, root=tmp_path)

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(
        item
        for item in notebook.cells
        if item.cell_type == "code" and "# task4-final-freeze" in item.source
    )
    namespace: dict[str, object] = {"TASK4_FREEZE_ROOT": ROOT}
    exec(cell.source, namespace)
    assert namespace["task4_freeze"]["decision"]["method"] == "R5"


def test_report_leads_with_visual_summary_and_cites_every_generated_figure() -> None:
    rendered = report.render_report(root=ROOT)

    glance = rendered.index('id="glance"')
    predictions = rendered.index('id="predictions"')
    first_prose = rendered.index('id="purpose"')
    assert glance < predictions < first_prose

    figures = re.findall(r"<figure\b.*?</figure>", rendered, flags=re.S)
    picture_figures = [block for block in figures if "<img" in block]
    assert len(picture_figures) >= len(report_figures.REPORT_FIGURE_NAMES)
    for block in picture_figures:
        assert "<figcaption>" in block
        assert 'alt="' in block
        assert "Source:" in block

    for name in report_figures.REPORT_FIGURE_NAMES:
        citation = f"../figures/task4/final/{name}"
        assert rendered.count(citation) == 1, name
        caption = report_figures.FIGURE_CAPTIONS[name]
        assert html.escape(caption, quote=False) in rendered, name
        owners = [block for block in picture_figures if citation in block]
        assert len(owners) == 1, name
        assert html.escape(caption, quote=False) in owners[0], name

    displayed_broad_examples = (
        "../figures/task4/hog/hog_examples.png",
        "../figures/task4/hog_fusion/hog_fusion_examples.png",
        "../figures/task4/baseline_examples.png",
    )
    for path in displayed_broad_examples:
        owners = [block for block in picture_figures if f'src="{path}"' in block]
        assert len(owners) == 1, path

    assert "scripts/task4/generate_report_figures.py" in rendered


def test_report_generation_is_byte_identical_across_runs() -> None:
    assert report.render_report(root=ROOT) == report.render_report(root=ROOT)


def test_report_generation_refuses_a_template_that_drops_a_figure_citation(
    tmp_path: Path,
) -> None:
    clean_root = tmp_path / "clean"
    for relative in (BUNDLE_RELATIVE, TEMPLATE_RELATIVE):
        destination = clean_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    template_path = clean_root / TEMPLATE_RELATIVE
    dropped = report_figures.REPORT_FIGURE_NAMES[0]
    template_path.write_text(
        template_path.read_text(encoding="utf-8").replace(
            f"../figures/task4/final/{dropped}",
            "../figures/task4/final/removed.png",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="figure"):
        report.render_report(root=clean_root)


def test_canvas_is_generated_from_strict_freeze_in_clean_checkout(
    tmp_path: Path,
) -> None:
    module_name = "scripts.task4.generate_model_comparison_canvas"
    module_spec = importlib.util.find_spec(module_name)
    assert module_spec is not None, "trackable Canvas generator is missing"
    canvas_report = importlib.import_module(module_name)

    clean_root = tmp_path / "clean"
    for relative in (BUNDLE_RELATIVE, canvas_report.TEMPLATE_RELATIVE_PATH):
        destination = clean_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    source = canvas_report.render_canvas(root=clean_root)
    assert canvas_report.managed_canvas_path(
        root=Path("/work/checkouts/MLA2"),
        home=Path("/home/student"),
    ) == Path(
        "/home/student/.cursor/projects/work-checkouts-MLA2/"
        "canvases/task4-model-comparison.canvas.tsx"
    )
    assert "Generated from strict final freeze SHA-256: 83ca0730" in source
    assert source.count('from "cursor/canvas"') == 1
    assert all(
        forbidden not in source
        for forbidden in ("fetch(", "http://", "https://", "XMLHttpRequest", "WebSocket")
    )
    assert re.search(r"""from ["']\.""", source) is None
    assert "Six final methods, ten stability runs" in source
    assert (
        '{ method: "R5", kind: "scratch candidate", '
        "score: 0.5028872133026174"
    ) in source
    assert (
        "r5: [0.507230331944694, 0.5028872133026174, "
        "0.49770189355926997, 0.5070024267953162, 0.506282916520546]"
    ) in source
    assert (
        '{ policy: "Teacher-only", normalization: "teacher", '
        "quality: 0.5029756263849006"
    ) in source

    output = canvas_report.write_canvas(
        tmp_path / "task4-model-comparison.canvas.tsx",
        root=clean_root,
    )
    assert output.read_text(encoding="utf-8") == source

    changed = json.loads((clean_root / BUNDLE_RELATIVE).read_text(encoding="utf-8"))
    changed["safety"]["holdout_opened"] = True
    changed["bundle_sha256"] = _canonical_sha256(changed)
    (clean_root / BUNDLE_RELATIVE).write_text(
        json.dumps(changed),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed"):
        canvas_report.render_canvas(root=clean_root)


def test_canvas_charts_have_explicit_axis_labels_with_units() -> None:
    canvas_report = importlib.import_module(
        "scripts.task4.generate_model_comparison_canvas"
    )
    source = canvas_report.render_canvas(root=ROOT)
    assert source.count("X-axis:") == 3
    assert source.count("Y-axis:") == 3
    assert "Y-axis: Mean linear nDCG@10 (unitless score)" in source
    assert "X-axis: Validation fold (fold number)" in source
