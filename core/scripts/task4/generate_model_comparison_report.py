"""Generate the final Task 4 development-only comparison report."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

from fashion.config import ROOT
from fashion.task4.final_freeze import (
    FINAL_FREEZE_RELATIVE_PATH,
    validate_final_comparison_bundle,
)
from fashion.task4.report_figures import (
    FIGURE_CAPTIONS,
    FIGURE_DIR_RELATIVE,
    REPORT_FIGURE_NAMES,
)

OUTPUT = ROOT / "results/reports/task4-model-comparison.html"
TEMPLATE_RELATIVE_PATH = Path(
    "scripts/task4/templates/task4-model-comparison-base.html"
)
FIGURE_CITATION_PREFIX = (
    f"../{FIGURE_DIR_RELATIVE.relative_to('results').as_posix()}/"
)


def _figure_caption_token(name: str) -> str:
    return "{{FIGURE_CAPTION_" + Path(name).stem.upper() + "}}"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table(headers: list[str], rows: list[list[object]]) -> str:
    header = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _score_chart(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for item in sorted(
        candidates,
        key=lambda value: value["selected_metrics"]["development_winner_score"],
        reverse=True,
    ):
        score = item["selected_metrics"]["development_winner_score"]
        rows.append(
            "<div class='bar-row'>"
            f"<strong>{html.escape(item['method'])}</strong>"
            "<div class='bar-track'>"
            f"<div class='bar' style='width:{score * 100:.4f}%'></div>"
            "</div>"
            f"<span>{score:.6f}</span>"
            "</div>"
        )
    return "".join(rows)


def _load(root: Path = ROOT) -> dict[str, Any]:
    """Load only the public, strict, compact final freeze."""

    return validate_final_comparison_bundle(
        root / FINAL_FREEZE_RELATIVE_PATH,
        root=root,
    )


def _render(data: dict[str, Any]) -> str:
    candidates = data["candidates"]
    deployment = data["deployment"]
    gallery = data["gallery"]
    winner = deployment["selected_model"]

    candidate_rows = []
    for item in sorted(candidates, key=lambda value: value["method"]):
        metrics = item["selected_metrics"]
        cost = data["costs"][item["method"]]
        p95_values = [
            row["value_seconds"]
            for row in cost["timing_summary"]
            if row["metric"] == "end_to_end"
            and row["percentile"] == "p95"
        ]
        candidate_rows.append(
            [
                item["method"],
                item["run_kind"],
                f"{metrics['development_winner_score']:.6f}",
                f"{metrics['cross_source_score']:.6f}",
                f"{metrics['source_robustness_ratio']:.4f}",
                f"{max(p95_values):.4f}",
                f"{max(cost['index_bytes'].values()) / 1_000_000:.2f}",
                "PASS" if all(item["gates"].values()) else "FAIL",
            ]
        )

    fold_rows = []
    spread_rows = []
    for summary in deployment["stability_summaries"]:
        for fold in summary["folds"]:
            fold_rows.append(
                [
                    summary["method"],
                    fold["fold"],
                    f"{fold['score']:.6f}",
                    f"{fold['coverage'] * 100:.3f}%",
                ]
            )
        spread_rows.append(
            [
                summary["method"],
                f"{summary['mean']:.9f}",
                f"{summary['sample_standard_deviation']:.9f}",
            ]
        )

    canvas_rows = [
        [
            row["method"],
            row["query_variant"],
            f"{float(row['ndcg_at_10']):.6f}",
            f"{float(row['ndcg_change_from_clean']):.6f}",
            f"{float(row['mean_top10_overlap']):.4f}",
        ]
        for row in sorted(
            data["canvas"],
            key=lambda value: (value["method"], value["query_variant"]),
        )
    ]
    gallery_rows = [
        [
            item["policy"],
            item["query_normalization_source"],
            f"{item['quality_at_10']:.9f}",
            f"{item['p95_end_to_end_seconds']:.6f}",
            f"{item['index_bytes'] / 1_000_000:.2f}",
            "SELECTED"
            if item["policy"] == gallery["final_policy"]["policy"]
            else "rejected",
        ]
        for item in gallery["policies"]
    ]
    attempt_rows = [
        [
            row.get("run_id", ""),
            row.get("method", ""),
            row.get("status", ""),
            row.get("error_type", ""),
            row.get("error_message", ""),
        ]
        for row in data["attempts"]
    ]
    hash_rows = [
        [path, value] for path, value in sorted(data["hashes"].items())
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task 4 learned model comparison</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#566474; --line:#d6dde5;
  --paper:#f5f7f9; --card:#ffffff; --blue:#1769aa; --green:#19734a; --amber:#9a5b00; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:38px 24px 64px; }}
h1 {{ font-size:36px; line-height:1.1; margin:0 0 8px; }}
h2 {{ margin:42px 0 12px; border-bottom:2px solid var(--line); padding-bottom:8px; }}
h3 {{ margin:24px 0 8px; }}
p {{ max-width:85ch; }}
.eyebrow {{ color:var(--blue); font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
.lead {{ color:var(--muted); font-size:18px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }}
.value {{ font-size:28px; font-weight:750; color:var(--green); }}
.unit {{ color:var(--muted); font-size:13px; }}
.bar-row {{ display:grid; grid-template-columns:42px 1fr 76px;
  gap:10px; align-items:center; margin:10px 0; }}
.bar-track {{ height:18px; background:#e7ebef; border-radius:3px; overflow:hidden; }}
.bar {{ height:100%; background:var(--blue); }}
.table-wrap {{ overflow:auto; background:var(--card);
  border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; min-width:700px; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line);
  text-align:left; vertical-align:top; }}
th {{ background:#edf2f6; font-size:13px; }}
tr:last-child td {{ border-bottom:0; }}
.callout {{ border-left:5px solid var(--green); background:#edf8f2; padding:16px 18px; }}
.warning {{ border-left-color:var(--amber); background:#fff7e8; }}
code {{ background:#e9edf1; padding:2px 4px; border-radius:3px; }}
footer {{ margin-top:44px; color:var(--muted); font-size:13px; }}
</style>
</head>
<body><main>
<div class="eyebrow">Development-only evidence freeze</div>
<h1>Task 4 learned model comparison</h1>
<p class="lead">Six final methods, ten stability runs, and one three-policy gallery study.
All numeric values below were read from validated tracked artifacts.
No holdout result is present.</p>

<section class="cards">
<div class="card"><div class="unit">Final scratch method</div>
<div class="value">{winner['method']}</div>
<div class="unit">Source: post-stability deployment judgement</div></div>
<div class="card"><div class="unit">Five-fold mean nDCG@10</div>
<div class="value">{winner['stability_mean']:.6f}</div>
<div class="unit">Development queries; higher is better</div></div>
<div class="card"><div class="unit">Sample SD</div>
<div class="value">{winner['stability_sample_standard_deviation']:.6f}</div>
<div class="unit">Across five fresh folds</div></div>
<div class="card"><div class="unit">Final gallery</div>
<div class="value">{gallery['final_policy']['policy']}</div>
<div class="unit">Source: three-policy development study</div></div>
</section>

<h2>1. Candidate breadth</h2>
<p>Mean linear nDCG@10 on development fold 1. B1 is pretrained and comparison-only.</p>
<div class="card">{_score_chart(candidates)}</div>
{_table(
    ["Method", "Kind", "nDCG@10", "Cross-source nDCG@10", "Source ratio",
     "Worst CPU p95 (s)", "Single-source index (MB)", "Speed/storage gates"],
    candidate_rows,
)}

<h2>2. Five-fold stability</h2>
<p>R5 and R3 were retrained from scratch for validation folds 0–4. Units are mean
per-query nDCG@10 and percent scorable-query coverage.</p>
{_table(["Method", "Fold", "nDCG@10", "Coverage"], fold_rows)}
<h3>Mean and spread</h3>
{_table(["Method", "Five-fold mean nDCG@10", "Sample SD"], spread_rows)}
<div class="callout"><strong>Deployment rule passed.</strong> The mean gap is
{deployment['mean_gap']:.9f}; pooled spread is {deployment['pooled_spread']:.9f}.
The gap is larger, and R5 passed scratch, CPU p95, and index-size gates.</div>

<h2>3. Source and canvas robustness</h2>
<p>The candidate table reports the teacher/V1 source robustness ratio. The table below
uses each manifest's clean, tall, and wide V1-query/V1-gallery canvas evidence.</p>
{_table(
    ["Method", "Query canvas", "nDCG@10", "Change from clean", "Mean Top-10 overlap"],
    canvas_rows,
)}

<h2>4. Gallery policy</h2>
<p>Quality is the equal mean of teacher and V1 query performance at K=10.
Latency is CPU batch-one p95 seconds. Storage is decimal MB.</p>
{_table(
    ["Policy", "Query normalization", "Quality nDCG@10", "CPU p95 (s)",
     "Index (MB)", "Decision"],
    gallery_rows,
)}
<div class="callout"><strong>Teacher-only is final.</strong> R5's original
<code>cost.json</code> V1 field is a pre-study assumption, not this decision.
Canonical source: <code>task9-final-gallery-decision.json</code>.</div>

<h2>5. Failures and retries</h2>
<p>Failed and abandoned attempts are evidence. R1/R2 non-finite-gradient runs and
bounded retries are not removed from the registry.</p>
{_table(["Run ID", "Method", "Status", "Error type", "Short message"], attempt_rows)}
<div class="callout warning"><strong>Known execution concern.</strong> The real gallery
process wrote durable valid outputs and then failed to shut down. The native cause is
unknown. Bounded supervision now prevents false command success.</div>

<h2>6. Final judgement and limits</h2>
<p>Select scratch R5, exact cosine search, minimum-distance product collapse, and a
teacher-only gallery. Reject B1 for submission because it is pretrained. Keep R1–R4,
V1-only, and two-view as comparison evidence.</p>
<ul>
<li>Metadata relevance is a proxy for human visual similarity.</li>
<li>V1 is another view of the same products, not independent data.</li>
<li>Wide and tall white canvases still reduce R5 quality sharply.</li>
<li>CPU timings are machine- and route-specific.</li>
</ul>
<div class="callout"><strong>Holdout sealed.</strong> Only
<code>data/processed/splits.csv</code> defines membership. The holdout, quarantine, and
official teacher-test images were not opened. Notebook 06 may evaluate the holdout once
after the exact R5 refit is registered.</div>

<h2>7. Tracked source hashes</h2>
{_table(["Artifact", "SHA-256"], hash_rows)}
<footer>Generated by <code>scripts/task4/generate_model_comparison_report.py</code>.
The method guide is not used as evidence.</footer>
</main></body></html>
"""


def _direction(row: dict[str, object]) -> str:
    return f"{row['query_source']} → {row['gallery_source']}"


def _final_section(data: dict[str, Any]) -> str:
    methods = data["methods"]
    stability = data["stability"]
    gallery = data["gallery"]
    decision = data["decision"]
    method_rows = [
        [
            item["method"],
            item["factor_change"],
            "comparison-only" if item["pretrained"] else "scratch eligible",
            f"{item['selected_metrics']['development_winner_score']:.9f}",
            f"{item['selected_metrics']['cross_source_score']:.9f}",
            f"{item['selected_metrics']['source_robustness_ratio']:.6f}",
        ]
        for item in methods
    ]
    protocol_a_rows = [
        [
            item["method"],
            _direction(row),
            row["metric"],
            row["aggregation"],
            f"{row['value']:.9f}",
            row["query_count"],
            row["class_count"] or "—",
        ]
        for item in methods
        for row in item["protocol_a"]
    ]
    protocol_b_rows = [
        [
            item["method"],
            _direction(row),
            row["metric"],
            f"{row['value']:.9f}",
            row["query_count"],
        ]
        for item in methods
        for row in item["protocol_b"]
    ]
    slice_rows = [
        [
            item["method"],
            _direction(row),
            row["slice"],
            row["metric"],
            "undefined" if row["value"] is None else f"{row['value']:.9f}",
            f"{row['coverage'] * 100:.3f}%" if row["coverage"] is not None else "—",
            row["caveat"],
        ]
        for item in methods
        for row in item["failure_slices"]
    ]
    example_rows = [
        [
            item["method"],
            row["slice"],
            row["query_variant"],
            row["query_id"],
            row["candidate_id"],
            row["metric"],
            "undefined" if row["value"] is None else f"{row['value']:.6f}",
            f"{row['distance']:.6f}",
        ]
        for item in methods
        for row in item["examples"]
    ]
    canvas_rows = [
        [
            item["method"],
            row["query_variant"],
            row["queries"],
            f"{row['ndcg_at_10']:.9f}",
            f"{row['ndcg_change_from_clean']:.9f}",
            f"{row['mean_top10_overlap']:.6f}",
        ]
        for item in methods
        for row in item["canvas"]
    ]
    cost_rows = [
        [
            item["method"],
            source,
            f"{item['cost']['parameters']:,}",
            f"{item['cost']['checkpoint_bytes']:,}",
            f"{item['cost']['embedding_bytes'][source]:,}",
            f"{item['cost']['index_bytes'][source]:,}",
            f"{item['cost']['per_source_index_cost'][source]['build_seconds']:.3f}",
            f"{item['cost']['per_source_index_cost'][source]['peak_rss_bytes']:,}",
            item["cost"]["measurement_route"],
        ]
        for item in methods
        for source in ("teacher", "v1")
    ]
    timing_rows = [
        [
            item["method"],
            _direction(row),
            row["metric"],
            row["percentile"],
            f"{row['value_seconds']:.9f}",
        ]
        for item in methods
        for row in item["cost"]["timing_summary"]
    ]
    fold_rows = [
        [
            summary["method"],
            f"Fold {fold['fold']}",
            f"{fold['score']:.9f}",
            f"{fold['coverage'] * 100:.3f}%",
            fold["scorable_query_count"],
            fold["total_query_count"],
        ]
        for summary in stability
        for fold in summary["folds"]
    ]
    spread_rows = [
        [
            summary["method"],
            f"{summary['mean']:.16f}",
            f"{summary['sample_standard_deviation']:.16f}",
        ]
        for summary in stability
    ]
    gallery_rows = [
        [
            {
                "teacher": "Teacher-only",
                "v1": "V1-only",
                "two_view": "Two-view",
            }[item["policy"]],
            item["query_normalization_source"],
            f"{item['quality_at_10']:.9f}",
            f"{item['p95_end_to_end_seconds']:.9f}",
            f"{item['index_bytes']:,}",
            "selected"
            if item["policy"] == gallery["final_policy"]["policy"]
            else "rejected",
        ]
        for item in gallery["policies"]
    ]
    attempt_rows = [
        [
            row["run_id"],
            row["method"],
            row["status"],
            row["error_type"],
            row["error_message"],
        ]
        for row in data["attempts"]
    ]
    source_rows = [
        [row["path"], row["sha256"]] for row in data["source_artifacts"]
    ]
    return f"""
<hr class="rule">
<section id="final-freeze" aria-labelledby="final-freeze-h">
  <p class="eyebrow">Final development-only evidence freeze</p>
  <h2 id="final-freeze-h">Scratch R5 + teacher-only: final learned decision</h2>
  <p class="lede">This section is generated only after the compact final bundle passes
  <code>fashion.task4.final_freeze.validate_final_comparison_bundle</code>. The producer first
  re-opened the six candidate packages and ten stability packages through the public strict
  Task 4 validators. The older HOG, HSV-edge, fusion, links, figures, costs, and explanations
  remain below unchanged as comparison context.</p>
  <div class="note good"><b>Final decision.</b> Scratch {html.escape(decision['method'])} wins.
  Use the {html.escape(decision['gallery_policy'])}-only gallery, exact cosine distance, and
  product-level minimum-distance collapse. B1 stays pretrained and comparison-only.</div>

  <h3>Candidate ladder and factor changes</h3>
  <p>“What changed from the prior candidate” is explicit here. R5 is an independent breadth
  branch, not another VICReg tweak.</p>
  {_table(
      [
          "Method", "What changed from the prior candidate", "Eligibility",
          "Development nDCG@10", "Cross-source nDCG@10", "Source ratio",
      ],
      method_rows,
  )}

  <h3>Protocol A — graded article type and colour relevance</h3>
  <p>Units are mean per-query or class-macro scores at K=10. Undefined queries are excluded
  with visible coverage in the strict source package.</p>
  {_table(
      ["Method", "Direction", "Metric", "Aggregation", "Value", "Queries", "Classes"],
      protocol_a_rows,
  )}

  <h3>Protocol B — product-family recovery</h3>
  <p>Recall@10, Hit Rate@10, Precision@10, and coverage are reported separately. Protocol B
  supports diagnosis; it does not replace the frozen Protocol A winner score.</p>
  {_table(
      ["Method", "Direction", "Metric", "Value", "Scored or total queries"],
      protocol_b_rows,
  )}

  <h3>Failure slices</h3>
  <p>These include grayscale, rare article type, rare type/colour, unusual geometry,
  family-unavailable, and weak-family cases. Teacher-derived slice labels remain a proxy
  even for V1 queries.</p>
  {_table(
      ["Method", "Direction", "Slice", "Metric", "Value", "Coverage", "Caveat"],
      slice_rows,
  )}

  <h3>Qualitative examples</h3>
  <p>Deterministically selected success and failure queries are represented by their first
  returned product. IDs point back to the validated examples artifacts; no image is opened
  by this report.</p>
  {_table(
      [
          "Method", "Slice", "Canvas", "Query ID", "Top result ID",
          "Metric", "Query value", "Cosine distance",
      ],
      example_rows,
  )}

  <h3>Clean, wide, and tall canvas robustness</h3>
  {_table(
      [
          "Method", "Canvas", "Queries", "nDCG@10",
          "Change from clean", "Mean Top-10 overlap",
      ],
      canvas_rows,
  )}

  <h3>Parameter, checkpoint, embedding, index-build, and memory costs</h3>
  {_table(
      [
          "Method", "Source", "Parameter count", "Checkpoint bytes",
          "Embedding bytes", "Index bytes", "Index build time (s)",
          "Peak RSS (bytes)", "Measurement route",
      ],
      cost_rows,
  )}

  <h3>CPU batch-one timing details</h3>
  <p>Encoding p50, Encoding p95, Search p50, Search p95, End-to-end p50, and
  End-to-end p95 are seconds on the recorded one-thread machine.</p>
  {_table(
      ["Method", "Direction", "Stage", "Percentile", "Seconds"],
      timing_rows,
  )}

  <h3>Five-fold finalist stability</h3>
  {_table(
      ["Method", "Fold", "nDCG@10", "Coverage", "Scorable", "Total"],
      fold_rows,
  )}
  {_table(["Method", "Five-fold mean", "Sample SD"], spread_rows)}
  <p><b>Pooled spread:</b> {decision['pooled_spread']:.16f}.
  <b>Mean gap:</b> {decision['mean_gap']:.16f}. The gap is larger than the pooled spread.</p>

  <h3>Teacher/V1/two-view gallery comparison</h3>
  {_table(
      [
          "Policy", "Query normalization", "Quality nDCG@10",
          "End-to-end p95 (s)", "Index bytes", "Decision",
      ],
      gallery_rows,
  )}
  <p>The original R5 <code>cost.json</code> V1 value is a
  <b>pre-study cost assumption</b>. It is not the final policy. The canonical post-study
  decision is teacher-only.</p>

  <h3>Failures and retries remain visible</h3>
  {_table(
      ["Run ID", "Method", "Status", "Error type", "Message"],
      attempt_rows,
  )}
  <p>The gallery worker wrote valid artifacts and then did not exit. Its native shutdown
  cause remains unknown. Bounded supervision now prevents false command success.</p>

  <h3>Limits and sealed boundary</h3>
  <ul>
    <li>Metadata relevance is a proxy for human visual similarity.</li>
    <li>V1 is another view of the same products, not independent data.</li>
    <li>R5 remains sensitive to large white canvases.</li>
    <li>Timing and peak RSS are machine- and route-specific.</li>
  </ul>
  <div class="note good"><b>Holdout remains sealed.</b> Only
  <code>data/processed/splits.csv</code> defines membership. Holdout, quarantine, and
  official teacher-test images were not opened. Notebook 06 may evaluate holdout once
  after the exact frozen R5 refit is registered.</div>

  <h3>Strict source hashes</h3>
  {_table(["Source artifact", "SHA-256"], source_rows)}
</section>
"""


def render_report(*, root: Path = ROOT) -> str:
    """Render the preserved broad report plus the strict final appendix."""

    data = _load(root)
    template_path = (root / TEMPLATE_RELATIVE_PATH).resolve()
    try:
        template_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("report template must stay inside the repository") from error
    template = template_path.read_text(encoding="utf-8")
    if template.count("</main>") != 1:
        raise ValueError("broad report template must contain exactly one main close")
    required_broad_content = (
        "HOG + HSV-edge fusion",
        "../figures/task4/hog_fusion/hog_fusion_examples.png",
        "../figures/task4/hog/hog_examples.png",
        "../figures/task4/baseline_examples.png",
    )
    if any(value not in template for value in required_broad_content):
        raise ValueError("broad report template lost required HOG/fusion content")
    for name in REPORT_FIGURE_NAMES:
        citation = f"{FIGURE_CITATION_PREFIX}{name}"
        if template.count(f'src="{citation}"') != 1:
            raise ValueError(
                f"broad report template must display report figure {name} exactly once"
            )

    summaries = {item["method"]: item for item in data["stability"]}
    r5 = summaries["R5"]
    r3 = summaries["R3"]
    r5_folds = {int(item["fold"]): item for item in r5["folds"]}
    galleries = {
        item["policy"]: item for item in data["gallery"]["policies"]
    }
    replacements = {
        "{{FINAL_METHOD}}": str(data["decision"]["method"]),
        "{{FINAL_GALLERY}}": str(data["decision"]["gallery_policy"]),
        "{{R5_MEAN}}": f"{r5['mean']:.9f}",
        "{{R5_SD}}": f"{r5['sample_standard_deviation']:.9f}",
        "{{R3_MEAN}}": f"{r3['mean']:.9f}",
        "{{MEAN_GAP}}": f"{data['decision']['mean_gap']:.9f}",
        "{{POOLED_SPREAD}}": f"{data['decision']['pooled_spread']:.9f}",
        "{{GALLERY_TEACHER_QUALITY}}": (
            f"{galleries['teacher']['quality_at_10']:.9f}"
        ),
        "{{GALLERY_TEACHER_P95}}": (
            f"{galleries['teacher']['p95_end_to_end_seconds']:.9f}"
        ),
        "{{GALLERY_TEACHER_BYTES}}": f"{galleries['teacher']['index_bytes']:,}",
        "{{GALLERY_V1_QUALITY}}": f"{galleries['v1']['quality_at_10']:.9f}",
        "{{GALLERY_V1_P95}}": (
            f"{galleries['v1']['p95_end_to_end_seconds']:.9f}"
        ),
        "{{GALLERY_V1_BYTES}}": f"{galleries['v1']['index_bytes']:,}",
        "{{GALLERY_TWO_VIEW_QUALITY}}": (
            f"{galleries['two_view']['quality_at_10']:.9f}"
        ),
        "{{GALLERY_TWO_VIEW_P95}}": (
            f"{galleries['two_view']['p95_end_to_end_seconds']:.9f}"
        ),
        "{{GALLERY_TWO_VIEW_BYTES}}": (
            f"{galleries['two_view']['index_bytes']:,}"
        ),
    }
    for name in REPORT_FIGURE_NAMES:
        replacements[_figure_caption_token(name)] = FIGURE_CAPTIONS[name]
    for fold in range(5):
        score = float(r5_folds[fold]["score"])
        replacements[f"{{{{R5_FOLD_{fold}}}}}"] = f"{score:.5f}"
        replacements[f"{{{{R5_FOLD_{fold}_BAR}}}}"] = (
            f"{max(0.0, min(100.0, (score - 0.25) / 0.30 * 100.0)):.2f}"
        )
    for token, value in replacements.items():
        if token not in template:
            raise ValueError(f"broad report template is missing required token {token}")
        template = template.replace(token, html.escape(value))
    if re.search(r"\{\{[A-Z0-9_]+\}\}", template):
        raise ValueError("broad report template contains an unresolved final token")

    rendered = template.replace("</main>", f"{_final_section(data)}\n</main>")
    visible_text = re.sub(
        r"\s+",
        " ",
        re.sub(r"<[^>]+>", " ", html.unescape(rendered)),
    ).lower()
    stale_claims = (
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
    if any(claim in visible_text for claim in stale_claims):
        raise ValueError("rendered final report contains a stale progress claim")
    return rendered


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_report(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
