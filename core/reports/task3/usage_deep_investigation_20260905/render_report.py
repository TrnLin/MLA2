"""Render the research report and an exact run-ID appendix from saved tables."""

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
from pathlib import Path

import mistune
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "pyproject.toml").is_file())


def main():
    runs = pd.read_csv(OUT / "run_ledger.csv", keep_default_na=False)
    assert len(runs) == 53 and runs.run_id.nunique() == 53
    lines = [
        "# Exact Usage run IDs",
        "",
        "Generated from run_ledger.csv. All F1 values below are single-fold scores.",
        "",
    ]
    for name in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "S1", "S2", "U1", "U2"]:
        lines.extend([f"## {name}", "", "| Fold | F1 | Run ID |", "|---|---:|---|"])
        for row in runs.loc[runs.model.eq(name)].sort_values("fold").itertuples():
            lines.append(f"| {row.fold} | {row.macro_f1:.6f} | `{row.run_id}` |")
        lines.append("")
    (OUT / "RUNS.md").write_text("\n".join(lines))
    markdown = mistune.create_markdown(plugins=["table"])
    style = """
      :root { color-scheme: light; font-family: Arial, sans-serif; color: #172633; }
      body { max-width: 1080px; margin: 40px auto; padding: 0 30px 70px; }
      h1 { font-size: 32px; line-height: 1.2; color: #174f71; }
      h2 { margin-top: 46px; border-bottom: 2px solid #dbe5ec; padding-bottom: 9px; }
      h3 { margin-top: 28px; color: #285a76; }
      p, li { font-size: 16px; line-height: 1.55; }
      img { display: block; width: auto; max-width: 100%; height: auto; margin: 24px auto; }
      table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 20px 0; }
      th { background: #e9f0f5; text-align: left; }
      th, td { padding: 10px; border: 1px solid #d5dfe6; vertical-align: top; }
      tr:nth-child(even) { background: #f7f9fb; }
      code { font-size: 0.88em; overflow-wrap: anywhere; }
      pre { background: #edf2f5; padding: 18px; overflow-x: auto; }
      a { color: #155e8a; }
      @media (max-width: 650px) { body { padding: 0 16px; } table { font-size: 12px; } }
      @media print { body { margin: 0; } img, tr { break-inside: avoid; } }
    """
    for name in ["REPORT", "RUNS", "SOURCES", "README"]:
        content = markdown((OUT / f"{name}.md").read_text())
        # Keep local navigation in rendered HTML; original Markdown links stay intact.
        for target in ["REPORT", "RUNS", "SOURCES", "README"]:
            content = content.replace(f'href="{target}.md"', f'href="{target}.html"')
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Usage investigation — {name}</title><style>{style}</style>"
            f"</head><body><main>{content}</main></body></html>"
        )
        (OUT / f"{name}.html").write_text(page)
    hashes = json.loads((OUT / "input_hashes.json").read_text())
    changed = [
        name
        for name, expected in hashes.items()
        if hashlib.sha256((resolve_task3_path(name, root=ROOT)).read_bytes()).hexdigest() != expected
    ]
    assert not changed, changed
    print(f"Rendered 4 documents, verified 53 unique run IDs and {len(hashes)} source hashes.")


if __name__ == "__main__":
    main()
