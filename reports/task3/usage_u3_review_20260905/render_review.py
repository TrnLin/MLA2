"""Render the review notes for visual inspection."""

from pathlib import Path

import mistune

HERE = Path(__file__).resolve().parent
STYLE = """
body { max-width: 1080px; margin: 36px auto; padding: 0 28px 60px;
       font: 16px/1.55 Arial, sans-serif; color: #172b36; }
h1 { font-size: 30px; color: #205968; }
h2 { margin-top: 38px; border-bottom: 1px solid #ccd8dd; padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 22px 0; }
th,td { padding: 9px; text-align: left; border: 1px solid #ccd8dd; }
th { background: #e6f0f0; } tr:nth-child(even) { background: #f6f8fa; }
img { display: block; max-width: 100%; height: auto; margin: 26px auto; }
code { overflow-wrap: anywhere; font-size: .87em; }
pre { padding: 15px; background: #f1f4f6; white-space: pre-wrap; }
a { color: #146476; }
@media print { img, tr { break-inside: avoid; } }
"""


def main():
    render = mistune.create_markdown(plugins=["table"])
    for name in ("README", "REVIEW"):
        content = render((HERE / f"{name}.md").read_text())
        content = content.replace('href="REVIEW.md"', 'href="REVIEW.html"')
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Usage U3 review</title><style>{STYLE}</style><main>{content}</main></html>"
        )
        (HERE / f"{name}.html").write_text(page)
    print("Rendered README.html and REVIEW.html")


if __name__ == "__main__":
    main()
