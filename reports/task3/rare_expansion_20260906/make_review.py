"""Create complete model-view contact sheets and a portable image/evidence browser."""

# ruff: noqa: E501 -- embedded HTML/CSS/JS strings retain browser formatting

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import base64
import html
import json
import math

from build_collection import CLASSES, DATA, OUT, ROOT, read_csv
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/open-sans/OpenSans-Regular.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def main():
    frame = read_csv(DATA / "manifest.csv")
    cards = []
    overview = Image.new("RGB", (1200, 940), "#f7f7f4")
    draw = ImageDraw.Draw(overview)
    draw.text((30, 22), "More images for four rare Usage classes", font=font(27), fill="#152c37")
    draw.text(
        (30, 65),
        "60 × 80 previews · labels supported by product text · NA excluded",
        font=font(17),
        fill="#47535a",
    )
    index_rows = []
    for class_index, (slug, label) in enumerate(CLASSES.items()):
        subset = frame.loc[frame.usage.eq(label)].reset_index(drop=True)
        rows = math.ceil(len(subset) / 12)
        sheet = Image.new("RGB", (1200, rows * 115 + 70), "#eeeeeb")
        sheet_draw = ImageDraw.Draw(sheet)
        sheet_draw.text(
            (20, 18), f"{label} — {len(subset)} prepared images", font=font(23), fill="#152c37"
        )
        for index, row in subset.iterrows():
            x, y = (index % 12) * 100, (index // 12) * 115 + 60
            with Image.open(resolve_task3_path(row.path, root=ROOT)) as im:
                sheet.paste(im, (x + 20, y))
            sheet_draw.text((x + 18, y + 83), f"{index + 1:03d}", font=font(13), fill="#152c37")
            index_rows.append(
                {
                    "usage": label,
                    "number": index + 1,
                    "asset_id": row.asset_id,
                    "external_id": row.external_id,
                    "product_name": row.product_name,
                    "path": row.path,
                }
            )
            thumb = base64.b64encode((resolve_task3_path(row.path, root=ROOT)).read_bytes()).decode()
            esc = html.escape
            text = f"{row.product_name} {label} {row.source_dataset}".casefold()
            cards.append(
                f'<article data-usage="{esc(label)}" data-search="{esc(text)}">'
                f'<img src="data:image/png;base64,{thumb}" alt="{esc(row.product_name)}">'
                f"<div><small>{esc(label)} · {index + 1:03d}</small><h2>{esc(row.product_name)}</h2>"
                f'<p>{esc(row.evidence_text)}</p><p class="source">{esc(row.source_dataset)}</p>'
                f'<a href="{esc(row.product_url)}" target="_blank" rel="noreferrer">Source product</a>'
                f' · <a href="{esc(row.image_url)}" target="_blank" rel="noreferrer">Source photo</a>'
                "</div></article>"
            )
        sheet.save(OUT / f"prepared_{slug}.png")
        oy = 110 + class_index * 200
        draw.text((30, oy), f"{label}  /  {len(subset)} images", font=font(22), fill="#152c37")
        # Deterministic spacing shows the range of sources, rather than only the first brand.
        chosen = subset.iloc[[round(i * (len(subset) - 1) / 7) for i in range(8)]]
        for pos, (_, row) in enumerate(chosen.iterrows()):
            with Image.open(resolve_task3_path(row.path, root=ROOT)) as im:
                enlarged = im.resize((90, 120), Image.Resampling.NEAREST)
            overview.paste(enlarged, (40 + pos * 145, oy + 38))
    overview.save(OUT / "preview.png")
    import pandas as pd

    pd.DataFrame(index_rows).to_csv(OUT / "prepared_contact_index.csv", index=False)
    counts = frame.usage.value_counts().to_dict()
    buttons = '<button class="active" data-label="All">All</button>' + "".join(
        f'<button data-label="{label}">{label} ({counts[label]})</button>'
        for label in CLASSES.values()
    )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rare Usage image collection</title><style>
body{margin:0;background:#f5f5f1;color:#172f38;font:16px system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:28px}
h1{font-size:30px;margin:0 0 8px}.intro{max-width:1000px;line-height:1.5}nav{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 12px}
button,input{font:inherit;padding:10px 14px;border:1px solid #9dafb4;border-radius:8px;background:white;color:#172f38}button{cursor:pointer}.active{background:#172f38;color:white}input{width:min(95%,550px)}#count{margin:16px 0;color:#53696f}
section{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}article{background:white;padding:18px;border-radius:10px;display:flex;gap:16px;align-items:start;border:1px solid #dde4e1}article[hidden]{display:none}
img{width:60px;height:80px;flex:none;border:1px solid #ddd}h2{font-size:16px;margin:4px 0 10px;line-height:1.35}p{font-size:14px;line-height:1.45;overflow-wrap:anywhere}small,.source{color:#51686b}a{color:#08616b;font-size:14px}footer{font-size:14px;line-height:1.6;margin:26px 0}
</style><main><h1>Rare Usage image collection</h1>
<div class="intro">COUNT images with product names, source text, and links. Labels are proposed from source wording or retailer collections. These images have no assigned training fold. NA is excluded.</div>
<nav>BUTTONS</nav><input id="search" placeholder="Find a product, brand, or class" aria-label="Search products"><p id="count"></p>
<section>CARDS</section><footer>Original files and 60 × 80 RGB copies are saved locally. Similar products share family groups in the CSV. Retailer images are not claimed to be openly licensed; see README.md and the manifest's rights fields. No model has been trained on this collection.</footer></main>
<script>let label='All';const input=document.querySelector('#search');function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!((label==='All'||a.dataset.usage===label)&&a.dataset.search.includes(input.value.toLowerCase()));if(!a.hidden)n++});document.querySelector('#count').textContent=n+' images shown'}document.querySelectorAll('button').forEach(b=>b.onclick=()=>{label=b.dataset.label;document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));filter()});input.oninput=filter;filter();</script></html>"""
    document = (
        document.replace("COUNT", str(len(frame)))
        .replace("BUTTONS", buttons)
        .replace("CARDS", "\n".join(cards))
    )
    (OUT / "gallery.html").write_text(document)
    (OUT / "review_artifacts.json").write_text(
        json.dumps(
            {
                "images": len(frame),
                "contact_sheets": [f"prepared_{slug}.png" for slug in CLASSES],
                "gallery_cards": len(cards),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Rendered {len(frame)} model-view thumbnails and evidence cards")


if __name__ == "__main__":
    main()
