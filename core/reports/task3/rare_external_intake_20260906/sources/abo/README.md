# ABO rare Usage intake

Scanned all 16 official metadata shards (`0`–`f`), totaling 147,702 market
records and 145,615 unique ASINs. This is an actual scan, not an estimate.
The retained metadata and image index total 93,893,504 bytes.

Saved 66 original main-image files (42,246,437 bytes):

| Seller term | ASINs | Distinct SHA-256 files | Evidence |
|---|---:|---:|---|
| Smart Casual | 16 | 6 | Exact standalone seller bullet |
| Travel | 50 | 44 | 7 weak style-field phrases; 43 weaker title/bullet mentions |

All Smart Casual hits belong to two model families: find. Chelsea boots and
loafers. No Smart Casual shirts or watches were found by this literal-term
scan. Travel had 88 eligible bag ASINs before the 50-image cap; explicit style
fields and single-purpose wording were ranked first. The term search does
not claim to cover translated synonyms.

`candidates.csv` has native source type and seller evidence, plus main-image
URL, dimensions, SHA-256, relative original path, grouping, and visual flags.
`selected_native_records.json` retains complete native metadata for every
market record of each selected ASIN. All compressed original shards are in
`raw/`. `term_hits.json` and `exclusions.json` preserve the broader audit.

Source labels are candidates only. All 50 Travel rows are withheld from a
training-ready set: none has an exact standalone Travel occasion/style value.
The seven `weak_style_text` rows use longer product names such as Travel
Backpacks in the style field. Smart Casual standalone bullets use
`source_exact_style`. The Travel category includes mixed use;
`weak_title` is a conservative bucket for weaker title **or bullet** mentions,
not proof of an exclusive occasion. The known Casual Travel Backpack remains
in the term/exclusion audit when it falls outside the image cap.

All three contact sheets were visually inspected. Six files show only a logo,
and one has a promotional overlay. These seven must be excluded. The CSV
and `visual_qa.json` also flag an accessory composition, a multipack title
with one pictured bag, and a wheeled bag whose teacher article compatibility
needs review. The original files remain for audit.

There are 22 conservative source groups. Model families and obvious color
variants share a group; raw model/brand text is preserved. This grouping does
not replace image duplicate checks. It also does not establish independence
from teacher images or teacher Usage-label equivalence.

Official release license and attribution: see `ATTRIBUTION.md`,
`raw/README.md`, and `raw/LICENSE-CC-BY-4.0.txt`.

Rebuild from saved inputs without network:

```bash
./.venv/bin/python reports/task3/rare_external_intake_20260906/sources/abo/intake.py --offline
```

The offline rebuild and Ruff lint pass. Each original decoded and matched
the official image-index dimensions. No teacher files, splits, training code,
commits, or pushes were changed by this intake.
