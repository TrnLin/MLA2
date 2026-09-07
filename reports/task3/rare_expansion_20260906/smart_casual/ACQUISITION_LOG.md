# Smart Casual expansion — 2026-09-06

Downloaded 158 real product images. Retained 157 SHA-256-unique images from
157 source product IDs, spanning 120 conservative source family groups.
One image was rejected after visual review because it shows only a trouser waist.
No image URL or SHA-256 among retained rows matches the previously admitted
external images in `data/processed/teacher_plus_rare_usage_20260906/added_images.csv`.
No teacher holdout images or labels were inspected. Central perceptual duplicate
and final label checks remain necessary.

## Sources and evidence

1. Amazon Berkeley Objects cached official metadata: scanned all 16 shards for
   smart casual, business casual, semi-formal, and office text. Office-only hits
   were not selected. Selected one main-image ID per image, retaining 79 images
   whose product text explicitly says semi-formal or business casual. Almost all
   are Symbol men's shoes, plus an Amazon Essentials mule and belt. The proposed
   Smart Casual label is an inference from that text. Exact wording is in the CSV.
   Original full matching records remain in `hits.json`.
2. Marks & Spencer public catalogue:
   https://www.marksandspencer.com/l/men/mens-smart-casual
   Retrieved all eight public pages, containing 359 displayed entries and 125
   distinct parent product IDs. Selected 79 native `Cut_Out` catalogue assets in
   clothing/shoe types, excluding multipacks; retained 78 after visual review.
   The retailer itself includes these products in “Men’s Smart-Casual Clothing”.
   This is explicit collection membership, not a claim of a per-item Usage field.
   Native data and HTML responses are saved in `ms_native_products.json` and
   `ms_page_1.html` through `ms_page_8.html`. Cut-out assets are product-only
   catalogue photographs. The rejected asset shows that source asset types can
   be wrong, so all four contact sheets were inspected.

The earlier indexed M&S long-sleeve smart-casual category returned HTTP 404;
its response is retained as `ms_page.html`. No access-control bypass was used.
The working public collection returned HTTP 200.

## Rights

ABO: CC BY 4.0 according to the official release license previously saved at
`reports/task3/rare_external_intake_20260906/sources/abo/raw/LICENSE-CC-BY-4.0.txt`.
Attribution: Amazon.com; Amazon Berkeley Objects dataset by the authors listed in
`reports/task3/rare_external_intake_20260906/sources/abo/ATTRIBUTION.md`.
Official source: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
ABO originals were downloaded unchanged.

M&S: retailer copyright. Public availability does not establish reuse permission.
No open license or permission was found or claimed. Each row explicitly records
`copyright_permission_not_established`; it must not be described as openly licensed.
Images were downloaded from the retailer's own image CDN at 768-pixel width.

## Checks and limitations

All 158 files decode. Retained rows have source URLs, evidence, native product IDs,
dimensions and hashes. Contact sheets 1–4 were visually inspected. They contain
clear isolated products, without collages, placeholder logos, or large promotional
text. Several shirts are folded but recognizable. No obvious identical image was
noticed, although similar shoe designs and clothing fits remain and should stay
in their family groups. Source family IDs fall back conservatively to normalized
English titles when ABO has no model number. Size/color variants are not asserted
to be independent model families. These candidates are not split or train-ready.

Reproduce downloads and final visual decisions in order:

```
./.venv/bin/python reports/task3/rare_expansion_20260906/smart_casual/acquire.py
./.venv/bin/python reports/task3/rare_expansion_20260906/smart_casual/finalize.py
```

No shared source code, splits, training registry, commits, or manifests were changed.
