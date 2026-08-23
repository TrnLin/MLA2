# Fashion Product Images source provenance

## Source

- Dataset: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data)
- Owner: Param Aggarwal
- Kaggle reference: `paramaggarwal/fashion-product-images-dataset`
- Dataset ID and version: `139630`, version `1`
- Licence: MIT
- Kaggle-reported unpacked bytes: `15,711,279,132`
- Local migration verified: 2026-08-23

This collection is optional Task 4 input. Shared preparation never scans it, parses
`images.csv`, or uses it for Tasks 1–3.

## Canonical local layout

```text
data/raw/external/fashion_product_images_v1/
├── images.csv
└── images/        # 44,441 JPEG files
```

The folder is outside Git. `images.csv` is retained only for provenance. It is not a
teacher-label source.

## Safe migration evidence

The former local extraction was inventoried before it was moved. The whole tree was
renamed first, so no image bytes were copied or deleted. Extra nested copies and style
metadata were then preserved under the ignored local folder
`data/raw/external/fashion_product_images_v1_legacy_extras/`.

Before and after, the combined local holdings were 177,778 files and 31,422,558,264
bytes. The canonical `images/` plus `images.csv` manifest contains 44,442 files. Its
full path/size/SHA-256 manifest digest is recorded in
`fashion-product-images-catalogue.json` and matched before and after the move.

Local proof files are in ignored `tmp/external-migration/`. They are machine-local audit
records, not submission files.

## Fresh acquisition

Download and unpack only the image tree and `images.csv`, then place them in the
canonical layout above. Do not commit the archive or extracted images. Record the archive
SHA-256 locally if a fresh download is made.

The public API URL for version 1 is:

```text
https://www.kaggle.com/api/v1/datasets/download/paramaggarwal/fashion-product-images-dataset?datasetVersionNumber=1
```
