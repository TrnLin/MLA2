# Fashion Product Images source provenance

## Source

- Dataset: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data)
- Owner: Param Aggarwal
- Kaggle reference: `paramaggarwal/fashion-product-images-dataset`
- Kaggle dataset ID: `139630`
- Version: `1`, “Initial release”
- Released/last updated: `2019-03-14T18:57:43.307Z`
- Kaggle-reported dataset file bytes: `15,711,279,132`
- Dataset licence: MIT
- Platform terms: [Kaggle Terms of Use](https://www.kaggle.com/terms)
- Metadata checked: 2026-08-21 through Kaggle's public dataset API
- Local acquisition/catalogue date: 2026-08-21. The earlier download time was not separately logged.

The project uses only `images/<id>.jpg` and, when present, the safe filename IDs in `images.csv`.
It never opens or uses `styles.csv` or `styles/<id>.json`. The teacher 60x80 image and Kaggle image
are treated as resolution variants of the same product, not independent observations.

## Exact version and archive limitation

The public API reports that version 1 is the only and current version. Its download endpoint returns
an `archive.zip` bundle. The original downloaded archive is not present in this checkout, so its
exact archive byte size and SHA-256 cannot be recovered without downloading another large bundle.
No archive digest is guessed. Kaggle reports `15,711,279,132` total unpacked dataset bytes. The
runtime keeps one canonical image tree only. Its byte count and digest are recorded in the compact
catalogue evidence.

The unpacked data is identified instead by:

- Full canonical image catalogue SHA-256: recorded in
  `docs/data-provenance/fashion-product-images-catalogue.json`
- Every canonical image's SHA-256: recorded in
  `data/processed/high_resolution/image_catalogue.csv.gz`

## Acquisition and evaluator access

The Kaggle CLI has no dataset-version download flag. Do not append `/1` to the dataset name. This
verified official API URL pins `datasetVersionNumber=1` and redirects to archive version `329006`:

```bash
mkdir -p data/downloads tmp/fashion-product-images-v1 data/fashion-dataset
curl --fail --location \
  'https://www.kaggle.com/api/v1/datasets/download/paramaggarwal/fashion-product-images-dataset?datasetVersionNumber=1' \
  --output data/downloads/fashion-product-images-dataset-v1.zip
sha256sum data/downloads/fashion-product-images-dataset-v1.zip
unzip -q data/downloads/fashion-product-images-dataset-v1.zip \
  'fashion-dataset/fashion-dataset/images/*' \
  'fashion-dataset/fashion-dataset/images.csv' \
  -d tmp/fashion-product-images-v1
mv tmp/fashion-product-images-v1/fashion-dataset/fashion-dataset/images \
  data/fashion-dataset/images
mv tmp/fashion-product-images-v1/fashion-dataset/fashion-dataset/images.csv \
  data/fashion-dataset/images.csv
```

Keep the printed archive digest in local acquisition records. Do not commit the archive or unpacked
dataset.

The archive is nested, but the runtime canonical layout is one flat image tree:

```text
data/fashion-dataset/
├── images.csv       # optional safe filename/link table
└── images/
```

The current checkout also contains a nested image copy. The catalogue compares image names and
sizes, device-plus-inode identity, and deterministic sampled SHA-256 values. It rejects any found
mismatch and scans `data/fashion-dataset/images` once. A clean evaluator needs only the single flat
tree above and never needs to create the duplicate extraction.
