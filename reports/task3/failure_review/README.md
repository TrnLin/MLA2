# Task 3 failure review

This is a small, offline report for the latest Gender and Usage failure review.

## Open it

Open `index.html` in a browser. No server, network, package install, or build step is needed.

If a browser blocks local JavaScript, run this from the repository root:

```bash
./.venv/bin/python -m http.server 4173 --directory reports/task3/failure_review
```

Then open `http://127.0.0.1:4173/`.

## Files

- `index.html` — report structure and written judgement
- `styles.css` — laptop, narrow-screen, and print layout
- `data.js` — compact, human-readable evidence used by the charts
- `overfit-data.js` — all-model train, validation, gap, analysis, and mean learning curves
- `app.js` — offline filters, charts, matrices, and gallery rendering
- `assets/evidence/` — four existing diagnostic figures copied for this report
- `assets/gallery/` — ten official 60 × 80 teacher images copied for error review
- `evidence/gpu_screen_2/` — 32 compact Micro-Swin proof files pulled from Drive, plus a checksum manifest
- `evidence/usage_e1_drive/` — five missing Usage E1 histories pulled from Drive, plus a checksum manifest
- `evidence/all_model_overfit_summary.csv` — exact 23-model train/validation ledger used by the overview
- `evidence/all_model_curve_means.csv` — 690 averaged epoch rows used by the dropdown graphs

## Main evidence

- `notebooks/04_task3_gender_usage.ipynb`
- `notebooks/task3_training/clean_slate_screen_1.ipynb`
- `notebooks/task3_training/micro_swin_clean_slate_screen_2.ipynb`
- `results/evidence/task3/results/runs.csv`
- `results/task3/experiments/task3_clean_slate_screen_1/`
- `results/evidence/task3/experiments/t3_gender_e8_early_stopping/`
- `results/evidence/task3/experiments/t3_gender_e9_semantic_filter/`
- `results/evidence/task3/experiments/t3_usage_e8_translation/`
- [Drive folder: `task3_clean_slate_micro_swin_screen_2`](https://drive.google.com/drive/folders/1PbluXQStiliwYZJPb-HXr8Oa6Akq1Kiu)
- `evidence/gpu_screen_2/manifest.json` maps every local proof file to its Drive file ID, byte size, and SHA-256 checksum

## Where overfitting is shown

- **What we tried to reduce overfitting** lists the six direct controls and says what each one achieved.
- **Overfitting across the search** shows matched mean train and validation scores for all 23 experiments.
- Its separate dropdown selects any Gender or Usage model. The result card, diagnosis, written analysis, macro-F1 curve, and loss curve update together.
- Gender has a gap in all 12 models. Usage has a generalisation problem across the chain, but the failure is mixed: most learned CNN/Swin runs overfit; TinyConvNeXt and the type cascade mainly underfit; E1, E8, and E9 show both problems.
- Solid train dots are separate finished-model train checks. Hollow train dots are the last online epoch because four older runs did not save that later check.
- Pooled OOF macro-F1 is the reported result. The gap is computed separately from arithmetic fold means; four older train values are last online epochs rather than finished-model evaluations.

## Important limits

- The four latest screens use only folds 0 and 4 and one seed.
- Gender gates used E9. The report also shows true E8 on the same IDs because E8 is the working eligible reference.
- Usage gates used E8 and preserve literal `NA` as a real class.
- Micro-Swin checkpoints, environment records, and six large row-level prediction dumps were not copied into this report. They are not needed for the displayed proof.
- The local registry does not contain the four Micro-Swin run rows.
- The Usage E1 aggregate file is not local. Its five histories were recovered from Google Drive; its pooled score still comes from the canonical notebook ledger.
- The blind human-observability review is not complete.
- No protected-holdout or three-seed promotion result is claimed.

## Asset provenance

Every gallery image came from `data/raw/teacher/train/images_train/<id>.jpg`. The report does not load images or scripts from the network.

## Checks completed

- All 18 locally available historical aggregates match the score ledger.
- All four latest aggregate, fold, gate, class, confusion, calibration, gap, time, memory, and parameter values match their source files.
- The compact Drive import contains 32 files and 63,879 bytes. All 18 JSON files parse, all 14 CSV files have consistent columns, and every imported file byte-matches the read-only pull.
- The five recovered Usage E1 histories have 30 epochs each and are bound to their Drive file IDs, sizes, and SHA-256 hashes in `evidence/usage_e1_drive/manifest.json`.
- The all-model overview contains 23 rows. Twenty-one models have epoch curves, totalling 690 averaged points; HOG–SVM and the type cascade correctly show final checks only.
- Development class counts and the Usage usual/exception slice were rebuilt from `data/processed/splits.csv`.
- All ten gallery files are 60 × 80, byte-match their teacher source, and match their saved predictions.
- Filters, metric switches, exact-data tables, matrices, and expandable case notes were exercised in a browser.
- Laptop and 390-pixel layouts have no page-wide horizontal overflow. Wide charts scroll inside their own boxes on narrow screens.
- The print control works and the browser parsed the print stylesheet. The automated browser does not expose a visual print-preview render.
- The page reported no browser warnings or errors and contains no external resource links.
