# Task 3 assets and submission checklist

The assignment spec (Sections 3–5) asks for the report, saved final models,
prediction CSV, and all code with setup steps. The rubric rewards comparison
breadth, justified choices and clear limits. Failed runs therefore keep their
code, scores and useful failure evidence.

## What is kept

- All 40 companion notebooks are in `notebooks/task3_training/`. Their saved
  outputs stay with them. `moves.csv` maps every old name to its new name.
- All training code remains under `src/fashion/`. The current untracked data
  extension, replacement, worker-budget and Usage HOG decision modules are included.
- Reports live here in dated folders. Figures remain in `results/figures/task3/`.
  JSON receipts and CSV records retain their original bytes, run IDs and hashes.
  `path-moves.json` maps the old paths. The main notebook translates the old
  `reports/task3_` prefix to `reports/task3/` when reading a receipt.
- `asset-inventory.csv` records each original report file, its destination,
  size, source hash and whether it is public or kept locally.
- `local-assets.csv` records the ignored registry, saved-run evidence, model
  weights, prepared variants and reviewed external inputs. These are still
  needed for full reproduction; they are not disposable.

## Restore a local checkout

From the repository root, use the retained source checkout:

```bash
./.venv/bin/python scripts/restore_task3_assets.py --source /home/dinhquan/personal/academic/RMIT/Machine-Learning/MLA2-eda
```

The command copies only missing ignored assets and checks their saved hashes.
It stops on a differing local file. It never changes the source checkout.
Use `--verify-only` to check an already prepared checkout without copying.

The original teacher data is supplied separately. Reuse the existing
`MyDrive/MLA2/data/task3-data.zip`, whose paths begin with `data/raw/teacher/`,
or put the supplied image folders and CSVs at:

```text
data/raw/teacher/train/images_train/
data/raw/teacher/train/styles_train.csv
data/raw/teacher/test/images_test/
data/raw/teacher/test/styles_prediction.csv
```

The existing local teacher files are at
`/home/dinhquan/personal/academic/RMIT/Machine-Learning/ASM2/A2_FashionDataset/`.
Copy or link these four items to the paths above. Keep the teacher CSVs unchanged.
See the root README for the pinned environment setup.

For Colab, reuse the named data ZIPs in each notebook's setup cell.
The notebooks fetch code from `fashion-analysis-and-cleanup` on GitHub.
After this branch is merged, that branch or the merged revision can be used.
The expanded E8 and MixUp/SAM notebooks verify their old ZIPs, retain the
archive snapshot, and load code from a separate current Git checkout.
No code-only change requires another data ZIP upload.

## Final model assets

- Gender: the five `final_epoch.pt` files in
  `results/evidence/task3/gender_sam25_cv_20260906/`, plus each run's
  `config.json` and `normalization.json`. The exact run IDs and hashes are in
  [the Gender manifest](gender_sam25_cv_result_20260906/model_manifest.json).
- Usage: the original five teacher-only E1 SmallCNN fold checkpoints under
  `results/evidence/task3/baseline/usage/`, with their own configurations and
  normalization. Use the selected five-fold recipe in the integrated E1 decision;
  do not substitute a two-fold screen or an expanded-dataset checkpoint.
- Keep the canonical `data/processed/splits.csv`, `label_maps.json`, and the
  Gender name-label variant. Expanded experiments additionally need their
  explicit saved split version and reviewed image files. Never rebuild folds
  as part of cleanup.
- Keep `results/runs.csv` and the dated registry snapshots. The main report
  reads its pinned snapshot, so opening it does not add a training run.

## Before submission

- Include all four assignment task models and inference code in the private
  submission ZIP. Task 3 alone does not complete the whole assignment.
- Include the Task 3 fold weights, class maps and normalization files above.
  A public source checkout intentionally does not contain model weights or raw data.
- Include the final four-target CSV with exactly
  `id,gender,articleType,season,usage`. Separate Gender/Usage files are evidence,
  not the full submission.
- Write the final report within five text pages plus two appendix pages,
  single column, 11pt, with the group names and student IDs. Follow Section 5
  for the required submission filename.
- Provide the approved added data to the evaluator through the existing
  private data route. Do not upload data/model ZIPs to the public repository.
- Carry forward the recorded evaluation limits. A reused holdout or recovered
  teacher-test label check is not a fresh independent evaluation.

## Excluded copies

Generated HTML previews, notebook text dumps, logs, caches, raw catalogue
downloads, ZIP copies and bulky duplicate saved-run snapshots stay ignored.
Their source copies are intact and listed in the inventories. Required
summaries, receipts, all code and unique failure evidence remain available.
No source checkpoint, source dataset, Drive file or source notebook was deleted.
