# Main Task 3 report evidence

The main notebook contains 45 numbered sections covering the shared EDA,
the five later Task 3 audits, both targets' E1–E10 experiments, classical
and transformer screens, later Gender hypotheses, the frozen SAM25 recipe,
reserved holdout results, Usage expansion trials, and the accepted E1 Usage recipe.
All analysis and plotting code executes inside the notebook. There are no
external analysis helper imports or hidden helper functions.

## Files and sources

- `results/runs.csv`: exact 698,959-byte copy of the training registry fetched
  from [Drive](https://drive.google.com/file/d/1wk8EvVnhkSqCHqUgSqkQKV5Wft54SUqr/view).
  Source modified time: 2026-09-06 12:15:48 UTC. The live local and Drive
  registries were not changed.
- `evidence_lock.json`: source hashes and explicit run IDs for every comparison.
  The baseline is the completed 2026-08-30 five-fold pack, already used by the
  earlier notebook. Its older superseded fold-0 run is excluded by identity,
  not by score. Incomplete runs are not compared.
- `results/classical_runs.csv`: six completed classical-model rows copied
  from the local registry. These fill gaps in the earlier Drive snapshot;
  overlapping run IDs are not counted twice. The live registry is unchanged.
- `analysis_assets.json`: hashes for the additional audit tables, histories,
  corruption records, classical registry, and expansion split files.
- `earlier_investigation.ipynb`: exact copy of the local main notebook before
  this rewrite, including pre-existing edits and Usage outputs. This is a
  historical archive; its original relative paths assume the `notebooks/` folder.
- `main_report.html`: executed preview with links adjusted to its new location.
- `render.py`: runs all report cells using the project interpreter, exports HTML,
  and checks local links. Run from the project root with
  `./.venv/bin/python reports/task3/main_report_20260906/render.py`.

Visible notebook cells read saved results without importing training code
or requiring a GPU. They also recheck every development image's hash and
decode it. Registry tables sum fold
confusion counts before calculating macro-F1. Final IEEE scores and original
versus corrected label bases are explicitly separated. The original teacher
Usage registry is a separate comparison from the new expanded datasets.

The notebook generates its charts directly with matplotlib, including
contact sheets, transform examples, class/family support, fold checks,
target relationships, nuisance probes, class comparisons, learning curves,
corruption changes, fit gaps, error galleries, and holdout confusion counts.
Gender evaluation reads the three holdout-only CSV files in this directory.
Usage also reads saved official-image test comparisons pinned in
[the final E1 pack](../usage_final_e1_20260907/README.md).
The teacher supplied no correct labels for the official test set; the saved
Usage scores use recovered reference metadata. Final acceptance followed
evaluation review. Run All does not run new inference or look up test labels.

The generated report figures are in `results/figures/task3/main_analysis/`.

The gender model selection is recorded in
`docs/decisions/0020-task3-gender-sam25-final-model.md`. Usage is recorded in
`docs/decisions/0022-task3-usage-e1-final-model.md`. Report execution does not
train, change checkpoints, or write submission predictions.

## Verification

- `verification.json` records execution, metric, source, and visual checks.
- Run All verifies pinned evidence before computing its results.
- The archive remains the original pre-edit notebook. HTML links are checked
  during rendering.

The two pre-existing formatting differences elsewhere in
`tests/test_notebook_scaffolds.py` are preserved.
