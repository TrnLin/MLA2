# Usage E1 source contract and development references

`model_manifest.json` is the report view of the source contract read by the
E1 training runner. The runner accepts exactly the original source checksum
or the pinned report view checksum. Both define the same five folds,
configurations, normalization and model inputs. Saved training runs retain
their original recipe identity; report metadata does not change that recipe.

`evidence_lock.json` also supplies saved Usage development comparisons to
[Notebook 06](../../../notebooks/06_task3_part1_gender_usage.ipynb), including
rare-class additions and the later two-fold trials. These comparisons retain
their own folds, populations and scoring basis.

The current Usage model is the **single teacher-only E8 refit at epoch 30**.
See its [manifest](../usage_final_e8_refit_20260907/model_manifest.json) and
[selected evaluation](../usage_e8_refit_holdout_20260907/README.md).
