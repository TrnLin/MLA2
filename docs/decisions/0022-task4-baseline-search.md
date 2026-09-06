# 0022 — Task 4 baseline search

- Status: Accepted
- Date: 2026-08-28

## Context

Later learned search models need one cheap, transparent comparison point. The
baseline must use the frozen development split, input contract, and retrieval
protocol without tuning against its results. It must also expose source shift,
runtime, storage, and known failure cases rather than report one quality score.

## Decision

Freeze the untrained `spatial-hsv-edge-4x4-v2` probe at the accepted `240×320`
RGB/LANCZOS letterbox input. It has 400 float32 features: equal normalized
spatial HSV and edge blocks over a `4×4` grid. No bins, weights, colour space,
grid, size, or distance were tuned for this baseline.

Use exact cosine distance and ADR 0019's fixed fold `1`. Evaluate all 6,556
eligible queries in each of four directions: teacher→teacher, V1→V1,
teacher→V1, and V1→teacher. Apply all exclusions before Top-K. When both image
variants represent one product, keep that product's minimum distance so one
product can occupy only one result slot. The primary score is mean per-query
linear nDCG@10.

Use the equal mean of teacher→teacher and V1→V1 as the same-source headline.
Freeze two hypotheses:

1. The same-source mean must beat the seeded random Protocol A nDCG@10 floor.
2. The equal cross-source mean must be at least 95% of the same-source mean.

This baseline is untrained. It has zero parameters and no checkpoint, so it
does not append a row to `results/runs.csv`.

For quality reproduction, use `chunk_size=256` and named absolute tolerance
`1e-5`. Default-thread V1→teacher exactly matched the stored preprocessing
evidence. Forcing one BLAS thread changed a few near-tie ranks; the largest
observed nDCG@10 drift was `9.709e-6`. The tolerance covers that measured
numerical drift without claiming bit-for-bit ranking identity.

Measure CPU batch-one timing with one thread, 100 warm-up queries, and every
6,556 query once per direction. Report encoding, search, and end-to-end p50 and
p95. The practical gates are p95 end-to-end below one second and each searchable
index below one GiB. Index build time and peak RSS are supporting,
machine-dependent evidence.

Freeze these failure slices before reading their scores:

- grayscale: image mode is `L`;
- rare article type: 1–9 eligible gallery positives;
- rare type-colour pair: fewer than 10 strict positives;
- unusual geometry: aspect ratio differs from `0.75`;
- family unavailable: zero eligible family positives;
- weak family: 1–4 eligible family positives.

## Why

The baseline clearly beats random. Its same-source mean is `0.48987767`, versus
the random nDCG@10 floor of `0.06235565`; the recorded gap is `0.42752202`.
Hypothesis 1 therefore passes.

The cross-source mean is `0.46171089`. Its ratio to the same-source mean is
`0.94250242`, below the frozen `0.95` threshold. Hypothesis 2 fails and is
rejected. Source appearance is therefore a measured weakness, not a solved
deployment issue.

The four direction nDCG@10 values are:

- teacher→teacher: `0.49521342`;
- V1→V1: `0.48454193`;
- teacher→V1: `0.46774604`;
- V1→teacher: `0.45567575`.

The predeclared slice counts are 44 grayscale, 40 rare-article-type, 833 rare
type-colour, 3,937 family-unavailable, and 1,659 weak-family queries. The two
observed unusual-geometry queries are retained. The tracked example canvas has
all eight selected rows. Its strongest known stress failure is the tall-canvas
example whose nDCG@10 change is `-1.0`. Across all V1→V1 queries, mean nDCG@10
falls from `0.48454193` clean to `0.10979303` wide and `0.17259584` tall.

Both indexes contain 26,217 rows × 400 features and occupy `42,156,936` bytes.
Teacher and V1 builds took `430.862066` and `1648.505369` seconds. Measured peak
RSS was `1,038,663,680` bytes for each child build.

End-to-end p50/p95 seconds were:

- teacher→teacher: `0.0844252695` / `0.08591582875`;
- V1→V1: `0.1314013365` / `0.1409705065`;
- teacher→V1: `0.0848337965` / `0.08672196125`;
- V1→teacher: `0.131808274` / `0.141162569`.

Both practical checks pass on the measured machine. These times are not
hardware-independent promises.

## Consequences

Every learned candidate is compared with this baseline under the same frozen
split and ranking rules. A learned method must be judged against its quality,
source robustness, failure slices, latency, index size, and qualitative
examples—not only its best score.

The baseline is not the final winner. Learned-model choice, deployed source,
final index, and ultimate judgement remain open. Metadata relevance remains a
proxy for human similarity. Large-background queries remain a known failure.
The labelled holdout stays sealed until Notebook 06 after all remaining choices
are frozen.

## Evidence

- `results/evidence/task4/baseline_summary.csv`
- `results/evidence/task4/baseline_query_metrics.csv`
- `results/evidence/task4/baseline_failure_slices.csv`
- `results/evidence/task4/baseline_timing.csv`
- `results/evidence/task4/baseline_cost.json`
- `results/evidence/task4/baseline_examples.csv`
- `results/figures/task4/baseline_examples.png`
- `scripts/task4/run_baseline.py`
