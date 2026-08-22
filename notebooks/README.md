# Notebooks

Use notebooks to tell the project story and show results.

`00_eda.ipynb` is the one official EDA workflow. It is intentionally
self-contained: a fresh-kernel **Run All** validates and reuses the unchanged shared
data contract, performs every EDA calculation, saves all report evidence and figures,
and ends with the modelling decisions. It follows the ML EDA lifecycle: scope,
provenance, data quality, leakage controls, target analysis, image-feature analysis,
evaluation design, and modelling recommendations. Each analysis block presents its
purpose, focused evidence, finding, and modelling consequence. Code stays in the
notebook for auditability but is collapsed by default. Small technical cells sit
beside the analysis they support instead of forming a helper-code wall at the start.
It must not import a project EDA helper module.

Cached validation is the default. It fully hashes the lean protected-safe prepared
pack and checks raw path/size inventory plus a fixed content sample. Use
`FASHION_EDA_MODE=full` for the slow forensic rebuild after raw inputs change.

Keep reusable loading, split, image-variant, training, retrieval-metric, and
evaluation contracts in `src/fashion/`. Both image variants share one product ID
and never count as independent evidence. Number later notebooks in reading order.
