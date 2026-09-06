# Camera Travel candidates

Eight real catalogue product images from six brands: Think Tank1, WANDRD1, Billingham3, Domke1, Tenba1, Manfrotto1. Each is a separate design, with one photo. All original files and the contact sheet were visually inspected. No images were generated and no training dataset was changed.

`candidates.csv` contains proposed_usage=Travel and audit product_type=Camera Bags, plus product-specific source text, product/image URLs, cached evidence paths, dimensions, SHA256 and visual notes. These are proposed Usage inferences from manufacturer text, not teacher labels or assigned articleType targets. Product text mentions travel/travellers/adventure travel; Think Tank's generic warranty paragraph was deliberately not used as evidence.

The intended teacher visual gaps are small shoulder camera bags/pouches: IDs35828 and39487. Teacher12348 is an erroneous folded-shirt image and is not a sourcing target. Larger messenger styles broaden coverage but do not precisely reproduce the tiny grey/black teacher camera pouches. Domke is small but a roll-top multi-purpose weather pouch; its label support and visual match are weaker than the compact camera shoulder bags. Billingham supplies three distinct designs with related styling, so source/style concentration remains. Keep those limits visible during central admission.

The Manfrotto original is a transparent cutout with generous empty margins. Composite on white and use normal foreground-aware preparation. Its first catalogue image included loose camera equipment, so the final candidate uses the closed bag-only front view. One Billingham PNG may also need alpha compositing. Tenba's386x386 image is above the60x80 target resolution. Originals are preserved unchanged.

`validation.json` confirms eight unique file hashes and no matches against saved original/prepared hashes from the existing split manifest. This is **not** a decoded/near-duplicate clearance; central teacher/new-pool image and family checks remain required. Retain source groups during fold assignment.

Three extra Lowepro candidates had good product-specific travel evidence, but their image CDN repeatedly returned HTTP405. Their HTML evidence and `download_errors.json` are retained; no failed candidate appears in candidates.csv. No attempt was made to evade access controls. Broad catalogue JSONs also contain unrelated products; only the eight candidate rows are proposed for intake.

Manufacturer photographs remain copyright-protected; permission for reuse is not established. This is recorded consistently with the other replacement intake rows.

Files: `candidates.csv`, `visual_qa.csv`, `contact_sheet.jpg`, `validation.json`, per-product HTML/JSON evidence and `download_errors.json`. Originals are under `data/external/rare_usage_replacement_20260906/candidates/camera_travel/`. Rebuild with acquire.py, then finalize_review.py; final review notes reflect the inspected snapshots and should be revisited if remote images change.
