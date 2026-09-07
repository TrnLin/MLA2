# Final prepared-image visual QA

Reviewed 2026-09-06. Input: `reports/task3/rare_replacement_20260906/prepared_candidates.csv`, 130 rows. Read the actual image in each row's **path** column, not its original source photo.

## Result

**All 130 images visually inspected. No preparation blockers.** All files decode, all are 60×80 pixels, and all prepared-file SHA-256 values match the manifest. No blank images, obvious broken transparency, accidental source-ID mismatch, or product crop that makes the garment/accessory unrecognizable was seen.

This is a visual preparation check, not a statement that each image is equally useful for learning the usage label. Product wording, brand lettering, small dial markings, and fine print are naturally unreadable at this resolution; the useful signal is the product outline, colour, texture, and composition.

## Inspection method and coverage

Generated four contact sheets directly from the prepared PNG paths. Each tile shows the unchanged 60×80 image at native size alongside a 120×160 nearest-neighbour enlargement, so no extra detail is invented by smooth enlargement. Opened and inspected every sheet with the image viewer.

| Sheet | Manifest indices, zero-based | Images inspected |
|---|---|---:|
| prepared_visual_contact_1.png | 000–034 | 35 |
| prepared_visual_contact_2.png | 035–069 | 35 |
| prepared_visual_contact_3.png | 070–104 | 35 |
| prepared_visual_contact_4.png | 105–129 | 25 |

`prepared_visual_index.csv` maps every tile index to its exact asset ID, source ID, title and prepared path. All contact sheets and this index are in the same `smart_casual` report directory.

Home cushion-cover sets remain recognizable; some fine patterns disappear, as expected. Party clutches and perfume bottles retain their outlines. Party dresses retain the garment from neckline through hem; missing lower legs are normal source framing, not a garment crop. Smart Casual ties, wallets, watches, full men's shirts, women's blouses and women's sandals are recognizable. Women's blouses have complete hems and sleeves on plain backgrounds. Travel handbags and the six selected camera bags remain complete.

## Non-blocking weak images

These images remain usable, but have visibly less signal than neighbouring items. No candidate or image was modified.

| Index | Exact asset ID | Item | Visual limitation |
|---:|---|---|---|
| 035 | replace_fd9d57023f9ab06116d7 | Fastrack Younique 6279SM01 | Small silver watch blends into its pale marble/staged background. Face and strap remain visible; background adds noise. |
| 058 | replace_fb73754b47e0ed73fc64 | OGL BRAVE Shell Cordovan Mid Wallet | Wide native source margins leave a small wallet in the 60×80 frame; outline visible but surface detail is weak. |
| 059 | replace_63a12c1a09d409ef5ceb | OGL BRAVE Shell Cordovan Short Wallet | Small isolated wallet footprint; recognizable silhouette, little remaining texture detail. |
| 125 | replace_b3f657475e04a6e53375 | WANDRD ROGUE 6L Sling | Bag occupies a small part of the prepared frame because the source already had large margins. Shape is clear, construction details are weak. |
| 126 | replace_92af14539f0ddf34daf8 | Billingham S4 Camera Bag | Smallest tan camera-bag footprint in the group. Body and strap remain visible, but buckles/pockets carry little detail. |

Secondary caution: index 062, `replace_60daf5f817890a89e441` (Minimalist Leather Wallet), retains an indoor background. The wallet itself is large and clear, so this is background/style noise rather than a scale or crop problem.

**Recommendation:** these do not block finalization. Keep the cautions in the audit; if later analyses show scale sensitivity, compare them with a consistently applied, documented foreground framing policy rather than silently introducing ad hoc cropping for a handful of examples.

## Limits

No new sources or images were acquired. No candidate, split, prepared image, or dataset was changed. This check does not replace duplicate/family leakage checks or source-label evidence review. Visual recognizability does not by itself prove a usage label.
