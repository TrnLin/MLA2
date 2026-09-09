# Flipkart Party original-image intake

This folder records source candidates, not final training admission. Only exact singleton merchant Occasion `Party` is retained. The first source-listed image URL is used, with an HTTP-to-HTTPS retry only. No image label is inferred from its appearance. Merchant labels remain weaker evidence than checked target annotations.

Run `./.venv/bin/python reports/task3/rare_external_intake_20260906/sources/flipkart_party/acquire.py` to verify saved image hashes and decode them without network access. `--acquire` explicitly permits bounded acquisition: at most 186 metadata groups, 150 successful images, 150,000,000 new response-body bytes, 2 MiB per image, one worker. Previously verified source photo bytes are reused. Originals are not resized here.

## Source and rights

The [publisher API](https://www.kaggle.com/api/v1/datasets/view/PromptCloudHQ/flipkart-products) was checked again on 2026-09-06: HTTP 200, license `CC BY-SA 4.0`; the complete response is saved in `publisher_recheck.json`. [Publisher dataset](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products).

The [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/) was also checked. It permits sharing and adapting licensed material with attribution, a license link, and a note of changes. Shared adaptations use the same or a compatible license. The license gives no warranty of all necessary rights. The released CSV contains remote photo URLs; the publisher's authority over every remote photo is not established. This is an unresolved photo-rights scope question, not a finding that research use is prohibited or a new approval requirement. This intake makes no claim of training or redistribution clearance.

Metadata attribution: Flipkart Products, PromptCloud / PromptCloudHQ, version 1, via Kaggle, CC BY-SA 4.0. Changes: parsed source specifications, exact singleton Party selection, and metadata grouping; source photo bytes retained unchanged.

`candidates.csv` preserves product and metadata-group IDs, title, source category and specifications, exact source Occasion evidence, product URL, original first image URL, original SHA-256, dimensions, attribution, and license basis. `attempts.csv` records actual requests; `exclusions.csv` distinguishes access failures and groups left after the success cap. `contact_*.jpg` are visual-review aids only.

## Result

All 186 available groups were considered. 112 originals decoded, including 7 verified cache reuses. The 74 failures each returned HTTP 404 at the original URL and HTTP 403 on the scheme-only HTTPS retry. The 253 new URL attempts transferred 26,016,777 successful-image body bytes. Saved originals total 27,188,314 bytes and have 112 distinct SHA-256 values. No teacher-duplicate check was performed here.

All four contact sheets were visually inspected. `visual_qa.csv` preserves all 112 review decisions. 53 show models or outfit context. Index 54 (`BRAEDGQFWPFGVPNF`) is a clear three-model, three-color collage and is recommended for exclusion. Seven images mapped to Heels appear flat or low-wedge and two mapped top/tunic images look dress-like: these are review flags, not source-label changes. No gross nonfashion object was seen. A photographed shoe pair is treated as one product presentation. The final admission process should decide how to handle model context and category mismatch; none of these flags silently removes a candidate here.

The default cached verification re-decoded all 112 originals and checked their hashes and dimensions with zero new network requests. Ruff passes for both scripts.
