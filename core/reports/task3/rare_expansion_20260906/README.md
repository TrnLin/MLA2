# Rare Usage image collection — 6 September 2026

Collected **567 additional images** for the four requested rare classes. NA was excluded.
These are additional to the 120 images in the earlier expanded training set.
The new collection has not been added to a training split or used for a model run.

| Proposed Usage | Images | Linked family groups |
|---|---:|---:|
| Home | 120 | 118 |
| Party | 127 | 126 |
| Smart Casual | 156 | 81 |
| Travel | 164 | 115 |
| Total | 567 | 440 |

Family groups are conservative similarity/source groups, not verified independent designs.
Several colour and size variants remain. In particular, Smart Casual has fewer independent
families than image files.

## Open the files

- `rare_usage_567_prepared.zip` contains accepted 60 × 80 images, the manifest, evidence,
  gallery and audit notes. It omits high-resolution source files.
- `rare_usage_567_images.zip` also contains every accepted original and the opaque source
  copies. Both archives preserve repository-relative paths; the gallery inside either one
  opens without a local server.
- [Browse all images and their evidence](gallery.html). This HTML contains its own thumbnails
  and works offline. The source-product and source-photo links require the internet.
- [Small visual preview](preview.png).
- [Accepted manifest](accepted_manifest.csv): product names, descriptions, proposed Usage,
  source evidence, product and image URLs, rights notes, image paths and hashes.
- [Counts](class_counts.csv), [audit summary](summary.json), and
  [two duplicate pairs](internal_near_duplicate_matches.csv).
- [Home](prepared_home.png), [Party](prepared_party.png),
  [Smart Casual](prepared_smart_casual.png), and [Travel](prepared_travel.png) show every final
  60 × 80 image. Numbers map to [prepared_contact_index.csv](prepared_contact_index.csv).

The collection's main manifest is
`data/external/rare_usage_expansion_20260906/manifest.csv`.
Read its `path` column to load the accepted RGB PNG files; do not glob candidate folders,
which also retain rejected source files for the audit.
`source_original_path` names the unchanged downloaded file. `original_path` names the
opaque source used for processing; it differs for images with transparency.

## Label rule

The user allowed a proposed Usage when the product name or description supports that purpose,
even without an exact source label. Every row preserves this evidence rather than presenting
the proposed label as a teacher annotation. `explicit_source_label` includes retailer
collection membership/tags; it does not claim a native nine-class Usage field.

Home contains cushion covers and decorative throw pillows, matching the kind of product in
the teacher's development Home example. Party contains dresses supported by Party collections
and occasion wording/tags. Smart Casual combines retailer Smart Casual clothing with shoes
described as semi-formal or business casual. Travel contains travel, overnight, hiking and
weekend bags/backpacks, plus some luggage. A product may suit more than one real-life purpose.

## Checks completed

Each retained source image was decoded. Original contact sheets were visually inspected,
and all final 60 × 80 contact sheets were inspected. Home rejected 60 unsuitable candidates;
Party rejected 34; Smart Casual rejected one; Travel rejected 15.

All 569 visually selected candidates were checked against **44,441 teacher images** across
all roles and the **120 earlier admitted external images**. This check reads image and
identity fields, not held-out Usage labels. It verifies current reference bytes, then checks
file/pixel identity and near matches across original, resized and foreground-cropped views.
No existing-image overlap was accepted by the frozen pixel/hash rule. This is a measured
duplicate check, not a guarantee against every possible edited copy.

The new collection also received a within-collection check, including products already
sharing a source family. It found two near-duplicate pairs; one Smart Casual and one Travel
image were removed. Exact prepared pixel hashes are unique. Similar product families remain
linked to support later fold assignment.

Prepared files use the shared 60 × 80 letterbox transform, with EXIF orientation, RGB colour,
LANCZOS downscaling and white padding. The 74 transparent sources were first composited onto
white, so transparent backgrounds do not turn black. Prepared image size, mode and hashes
were verified. No training statistics were fitted.

## Sources and attribution

Home: Amazon Berkeley Objects (ABO), 120 images. Smart Casual: ABO, 78, and Marks & Spencer,
78. Travel: ABO, 46; Cotopaxi, 19; Dakine, 13; Db, 52; Eagle Creek, 27; Tom Bihn, 3;
Topo Designs, 4. Party: Beginning Boutique, 60; MESHKI, 48; Red Dress, 19.
Travel reuses some previously downloaded but withheld ABO source files; these were never
in the earlier 120 admitted images. They are newly eligible under the user's broader rule.

ABO image and product data: **Amazon.com**. Dataset contributors: Matthieu Guillaumin,
Thomas Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi (Brian) Zhang,
Tomas Yago-Vicente, Jasmine Collins, Shubham Goel and Jitendra Malik.
The [official release](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)
and its downloaded licence declare CC BY 4.0. The AWS registry and paper instead state
CC BY-NC 4.0. The conflict is recorded; this collection does not assert unrestricted
commercial rights. The actual licence and release attribution are included under `licenses/`.
No endorsement is implied.

Other images come from the named retailers' public catalogues and image servers.
Retailer/photographer copyright remains applicable. No open reuse or redistribution licence
was established for those photos. Public availability is not a licence. Per-image source
links and rights notes are retained. This is a local academic research collection and has
not been published or uploaded to a public dataset host.

See the source acquisition logs in `party/`, `smart_casual/`, and `travel/` and the Home
visual-selection file for the source-specific decisions and cached metadata.

## Limits and next use

Extra images do not prove that model results will improve. Home is almost entirely from one
dataset; Party is adult women's dresses, often photographed on models; Smart Casual includes
many similar shoes; Travel contains repeated bag designs in different sizes/colours and some
luggage beyond the teacher's main product types. These differences can become shortcuts for
a model. Proposed occasion labels can also be noisy.

Every row has `partition=unassigned`. A later integration must create an explicit new dataset
version, keep linked families together, preserve all teacher folds and the search gallery,
and keep the same nine-class Usage map. The existing canonical and expanded splits are
unchanged. No training run or commit was made for this collection.

For an offline audit rebuild, run the class audits, the internal comparison, finalisation and
review generation from the repository root:

```bash
./.venv/bin/python reports/task3/rare_expansion_20260906/build_collection.py home
./.venv/bin/python reports/task3/rare_expansion_20260906/build_collection.py party
./.venv/bin/python reports/task3/rare_expansion_20260906/build_collection.py smart_casual
./.venv/bin/python reports/task3/rare_expansion_20260906/build_collection.py travel
./.venv/bin/python reports/task3/rare_expansion_20260906/check_internal_duplicates.py
./.venv/bin/python reports/task3/rare_expansion_20260906/build_collection.py finalize
./.venv/bin/python reports/task3/rare_expansion_20260906/make_review.py
```

These steps require the existing source files and teacher image inventory. They do not train
a model. Source acquisition scripts are separate; preserve their final reviewed manifests
before rerunning downloads because public catalogues can change.
