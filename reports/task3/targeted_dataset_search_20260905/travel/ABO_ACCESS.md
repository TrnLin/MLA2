# ABO image access check — 5 September 2026

All three requested original images were fetched from the official ABO S3 release and opened successfully. Their decoded dimensions match the official image index. Total image download: **1,130,849 bytes**. The index was **6,430,535 bytes**. No CDN paths were guessed.

Resolution route: read the existing official product shard, take `main_image_id`, join it to `images/metadata/images.csv.gz` by `image_id`, then append its `path` to `images/original/`, as specified by the [official image README](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/README.md). Exact joins, file hashes, original text, and dimensions are saved in `abo_access/access_evidence.json`.

- **B07255FKTS → 71gjcdJqBdL → `5f/5fdc8e6a.jpg`.** 1498×1030, 168,818 bytes. Original product bullet contains `Smart casual`. Visually inspected: one brown ankle boot on white, filling most of the frame. [Original source](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/5f/5fdc8e6a.jpg).
- **B0727R3L19 → 71U1Nx2vKhL → `36/3662c001.jpg`.** 1969×2560, 198,164 bytes. Original product bullet contains `Smart casual`. Visually inspected: one burgundy loafer on white with a reflection and a large empty margin above it. [Original source](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/36/3662c001.jpg).
- **B07PGMRKSC → 91YzxsRGIWL → `aa/aad0fd97.jpg`.** 2322×2322, 763,867 bytes. Original title is `AmazonBasics Casual Travel Backpack, Grey`; `product_type=BACKPACK`. Visually inspected: one grey backpack on white, seen from the front at an angle. [Original source](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/aa/aad0fd97.jpg).

This verifies item-to-image access only. The merchant text is existing evidence, not a newly assigned Usage label. No image relabeling, teacher duplicate clearance, claim of independence, training, or evaluation-label reading occurred. The loafer's empty margin should be considered when testing preprocessing later; no crop or resize was applied here.

The actual [image README](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/README.md) and [release licence](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/LICENSE-CC-BY-4.0.txt) declare **CC BY 4.0**. The [AWS registry summary](https://registry.opendata.aws/amazon-berkeley-objects/) currently says CC BY-NC 4.0, a conflicting catalogue entry. This check records the actual release's explicit licence rather than silently merging those claims.

Attribution for these unmodified images: **Amazon.com**. Dataset construction credit: **Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine Collins, and Jitendra Malik**. [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/); release includes a disclaimer of warranties. Local copies are unchanged.

Recheck the saved files with:

```bash
./.venv/bin/python reports/task3/targeted_dataset_search_20260905/travel/abo_access/audit.py
```
