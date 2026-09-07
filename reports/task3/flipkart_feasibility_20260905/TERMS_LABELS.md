# Flipkart: release terms and Usage label fit

Checked 2026-09-05. Verdict: metadata reuse has a stated license. The labels support a small weak-label or auxiliary-label experiment, not a clean replacement for the nine-class Usage data. These are merchant text tags, not checked image labels. No new human annotation is required for the proposed automatic subset, but its label noise remains unknown.

## Verified release terms and remaining gap

- [Publisher page](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) and [public publisher API](https://www.kaggle.com/api/v1/datasets/view/PromptCloudHQ/flipkart-products): PromptCloudHQ, dataset 2506, version 1, last updated 2017-09-15, license **CC BY-SA 4.0**. The API snapshot is `publisher_metadata.json`; the web page itself did not expose readable text through the web tool. The publisher describes a crawl of Flipkart and lists an image field. The locally inspected release has CSV metadata and remote image URLs, not a bundled image collection.
- [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/) permits copying and adaptation, including commercial use. When sharing, keep attribution, source and license links, supplied notices, and a record of changes. Share adapted licensed material under the same or a compatible license and add no restrictive terms.
- [Legal code, sections 1–5](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) limits the grant to rights the licensor can license, covers relevant database rights, and gives no rights warranty. Thus the publisher's dataset license is verified; its authority over every remotely hosted product photo is not established by this release. This is an unresolved scope question, not evidence that research use is prohibited. It also does not establish that every trained model must use CC BY-SA.

Suggested metadata attribution: “Flipkart Products, PromptCloud / PromptCloudHQ, version 1, via Kaggle, CC BY-SA 4.0. Changes: parsed specifications, filtered rows, and derived label candidates.” Retain both links above when sharing the derived metadata. Remote photo redistribution needs its own established basis; this audit does not claim it has one.

## Counts and policy

Only source metadata and `data/processed/taxonomy.json` were read. No protected teacher labels or target evaluation rows were read. The audit script can be rerun with `./.venv/bin/python reports/task3/flipkart_feasibility_20260905/label_audit.py`.

There are **20,000 rows / 19,998 unique product IDs**. Occasion exists for **9,954 rows / 9,954 IDs**; 10,046 rows lack it. Among tagged rows, 8,142 have one token and 1,812 have several. **489 rows contain several canonical Usage names**, so selecting the first tag would manufacture a single-label answer. There are no differing Occasion token sets across repeated product IDs. IDs do not establish visual independence.

The conservative candidate rule requires exactly one token, an exact canonical label, and one of five broad fashion roots: Clothing, Footwear, Jewellery, Watches, Bags, Wallets & Belts. This is a preliminary coarse filter, not the final article-type mapping. Malformed roots and Baby Care are omitted here even when their items may be fashion. The final type and image checks can reduce or revise the count.

| Canonical Usage | All exact singleton rows/IDs | Coarse fashion candidates rows/IDs |
|---|---:|---:|
| Casual | 4,910 | 4,767 |
| Ethnic | 37 | 37 |
| Formal | 470 | 467 |
| Home | 0 | 0 |
| NA | 0 | 0 |
| Party | 363 | 342 |
| Smart Casual | 0 | 0 |
| Sports | 137 | 130 |
| Travel | 2 | 2 |

The conservative total is **5,745 distinct IDs**, before image access, type compatibility and duplicate checks. Casual dominates. The smaller candidates are useful mainly for Formal and Party; Ethnic is narrow and Travel is negligible. All 56 rows carrying an Ethnic token are Footwear; this is not evidence of broad ethnic clothing coverage. All 20 Travel-tagged rows are bags, and only two are singleton Travel (both named Brandvilla Shoulder Bag). Workwear is overwhelmingly jewellery (770 of 777), which directly undermines treating it as an automatic synonym for Formal or Smart Casual.

## Retain, exclude, and experiment

Retain exact singleton **Casual, Ethnic, Formal, Party, Sports, Travel** as *provisional weak Usage labels* after the final article-type and image filters. Same spelling is insufficient proof of the same concept: seller Occasion can mean suitable occasions, while the target expects one Usage class. Cap dominant Casual sampling, preserve the existing target split, and compare against the same target-only baseline. These counts do not imply improved accuracy or minority recall.

Do not turn missing Occasion into **NA**. Do not map **Everyday → Casual**, **Work/Workwear → Formal or Smart Casual**, **Lounge Wear → Home**, **Festive/Religious/Wedding → Ethnic**, or **Party-Wedding/Evening/Party → Party** without evidence. The remaining tags—Love, Wedding and Engagement, Wedding, Religious, Festive, Lounge Wear, Beach Wear, Party-Wedding, Evening/Party, Work, Special, Special Occasion, Lifestyle, Party and Celebration, Festival, All Occasions, Outdoor Adventure, Birthday, Event, Friendship Day, and the malformed `|Wedding and Engagement`—stay outside the direct Usage subset, along with Everyday and Workwear. Multi-token rows are excluded from direct single-label use even if only one token happens to match the target names.

An auxiliary experiment makes better use of the source as it is: keep its Occasion vocabulary as a separate multi-label prediction head and preserve all tags per product. Use only compatible fashion images with present Occasion; missing tags are unknown, not verified negatives. Drop the malformed token or document a deterministic formatting repair, and omit unsupported rare outputs by a declared minimum count. Train from scratch, then test transfer on the fixed target split. This avoids invented semantic merges, although seller tags remain noisy and incomplete. It is an optional comparison, not a promise of useful transfer.

`label_candidates.csv` records every row's decision; `label_compatibility.json` stores counts and conflicts; `occasion_by_product_group.csv` stores token counts by raw root category. These files intentionally do not perform the other audit's fine article-type mapping, deduplication, or image availability tests.
