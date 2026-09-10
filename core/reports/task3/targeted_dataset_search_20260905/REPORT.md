# More datasets for the weak Usage classes

5 September 2026. **A targeted mix is plausible. The strongest new item-level leads are ABO and Amazon Reviews 2023. NA remains unresolved.** This was a source and small-sample check; no new dataset has been admitted to training, and no performance improvement has been measured.

The search covered product catalogues, academic fashion releases, bags, household textiles, existing occasion annotations and synthetic collections. Separate searches checked Smart Casual, Travel, and Home/NA. Dataset derivatives were traced to their underlying sources rather than counted as independent data.

## What the teacher task needs

The saved development audit has Smart Casual **47 images/37 families**, Travel **22/22**, Home **1/1**, Party **12/10**, and NA **61/49**. Casual has 25,151 images; Sports 3,346, Ethnic 2,183 and Formal 1,949. A family means a related product group, which matters more than counting repeated views. See [verified class support](../usage_deep_investigation_20260905/class_support.csv).

The weak classes also have different product mixes. Smart Casual includes many watches and shirts; Travel contains bags; Home is a cushion cover. A new label with the same name can still mean something else. In the saved model audit, persistent errors also appear in Sports T-shirts, Formal shirts/shoes and Ethnic earrings/kurtas. More data should target these product/use combinations, with contrasts from the same product type. See [error slices](../usage_deep_investigation_20260905/persistent_error_article_types.csv). Protected evaluation labels were not used to find or select external records.

## Ranked source mix

| Target | Useful sources | Verified signal | Decision |
|---|---|---|---|
| Smart Casual | **ABO**; FashionStylist reserve | ABO: two footwear IDs with original standalone Smart Casual bullets, both images accessible. FashionStylist: 48 style-description rows; independent product count unresolved. | Expand ABO metadata first. Three FashionStylist links returned login pages; no watch coverage among its matches. |
| Home | **Amazon Reviews 2023**; IKEA reserve | Seven Amazon throw-cover products with exact seller Occasion=Home in the prefix. IKEA derivative index: 183 cover SKUs. | Amazon is the closer label match. IKEA is a product-type helper only. Photo content still needs checking. |
| Travel | **Amazon Reviews 2023 + ABO**; Garments2Look reserve | Sixteen Amazon bag-title matches and one ABO Casual Travel backpack. G2L supplement shows a travel-bag product type. | Purpose clues only. No clean single-label Travel bag set verified. Preserve mixed use descriptions. |
| Sports | **Amazon Reviews 2023**; iMaterialist reserve | Eleven matched clothing/footwear products with sport-purpose fields in the prefixes. iMaterialist has athletic garment categories. | Useful separate source task; do not label every outdoor or fan product Sports. |
| Ethnic | **Wardrobe Assistant**; IndoFashion reserve | Wardrobe: 4,570 Ethnic category rows, linked photos. IndoFashion: existing Indian garment-type labels. | Wardrobe is accessible but label construction unclear; mostly one retailer and full outfits. IndoFashion remains gated and has Myntra overlap risk. |
| Party / Formal contrast | **Keep Flipkart**; iMaterialist reserve | Prior Flipkart audit: 186 Party and 300 Formal metadata-group candidates. iMaterialist distinguishes dress occasions and shirt/shoe categories. | Continue the existing partial lead, with seller labels kept separate. |
| NA | **No compatible source verified** | Other NA fields mean missing metadata or invisible parts. | Do not manufacture NA from absent external labels. |

The counts are deliberately limited to what was checked. Amazon means fixed byte prefixes, ABO means one metadata shard, IKEA's 183 comes from a derivative index, and Wardrobe counts rows rather than independent families. None is a guaranteed training yield. [Catalogue audit](catalogue/FINDINGS.md), [previous Flipkart audit](../flipkart_feasibility_20260905/REPORT.md).

## Why ABO and Amazon come first

The [ABO release](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) packages product metadata and images with an explicit licence. One 5.4 MB shard already contains two standalone Smart Casual seller bullets. Three exact item-to-image joins, including a Travel backpack, were checked through the official image index. This is a practical route to inspect the remaining metadata for more relevant product types; it is not yet evidence of enough independent Smart Casual watches/shirts. [Saved original text and image joins](travel/ABO_ACCESS.md).

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) has product photos plus structured detail fields. The seven Home-tagged pillow covers are a closer existing-label lead than household scenes. Sports purpose fields give more precise supervision than a broad shopping department. Travel and Smart Casual marketing text is often mixed with other uses, so it stays weak evidence. The public card did not establish a blanket photo reuse grant. Source access, reuse terms, label meaning and teacher overlap are separate checks. [Exact records and limits](catalogue/FINDINGS.md).

The small image probes passed **11/11** across Amazon, ABO and Wardrobe. Visual inspection showed a real limitation: even correctly linked product metadata can lead to a flat printed design, a collage or a whole outfit. Some detail becomes tiny at 60×80. These probes establish access and expose selection issues; they do not establish label quality, independence or model benefit.

## Other sources worth knowing about

**FashionStylist** is a new, expert-annotated release with 1,000 outfits and 4,637 item rows. Among 48 English Smart Casual phrase matches, 29 rows expose 23 distinct product IDs, 18 have unresolved short links and one is explicitly marked AI-generated. The total independent-product count is therefore unknown. Translation differences also remain in the Chinese labels. The words occur in item-style descriptions, not an exact Usage column. Three sampled source links led to login pages. Its Travel/Home counts are outfit labels, which cannot be copied to every item. Keep it as an access-blocked lead. [Author release](https://github.com/recsys-benchmark/FashionStylist), [bilingual evidence and access results](smart_casual/FINDINGS.md).

**Garments2Look** connects real product reference images to generated outfit images and model-assisted text. A bounded metadata sample verifies Smart Casual outfit styles; the supplement shows `bags::travel bags`. Those do not make every component item Smart Casual or Travel. Its Mytheresa product metadata is a possible later narrow source; no eligible bag total was checked. Its Polyvore branch shares a source family with other Polyvore datasets. [Publisher](https://huggingface.co/datasets/ArtmeScienceLab/Garments2Look), [Travel check](travel/REPORT.md).

**Fashion32 through FashionRec** provides 1,724 Travel outfit rows in the checked derivative, versus 1,706 in the original paper. The release versions differ. More importantly, the derivative code copies outfit occasion into each item record and keeps the first duplicated image. An apparent item Occasion column is therefore not independent item-level ground truth. Do not use those rows to enlarge teacher Travel. [Original paper](https://arxiv.org/abs/1912.06227), [construction code](https://huggingface.co/datasets/Anony100/FashionRec/blob/main/construct_parquet.py), [count audit](travel/fashionrec_metadata_audit.json).

**IKEA US 2025** has seller category labels and image references, including a derivative index of 183 throw-cover SKUs. These are product-type labels, not Usage. The original total was not independently recounted; derivative records include color variants from the same named design. Its blank Style column is not NA. **IndoFashion** has relevant garment categories but requires an access request; its NA means missing attributes, and its cited sources include Myntra. [IKEA original release](https://huggingface.co/datasets/jeffreyszhou/ikea-us-products-2025), [IndoFashion official docs](https://indofashion.readthedocs.io/en/latest/tutorials/info.html), [source checks](home_na/findings.md).

**Wardrobe Assistant** provides existing Ethnic category labels and working sample images. All inspected image URLs point to Andaaz Fashion, despite a broader publisher description. Its 4,570 relevant category rows are not a nine-class occasion set; combined source occasions and unspecified label creation need care. The two images show complete ensembles. [Publisher](https://www.kaggle.com/datasets/shahzaibmalik44/wardrobe-assistant), [verified metadata and images](catalogue/FINDINGS.md).

**iMaterialist** remains a useful comparison source: the [actual author label map](https://raw.githubusercontent.com/visipedia/imat_fashion_comp/master/iMat_fashion_2018_label_map_228.csv) includes Athletic Shirts/Pants/Shorts, Casual Shirts, Dress Shirts, Casual Shoes, Business Shoes and Running Shoes, plus Casual/Formal/Party Dresses. These are native categories that could support contrasts within a product type. This is broader than a dress-only helper, but it does not add Travel, Smart Casual or Home Usage labels. Live image access and dataset reuse rules remain unverified. [Author release](https://github.com/visipedia/imat_fashion_comp).

Other checked sources failed for clear reasons: Bag6k, baggage re-identification and luggage detection label objects/scenes; Dress Code is virtual try-on; IQON and Hipster Wars use different tasks/styles; DeepFashion attributes are not the nine occasions; DeepFashion-MultiModal NA means not visible; LookMatch is entirely synthetic. The companion reports retain sources and exclusions so they do not need to be rediscovered. Generic scene or clothing datasets were not shortlisted merely because they contain images.

## Concrete next step

1. **Expand metadata where there is already item-level evidence.** Scan all ABO metadata for native Smart Casual phrases and retain exact text, product types and sibling IDs. Expand Amazon Home/Travel/Sports metadata under a recorded byte/row cap; keep the seven Home examples as reproducible seeds. Do not infer dataset-wide quantities from the current prefixes.
2. **Audit a small image intake by target and product type.** Check exact/decoded/near-duplicate images against every teacher role using image-only columns. Group repeated source products and views across ABO and Amazon, which can share products. Keep mixed labels and uncertain label provenance visible. A lack of clear usable source evidence is a reason to withhold that record, not to invent its label.
3. **Only then define a matched training test.** Keep source targets separate at first, preserve teacher exposure, optimizer-step budget and all nine evaluation classes. Use scratch models, the canonical split, and the run registry. Evaluate changes per class and on confusing product/use pairs. Home's one development example cannot support a stable estimate of general Home accuracy, even if external training becomes possible.

This sequence adds no new human-labelling requirement. It also does not claim external data will cure ambiguous labels: an item genuinely used in several contexts may still have only one teacher target. The search has found a concrete multi-source route and its limits; image admission and model experiments remain separate future work.

## Reproducibility and unchanged contracts

The new files are confined to this report folder and its figures. The canonical split, teacher labels, training code and run registry were unchanged by this search. No fit, commit, push or bulk image download occurred. All report tables can be traced to saved metadata or explicit publisher statements. See [verification.json](verification.json) for cached count/hash checks and [source notes](catalogue/FINDINGS.md) for image provenance. Reports were rendered and visually inspected; no visual-quality conclusion is based on text extraction alone.
