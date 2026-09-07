# Independent evidence review: Party and camera bags

Reviewed 2026-09-06. Read-only review of the selected 44 Party rows and six camera-bag Travel rows. No downloads, selection edits, training, or data integration.

## Result

No unsupported product IDs or image-to-product mismatches found in the selected files. All 50 local image SHA-256 values match their manifest. This confirms local file integrity, not an independent certificate of source authenticity.

## Party: 44 rows

- **18 Olga Berg clutches:** Every selected numeric product ID exists in `party/olga.json`; the product handle matches the selected product URL. Every recorded occasion tag exists on that exact product. Every selected image matches one of that product's image URLs after removing only the added `width=800` transformation. Tags are Cocktail, Evening, and/or Night Out. These support occasion inference; Evening and Night Out are not literally the teacher label Party. In particular, AMALIA `7242107617454` and ZARA `7546350174421` have Evening only; BETH `6280945270958` and NIC `6261422882990` have Night Out only. These four are semantically weaker than explicit Cocktail/Party wording, but their evidence is accurately recorded.
- **Eight Bella Vita perfumes:** Every ID and hero image matches `party/bellaparty.json`. Independently checked `party/bellaparty.html`: its page title is Party and all eight selected product handles occur in that page. Therefore the collection membership is supported by both native product data and the retained collection HTML. No claim that a perfume is exclusively for parties is warranted.
- **12 Oh Polly dresses:** Every ID, product handle, and hero image matches `party/ohpolly_party.json`. The acquisition code and manifest record the official `https://au.ohpolly.com/collections/party-dresses/products.json?limit=250` endpoint. The cache itself is plain Shopify JSON without HTTP response headers or embedded request URL; the endpoint provenance depends on the retained acquisition record. No product/image mismatch found. Reviewed `party/supplemental_contact.jpg`; dress images show the described black dresses rather than unrelated items.
- **Three Fastrack watches:** `68026WM01`, `6288QM01`, and `6279SM01` are linked by the cached Partywear listing. The collection text explicitly describes party ensembles and night-out use. Product-click records link listing link IDs 64, 70 and 73 to the corresponding model URLs/names. The observed image URLs contain the same exact model numbers, and these URLs match the CSV. The cached evidence also includes an unselected `6279WM01` and an unrelated After Dark page; neither was mistaken for one of the three selected products.
- **Three Titan watches:** `95318WM01`, `95319WM01`, and `95320WM02` are reached from the cached official Raga Cocktail collection. Its copy names cocktail parties, date nights, weddings and festive evenings. `95318WM01` has an explicit Raga Cocktails product collection specification in the retained product extract. Product titles and source click records identify the other two as Raga Cocktails. All three selected image URL filenames use the correct model number.

### Watch redirects and cache limits

`95319WM01` starts at a URL containing `blue-dial`, then redirects to a URL containing `rose-gold-dial`. The model number remains **95319WM01** through the redirect, product title and observed image URL (`95319WM01_1.jpg`). The selected manifest does not claim a blue dial; it uses the model name. Thus this is a stale descriptive slug, not evidence of a product identity mismatch.

Several image-link web fetches timed out. Their cached tool records still expose the exact image URL that was clicked; local image files exist and hash-check against the selected manifest. The evidence supports URL/ID alignment but does not independently prove the HTTP download response that originally produced each image. Product web extracts vary in completeness: some retain only the source/title summary. Read the limitation as provenance depth, not missing model identity. Supplementary contact-sheet review is consistent with the named watches.

## Camera-bag Travel: six rows

For each of the following five Shopify products, decoded its individual cached JSON, matched numeric product ID, checked the exact selected image URL against its native image list, and found the recorded travel sentence in the decoded product description:

| Product ID | Product | Source evidence |
|---|---|---|
| 4097572405309 | Think Tank Retrospective 4 V2.0 | Explicitly addresses travel and street photographers |
| 7214741520464 | WANDRD ROGUE 6L Sling | Explicitly describes traveling with essentials |
| 8963050963181 | Billingham S4 Camera Bag | Meets cabin size restrictions and is suitable for travel |
| 8963048407277 | Billingham Mini Eventer | Explicitly addresses tech-savvy travellers |
| 8963047555309 | Billingham 225 MKII | Product-specific travel/photography use retained in description |

For Tenba `638-570`, `camera_travel/dna-9-messenger-bag-black.html` contains the exact SKU, selected image URL and recorded adventure-travel wording. The snippet begins mid-sentence (`020 model]`), but the substantive travel claim is present in the product description; it is not invented. All six are correctly marked **product_text_inference**, not original teacher labels.

Viewed `camera_travel/contact_sheet.jpg`. The six selected bags match the described closed shoulder/sling camera bags. The sheet also contains the unselected Domke and Manfrotto items; confirmed neither appears in the selected six-row CSV.

## Scope limits

This review validates retained source evidence, product/image alignment, and local image integrity. It does not establish exclusive usage, open image reuse rights, teacher-distribution similarity, external/teacher duplicate freedom, or correct family split assignment. Those remain central audit responsibilities.
