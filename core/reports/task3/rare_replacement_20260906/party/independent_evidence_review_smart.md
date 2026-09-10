# Independent Smart Casual evidence review

Reviewed all 62 rows in `../smart_casual/selected_candidates.csv` against cited local JSON/HTML. No candidate file changed. No invented source evidence, wrong product membership or wrong source image found. All 59 Shopify product IDs, handles and image URLs match their cited product cache. The other three are Care of Carl products.

## Confirmed evidence

- **19 DA:YT watches:** every ID exists in the cached `dayt.json` dress-watch collection. `dayt_collection.html` defines that collection as made for formal, business and smart-casual wear. This supports the allowed collection inference, with mixed-use/medium confidence; it is not an exact exclusive Smart Casual label.
- **12 M.J. Bale shirts:** each exact product has the native `usage:Smart Casual` tag in `mjbale_shirts.json`. Pass.
- **3 Care of Carl wallets:** IDs `careofcarl_23884210`, `careofcarl_26843310`, `careofcarl_31528210` have Smart Casual inside the product's `pdp-romance__badges` block. The pages also contain unrelated onboarding Smart Casual text, but the selected evidence genuinely exists in the product badge block. Pass.
- **Other 12 sandals/ties/wallets:** product-specific smart-casual wording exists in the relevant product description. Their selected hero URL belongs to that product. Pass, allowing mixed uses and style-language inference.

## Two wording limits to preserve

**14 Sekonda/Edmonds watches:** each exact product description says it can coordinate with business casual through everyday wear. The evidence is product-specific, although repeated boilerplate. It says **business casual**, not Smart Casual. The CSV notes saying the retailer explicitly mentions smart casual are inaccurate for these rows. Admission is supported only if business casual → Smart Casual is an allowed mapping. If that mapping is not allowed, exclude these IDs (prefix `www_edmondsjewellers_com_`):

`9793970897235`, `9793971192147`, `9793971224915`, `9793971290451`, `9793971749203`, `9793972666707`, `9794459402579`, `9794459435347`, `9794459861331`, `9794459926867`, `9794459959635`, `9794460057939`, `9794461237587`, `9796622844243`.

**2 Timex Marlin watches:** `shop_timexindia_com_10156312953121` and `shop_timexindia_com_9313635598625` are real members of the Marlin collection and both descriptions confirm natural leather straps, acrylic crystal and vintage/retro design. The collection says a leather-strapped piece *like* the Marlin Blue Round Dial Analog works for office/smart-casual wear. Neither selected model is that named blue-dial example (one is silver-dial chronograph; one black-dial automatic). Thus this is a reasonable attribute-based collection inference, but not exact-product evidence or an explicit label applying to every Marlin. **For a strict exact-product or whole-collection-label rule, hold these two out.** For the broader allowed collection-text inference rule, retain only as medium confidence and describe the extrapolation honestly.

## Conclusion and limits

Parent clarified the earlier v2 intake rule explicitly permits business-casual → Smart Casual and this form of collection-level style inference. Under that clarified rule, **all 62 have permitted evidence; no unsupported IDs found**. The 14 Sekonda rows qualify as business-casual text inferences, and the two Timex rows qualify as weaker attribute-based collection extrapolations. Neither group should be described as product-specific native Smart Casual labels. Parent will correct central provenance wording for the 14 Sekonda rows.

This is a saved-source evidence audit, not a licence, teacher/external duplicate or fold audit. No fresh acquisition or visual re-grading was performed; image/source identity was checked against metadata. The existing README calls Sekonda descriptions smart-casual, which should be corrected to business-casual mapping.
