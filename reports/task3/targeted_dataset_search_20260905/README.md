# Targeted search for extra Usage data

**Several sources can target different gaps. They are candidates for a small test; none is cleared for training yet.** Checked on 5 September 2026. Existing source labels only; no new human labelling is required by this plan.

| Weak class | Best new lead | What we actually checked |
|---|---|---|
| Home | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | **7 pillow-cover products with seller `Occasion: Home`** in a small metadata prefix. Two sampled image links worked. |
| Smart Casual | [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | **2 footwear products with a standalone `Smart casual` seller bullet** in one metadata shard. Both images worked. This does not fill the watch gap. |
| Travel | Amazon Reviews 2023 + ABO | **16 bag products with Travel in the title** in the Amazon prefixes; one ABO backpack has both Casual and Travel in its title. Three sampled images worked. These are weaker purpose clues, not verified Usage labels. |
| Sports | Amazon Reviews 2023 | **11 clothing/footwear products with existing sport-purpose fields** under a narrow metadata rule. Two sampled images worked. |
| Ethnic | [Wardrobe Assistant](https://www.kaggle.com/datasets/shahzaibmalik44/wardrobe-assistant) | **4,570 rows in men's/women's Ethnic categories**, with image URLs. Two images worked. The saved URLs all point to Andaaz Fashion; many photos show a whole outfit. Label creation is not explained. |
| NA | No suitable source verified | Missing tags, unknown styles and unseen body parts have different meanings. Do not convert them to teacher NA. |

Counts above are metadata products/rows, **not independent families or training-ready images**. The Amazon counts come from fixed first-8-MB prefixes, not full datasets or random samples. All **11 selected Amazon/ABO/Wardrobe images decoded** and were visually inspected. No teacher overlap check was performed in this search.

Two further leads have useful limits: [IKEA](https://huggingface.co/datasets/jeffreyszhou/ikea-us-products-2025) has a derivative index of **183 throw-cover SKUs**, but only product-type labels. [FashionStylist](https://github.com/recsys-benchmark/FashionStylist) has **48 “smart casual” description rows**, but its independent product count is unresolved, one matched record is marked AI-generated, and three sampled links led to login pages. Outfit-labelled and synthetic sources are documented in the full report.

**Recommended next step:** expand the ABO Smart Casual metadata check and the Amazon Home/Travel/Sports check, then run a small image-intake audit. Retain Flipkart for Party. Keep original source labels separate until their meaning is tested against the teacher task. Wardrobe is a later Ethnic helper source once its label provenance is clearer. No model has been trained on these new sources.

[Full findings](REPORT.md) · [Catalogue evidence and samples](catalogue/FINDINGS.md) · [Smart Casual search](smart_casual/FINDINGS.md) · [Travel search](travel/REPORT.md) · [Home/NA search](home_na/findings.md)
