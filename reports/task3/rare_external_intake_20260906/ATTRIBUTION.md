# Source records and image changes

Each accepted row preserves its original product URL, image URL, source text and SHA-256 hashes
in [splits.csv](splits.csv). Original photos stay unchanged. Prepared versions apply EXIF orientation,
RGB conversion, LANCZOS downscaling and white letterboxing to 60×80 PNG. No crop or new Usage label
was applied to a training image. Foreground crops exist only in the duplicate-comparison cache.

## Amazon Berkeley Objects

Images and product data: **Amazon.com**. Dataset construction: Matthieu Guillaumin, Thomas
Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi (Brian) Zhang, Tomas Yago-Vicente,
Jasmine Collins, Shubham Goel and Jitendra Malik.

The [official release](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The actual release files were checked;
the full [licence](sources/abo/raw/LICENSE-CC-BY-4.0.txt) and
[source attribution](sources/abo/ATTRIBUTION.md) are saved. No endorsement is implied.
Six prepared Smart Casual images are included from two source model families.

## Flipkart products

Dataset: **PromptCloud / PromptCloudHQ, Flipkart Products, version 1**, through the
[publisher's Kaggle release](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products).
The publisher metadata declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Retailer image ownership remains separate; the metadata release does not establish that the
publisher owns every linked photo. This local academic intake preserves that limit and does not
claim unrestricted image redistribution rights. Product/photo sources are linked per row.
The prepared set contains 97 Party images and one Travel image.

## Amazon Reviews 2023

Dataset: **McAuley Lab, Amazon Reviews 2023**. See the
[publisher page](https://amazon-reviews-2023.github.io/) and
[metadata release](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
Images are original seller product photos linked by that metadata. The checked public release
does not supply a blanket photo redistribution grant; retailer/image ownership remains intact.
Sixteen Home images were selected for this local academic experiment. Their existing seller
`Occasion: Home` values are preserved, with no teacher-label equivalence asserted.
