# Stronger Party watches and simple black dresses

`supplemental_candidates.csv` contains 18 additional photos: 6 women's watches and 12 black mini dresses. The original `candidates.csv` was not changed.

## Watches

Three distinct Fastrack models (68026, 6288, 6279) are directly listed in the official Partywear collection. One colour per model. Exact listing/product web reads and observed image URLs are cached in `fastrack_*web_evidence.json`. The collection text explicitly discusses party ensembles and nights out. This is stronger evidence than generic special-occasion wording.

Three distinct Titan designs (95318, 95319, 95320) are directly linked by the Raga Cocktails page and have Raga Cocktails as their product collection. The collection explicitly describes cocktail parties, date nights, weddings and festive evenings. Saved evidence: `titan_*web_evidence.json`. Product 95319's original blue-dial URL redirects to the rose-gold product page; the CSV uses the resolved product identity and corresponding image. All three designs have different model numbers and different case/bracelet designs.

Public product pages returned HTTP 403 to ordinary requests, but web reads worked. Image URLs were observed by following the product's image links; direct public image downloads succeeded. No login or access-control bypass was used. Some web-image fetches timed out; the observed URLs downloaded successfully through requests and decoded as photos.

## Dresses

Twelve plain black mini dresses from the Oh Polly AU Party Dresses collection. Metadata is cached unmodified in `ohpolly_party.json`. Collection membership explicitly supports Party. Embellished, sheer, lace and faux-fur titles were excluded to favour simple designs resembling the teacher examples. One hero per product, no colour variants. The source differs from MESHKI, Beginning Boutique and Red Dress.

## Visual review

Inspected `supplemental_contact.jpg`. All 18 are usable: clear single products; all watches fully visible, five isolated white backgrounds and one light-grey setting. Dresses are fully visible on models against simple light backgrounds. No large text overlays or collages. Different necklines, sleeves, waists and skirt shapes provide variation. Images were not edited beyond the thumbnail contact sheet.

Recommended: use these 6 watches instead of the earlier 12 weak mixed-occasion Dayt watches. Parent performs final teacher/external duplicate audit and saved-fold integration. All 18 supplemental file hashes and model families differ internally.
