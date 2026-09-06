# Party replacement candidates — 2026-09-06

38 downloaded candidates: 18 clutches, 8 perfumes and 12 women's watches. No dresses, heels or tops added after the parent audit found enough of those types already.

## Label strength

- **18 clutches: strong.** Olga Berg product tags explicitly say Cocktail, Evening or Night Out. One hero per named design; colour variants excluded. Sources include woven, shell-shaped, fringe, crystal, mesh, envelope, lace and hardcase designs. A top-handle bag was excluded from the final candidate list because it was not a clutch.
- **8 perfumes: strong source label, mixed use.** All eight are returned by the manufacturer's Party collection. Some also have daily/date/office uses. They are different named fragrances, although several share the same rectangular bottle design. Do not treat bottle colour alone as the source of the Party label.
- **12 watches: conditional.** Product descriptions explicitly mention special occasions (11) or evening occasions (1). They also mention everyday or work use. These are **medium-confidence contextual Party inferences**, not explicit Party labels. The parent must decide whether this ambiguity is acceptable; otherwise keep all 12 outside the admitted dataset. Do not use the generic Dress collection alone as evidence for Party.

## Sources and saved evidence

- `olga.json`: https://olgaberg.com/collections/clutch/products.json?limit=250
- `bellaparty.json`: https://bellavitaorganic.com/collections/party/products.json?limit=250
- `bellaparty.html`: https://bellavitaorganic.com/collections/party
- `dayt.json`: https://dayt.ie/collections/dress-watch/products.json?limit=250 (copied unchanged from the other worker's public-source cache; source verified by web read)
- `candidates.csv` has product URLs, exact source evidence, full descriptions, image URLs, family IDs, hashes, dimensions and rights status.

Retailer/manufacturer photos remain copyrighted. No open image licence was established. These are research review candidates, not a claim of redistribution permission.

## Visual check

Viewed both `contact_01.jpg` and `contact_02.jpg` at full sheet size after downloading. Re-viewed contact 1 after replacing the top-handle bag with BONNIE Lace Clutch. All final photos show one clear product against white or light grey backgrounds. No broken photos, collages, text-heavy promotions, humans obscuring items or missing products found. Perfumes have text only on their normal bottle labels. Watch faces and bands are visible. Raw photos were not edited; contact sheets are thumbnails only.

38 distinct file hashes. Cross-source/teacher duplicate checks and final saved folds are the parent's responsibility. No split or training files changed.

## Limits

Titan, Fastrack, Nykaa, Lulus, Yours and Dune returned HTTP 403 to public requests. No access controls were bypassed. Their error pages and other unused source caches remain here as discovery records. ALDO party shoes, Beginning Boutique party tops and Esther party tops were accessible but no photos were downloaded because the parent reprioritized the missing accessory categories.

`acquire.py` reproduces selection and downloads. Files in the image folder that are absent from `candidates.csv` are excluded discovery downloads, not admitted candidates.
