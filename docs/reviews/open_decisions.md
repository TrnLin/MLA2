# Open human decisions

These checks are useful later. They never stop data preparation, cache checks, tests, or EDA
Run All. The current automatic rules stay conservative while the questions are open.

## 1. Cross-role visual matches

Decide whether these labelled/prediction pairs show the same product. Current safe action: keep
every affected labelled product in quarantine and keep every prediction product out of development.

`50723/52131`, `49743/58884`, `50311/58893`, `48604/53103`, `48716/59550`,
`49740/58893`, `50305/58893`, `49743/58888`, `48708/59545`, `48591/58619`.

## 2. Broad product-name families

Decide whether normalized names are too broad as split blocks. Current safe action: keep each block
whole, which prevents leakage but reduces the number of independent units. Start with family
`family_3a8dd25529104cb0`, name `Lucera Women Silver Earrings`, 80 product IDs from `48590` to
`48728`. The full ID list is in `data/processed/splits.csv`.

## 3. Low/high image alignment warnings

Decide whether any same-ID high-resolution image should be excluded. Current action: keep both
same-ID files, label the 84 high hash-distance pairs as warnings, and test original-only versus
paired inputs before choosing a model.

`2616`, `2618`, `2662`, `2664`, `3673`, `3896`, `4894`, `4895`, `6833`, `8250`, `8354`,
`8420`, `8503`, `8619`, `9191`, `9547`, `9560`, `10245`, `10654`, `10655`, `11788`,
`12091`, `12337`, `13232`, `13233`, `13290`, `13331`, `13332`, `13796`, `14659`,
`15024`, `16643`, `17080`, `17085`, `17117`, `17705`, `17797`, `17808`, `17858`,
`18041`, `18272`, `18312`, `20412`, `20455`, `22033`, `23261`, `24179`, `24912`,
`25404`, `27220`, `27315`, `27893`, `30246`, `31172`, `31549`, `33028`, `33218`,
`33399`, `34108`, `34118`, `34156`, `34157`, `35437`, `35446`, `37303`, `37320`,
`37580`, `38737`, `42609`, `45135`, `46167`, `47370`, `48764`, `48851`, `52209`,
`52210`, `52214`, `53621`, `55004`, `55034`, `56199`, `56201`, `56219`, `59770`.

## 4. Task 4 relevance proxy

Decide whether the metadata grades match what people call visually similar. The proxy is not
real-world similarity ground truth. Start with validation query `1535` and train-gallery examples
`1566` (same type and colour), `1581` (same type), `1526` (same colour only), and `1163`
(unrelated). Current action: report proxy metrics only and keep this limit beside every result.
