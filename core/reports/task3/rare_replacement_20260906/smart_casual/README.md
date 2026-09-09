# Smart Casual replacement candidate review

Downloaded 63 products. Visually reviewed both contact sheets. Selected 50 candidates; rejected 13.

Selected counts: {'Sandals': 3, 'Ties': 4, 'Wallets': 8, 'Watches': 35}.

Use `selected_candidates.csv` for the parent duplicate/fold audit. `candidates.csv` keeps all downloads; `visual_qa.csv` records each decision. Source HTML and Shopify product JSON remain beside these CSV files. Source URLs, evidence text, source cache paths, file hashes and image dimensions are retained per row.

Evidence strength: Care of Carl wallets have product-specific Smart Casual style badges. Sekonda, other wallets, ties and sandals have product-specific smart-casual descriptions. DA:YT watches have weaker collection membership evidence: its dress-watch collection is explicitly defined for formal, business and smart-casual wear. Two selected Timex watches use the Marlin collection's leather-strap smart-casual description; only actual leather-strap images pass. These collection inferences remain medium confidence and should be identified in the report, not claimed as original teacher labels.

All photos are retailer catalogue assets. No open reuse rights were established. These are candidate assets for local academic evaluation, not a redistributable open dataset. No teacher/external duplicate audit, fold assignment, dataset integration or training was done here. Related watch designs share conservative family identifiers where identified; the parent should still run image similarity checks across all data.

Rejected paths are kept for audit only. The collection quota was not forced: weak source text or poor images were not relabelled to fill it. Titan blocked direct requests with HTTP 403; failed request HTML is retained. A few initially found product descriptions did not retain the matching text in their live JSON, so they produced no candidate row.

## Full-shirt extension

Added 12 complete front-shirt photos from M.J. Bale, each a distinct named shirt design. Every product has the explicit native tag `usage:Smart Casual` in `mjbale_shirts.json`. All 12 passed visual review in `shirts_contact.jpg`: sleeves and hem are visible, with plain light backgrounds and no folding. These are new source products, not alternate photos of prior M&S items. Total now 75 downloaded, 62 selected: 35 watches, 12 shirts, 8 wallets, 4 ties, 3 sandals. Existing 50 selected rows are preserved. Run `finalize_shirts.py` after `finalize_review.py` if rebuilding the selected CSV.
