# Exact-label/type candidate image probe

37 of 41 representative products yielded usable decoded images; 4 failed. 0 verified saved images reused. Additional requests: 45; response body bytes consumed: 6932149.

- Casual: 7/8 usable.
- Ethnic: 8/8 usable.
- Formal: 6/8 usable.
- Party: 7/8 usable.
- Sports: 8/8 usable.
- Travel: 1/1 usable.

Selection comes from candidate_access_manifest.csv: at most eight deterministic metadata-group representatives per available exact source label. These source labels and candidate types remain metadata, not confirmed training labels. Availability supports further feasibility review only for the observed sample. It does not measure the entire pool or clear leakage.

Compared image SHA and dHash against all 44441 existing cached teacher images, including all split roles and prediction. Cache ID/path/SHA coverage verified against image-only manifest columns. 0 exact-byte or dHash <=2 candidates; candidate pixel metrics are recorded where applicable. No target columns read. No images admitted to training.

The script defaults to saved-file verification with zero network requests. The initial probe requires --refresh. Verified prior images are reused even with refresh. Source first image URL only; failed HTTP permits one scheme-only HTTPS retry; no alternative CDN paths. Four workers, 15-second connect/read timeouts, 2 MiB per file and 40 MB additional body budget. Error response bodies are not consumed. No further sampling is planned.
