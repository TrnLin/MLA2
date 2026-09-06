# Bounded Flipkart image access audit

Tested 48 distinct products: three per each of eight broad roots, then rare exact occasion values to reach 48. Hash ordering uses salt 2753. This is a purposive diagnostic sample, not a population estimate.

Usable decoded images: 41. Failed products: 7. Requests: 55.

Each product uses its first source-listed image URL. A failed HTTP URL gets one HTTPS scheme-only retry. URLs, responses, errors, decoded dimensions and SHA-256 are in image_access.csv. No alternate CDN paths, authentication bypass, or unofficial copies were attempted. Four workers; 15-second connection/read timeout; 2 MiB per response; 48 products bound saved image data to 96 MiB.

Duplicate audit: cached image-only comparison. Candidate count: 0. Cached teacher role counts: {'development': 32773, 'prediction': 5829, 'holdout': 5778, 'quarantine': 61}. Existing repository dHash algorithm is reused; dHash distance <=2 is only a candidate flag, with aHash and full-canvas/foreground pixel metrics recorded. It does not prove a duplicate or prove the absence of leakage. No protected labels are read. No external images enter training.
