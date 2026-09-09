# Flipkart data check

**Worth a small trial with uncertain seller labels. It does not cover the whole Usage problem.**

- Product tags match **85 of 124** teacher types. This is a metadata match, not verified image coverage.
- After filtering and grouping repeats, **3,094 candidates across 50 types** remain: 2,497 Casual, 300 Formal, 186 Party, 91 Sports, 19 Ethnic and 1 Travel.
- There are **no exact Smart Casual, Home or NA candidates**.
- **37 of 41** images worked in the check aimed at this candidate pool. None matched the teacher fingerprints under the checked rules.
- The previews show some wrong-looking product tags and photos containing several items. Seller labels should remain a separate learning task.

**Next:** a bounded intake for a Casual/Formal/Party helper task, up to 150 images per source label. Every selected image still needs the full overlap check. [Saved intake plan](pilot_intake_plan.json). No model has been trained on Flipkart.

[Full findings](REPORT.md) · [All 124 product types](TYPE_COVERAGE.html) · [Candidate list](deduplicated_candidates.csv) · [Checks](verification.json)

Source: [Flipkart Products, PromptCloud, version 1](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products), declared [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes: parsed specifications, type matching, grouping and candidate filtering. The release's licence declaration does not by itself establish rights to redistribute every linked photo.
