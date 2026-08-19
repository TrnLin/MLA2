# Project Decisions

Important project choices are recorded here as one Markdown file per decision. Each record
states the context, decision, consequences, and evidence so later notebooks do not have to
guess why a rule exists.

## Accepted decisions

1. [Use original images only for official training IDs](0001-use-original-images-with-official-train-ids.md)
2. [Treat high- and low-resolution files as paired views](0002-treat-resolutions-as-paired-views.md)
3. [Separate raw data by provenance and use processed manifests](0003-data-directory-layout.md)
4. [Use review-driven cleaning and target-specific masking](0004-use-review-driven-cleaning-and-target-masking.md)
5. [Use group-aware splits with a catalogue holdout](0005-use-group-aware-splits-with-a-catalogue-holdout.md)
6. [Preserve class prevalence and test imbalance handling](0006-preserve-class-prevalence-and-test-imbalance-handling.md)

Add future records with the next four-digit number. Do not rewrite an accepted decision
silently; mark it superseded and link to the replacement record.
