# Fashion Intelligence

Shared language for the fashion classification and visual-search investigation.

## Language

**Candidate run**:
A learned model trained on the allowed development-training folds and measured on the fixed
development validation fold. Adjacent candidate runs change one main experimental factor.
_Avoid_: Trial, random model

**Current champion**:
The completed candidate with the highest development winner score. It becomes the parent of the
next incremental candidate.
_Avoid_: Deployment winner, final model

**Incremental candidate**:
The base candidate or a child that changes one planned factor from the current champion.
_Avoid_: Unrelated model

**Breadth candidate**:
A separately trained scratch model from a meaningfully different model family. It may become a
stability finalist, but it is not the parent of changes designed for the incremental family.
_Avoid_: Comparison-only benchmark

**Comparison-only benchmark**:
A model used to show context or an expected ceiling, but not eligible to become the submitted
Task 4 model. A model with pretrained weights belongs only in this group.
_Avoid_: Final model, submitted model

**Development winner score**:
The fixed Protocol A mean per-query linear nDCG@10 used to rank candidate configurations and
choose the two stability finalists.
_Avoid_: Final judgement

**Deployment judgement**:
The final recommendation based on retrieval quality, source robustness, failure behaviour,
latency, and storage. It need not be the model with the highest development winner score.
_Avoid_: Best accuracy, winner score

**Source robustness ratio**:
The equal cross-source Protocol A score divided by the equal same-source Protocol A score.
The frozen 95% value is an improvement target, not an eligibility gate.
_Avoid_: Source accuracy

**Stability finalist**:
One of the two highest-scoring candidate configurations retrained from scratch across all five
development folds.
_Avoid_: Holdout finalist

**Teacher view**:
The provided low-resolution image for a product.
_Avoid_: Independent source

**V1 view**:
The high-resolution external-catalogue image variant for the same product ID. It is not
independent evaluation data.
_Avoid_: External test image, independent image

**Cross-source pair**:
The teacher view and V1 view that share one development product ID.
_Avoid_: Two products, independent pair

**Two-view gallery**:
A gallery that stores both the teacher-view and V1-view embedding for each product but allows
that product to occupy only one result position.
_Avoid_: Duplicate products

**Task 4 embedding**:
A unit-length vector of 128 floating-point numbers produced by an image encoder.
_Avoid_: Class prediction, similarity score

**Task 4 search result**:
A ranked Top-K list of distinct product IDs produced by comparing one query embedding with the
gallery embeddings.
_Avoid_: Embedding, classification prediction

**Stability tie**:
Two stability finalists whose five-fold mean scores differ by no more than their pooled
fold-to-fold standard deviation.
_Avoid_: Exact tie
