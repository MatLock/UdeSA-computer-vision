# Implementation Tasks

Tasks grouped by milestone. Each milestone ends with its test suite green before
moving on.

## Milestone 1 — Product Enrichment

- [ ] 1.1 Define the enriched-product data model (inferable fields)
- [ ] 1.2 Reuse CV pipeline (TinyVGG, ResNet, K-Means, BLIP) to produce fields
- [ ] 1.3 Add gender / material / pattern inference (image + tags + LLM)
- [ ] 1.4 Implement structured title builder (`Brand+Gender+Product+Attr+Color+Size`)
- [ ] 1.5 Implement `product_type` → `google_product_category` mapping table
- [ ] 1.6 Handle missing-image → `partial`, unknown-type → `needs_category_review`
- [ ] 1.7 Write tests (see `specs/product-enrichment/spec.md` → Tests)

## Milestone 2 — Merchant Feed

- [ ] 2.1 Define input CSV contract (business vs inferable columns)
- [ ] 2.2 Implement merge of enriched + business fields
- [ ] 2.3 Implement Google Merchant schema mapper
- [ ] 2.4 Implement validators (required fields, GTIN checksum, price format)
- [ ] 2.5 Implement validation report (valid/invalid counts + reasons)
- [ ] 2.6 Write tests (see `specs/merchant-feed/spec.md` → Tests)

## Milestone 3 — Merchant Publishing

- [ ] 3.1 Implement secure credential loader (env / service account)
- [ ] 3.2 Add environment selector (test default / production)
- [ ] 3.3 Integrate Merchant API client (upload valid rows only)
- [ ] 3.4 Handle partial upload failures with per-product results
- [ ] 3.5 Implement product status readback
- [ ] 3.6 Write tests with API client mocked (see `specs/merchant-publishing/spec.md`)

## Milestone 4 — Ads Optimization

- [ ] 4.1 Integrate Google Ads API client + auth
- [ ] 4.2 Implement campaign create/update (Shopping / Performance Max)
- [ ] 4.3 Implement metrics retrieval (impressions, clicks, CTR, conv, cost)
- [ ] 4.4 Implement rule-based optimization (pause/down-bid, flag low-CTR titles)
- [ ] 4.5 Implement dry-run mode
- [ ] 4.6 Write tests with API client mocked (see `specs/ads-optimization/spec.md`)

## Cross-cutting

- [ ] X.1 Document free vs paid setup (`requirements.md`) and link from project README
- [ ] X.2 Add integration smoke test against test/sandbox accounts (opt-in, skipped in CI)
- [ ] X.3 Ensure no credentials are committed (gitignore + secret scanning)
