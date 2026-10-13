# Upgrade: Automated Google Ads Positioning

This folder contains the OpenSpec change proposal to extend the UdeSA Computer Vision
project from a **product-attribute generator** into an **automated Google positioning
pipeline** (Google Merchant Center feed + Google Ads campaign management).

## Contents

| File | Purpose |
|---|---|
| `proposal.md` | Why, what changes, and impact of the upgrade |
| `tasks.md` | Implementation checklist, grouped by the 4 milestones |
| `requirements.md` | Free (sandbox/test) vs Paid (production) requirements |
| `specs/product-enrichment/spec.md` | Milestone 1 — content enrichment capability (ADDED) |
| `specs/merchant-feed/spec.md` | Milestone 2 — Google Merchant feed capability (ADDED) |
| `specs/merchant-publishing/spec.md` | Milestone 3 — publishing / sync capability (ADDED) |
| `specs/ads-optimization/spec.md` | Milestone 4 — campaign & optimization capability (ADDED) |

## The 4 Milestones

```
Milestone 1 · Product Enrichment   → fill inferable fields from image + tags
Milestone 2 · Merchant Feed        → map to Google schema + validate feed
Milestone 3 · Merchant Publishing  → credentials + upload feed to Merchant Center
Milestone 4 · Ads Optimization     → campaigns + metrics + optimization loop
```

Each milestone is independently testable and builds on the previous one.

## Conventions

- All requirements use EARS-style `SHALL` statements.
- All capabilities are introduced as `## ADDED Requirements` (greenfield; nothing is modified).
- Every milestone ships with its own test scenarios defined via `#### Scenario:` blocks.
