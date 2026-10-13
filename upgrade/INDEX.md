# Upgrade Index

Complete reference for the **Automated Google Ads Positioning** upgrade proposal and implementation.

## 📋 Specification & Requirements

| Document | Purpose |
|---|---|
| **[README.md](README.md)** | Overview of the 4 milestones and folder structure |
| **[proposal.md](proposal.md)** | Why this upgrade, what changes, and impact analysis |
| **[requirements.md](requirements.md)** | FREE (sandbox/test) vs PAID (production) setup |
| **[tasks.md](tasks.md)** | Implementation checklist grouped by milestone |

## 🎯 Milestone Specifications (OpenSpec Format — ADDED mode)

All requirements use EARS syntax (`SHALL`) and include test scenarios.

### Milestone 1 — Product Enrichment
**File:** `specs/product-enrichment/spec.md`

Fills inferable fields (gender, material, pattern, title, category) from the image and existing tags.

- **Key Requirements:**
  - Inferable field generation
  - Structured title format (Brand + Gender + Product + Attr + Color + Size)
  - Google category mapping
- **Scenarios:** 6
- **Tests:** 6

### Milestone 2 — Merchant Feed
**File:** `specs/merchant-feed/spec.md`

Maps enriched + business fields to Google Merchant schema and validates the feed.

- **Key Requirements:**
  - Input CSV contract (business vs inferable)
  - Google Merchant schema mapping
  - Feed validation (GTIN, price format, etc.)
  - Validation report
- **Scenarios:** 7
- **Tests:** 7

### Milestone 3 — Merchant Publishing
**File:** `specs/merchant-publishing/spec.md`

Manages credentials, uploads validated feed to Merchant Center, reads back statuses.

- **Key Requirements:**
  - Credential management (secure loading)
  - Environment selection (test/prod)
  - Feed upload (valid rows only)
  - Status readback
- **Scenarios:** 6
- **Tests:** 6

### Milestone 4 — Ads Optimization
**File:** `specs/ads-optimization/spec.md`

Creates campaigns, retrieves metrics, runs optimization loop.

- **Key Requirements:**
  - Campaign creation / reuse
  - Metrics retrieval
  - Optimization loop (pause underperformers, flag low-CTR titles)
  - Dry-run mode
- **Scenarios:** 7
- **Tests:** 7

## 📊 Synthetic Test Data

**Location:** `data/`

| File | Purpose |
|---|---|
| **[data/README.md](data/README.md)** | Usage guide + schema overview |
| **[data/MANIFEST.md](data/MANIFEST.md)** | Detailed schema + data distribution |
| **data/pants_google_500.csv** | 500 pants products (complete schema) |
| **data/shoes_google_500.csv** | 500 shoes products (complete schema) |
| **data/tops_google_500.csv** | 500 tops products (complete schema) |

### Data Summary
- **Total products:** 1,500 (500 per type)
- **Columns:** 20 (original + business + inferable)
- **Size:** ~664 KB
- **Ready for:** All 4 milestones (enrichment → validation → publishing → optimization)

### Field Breakdown

**Original fields:** `id`, `image_url`, `relative_path`, product-type tags

**Business fields (synthetic but realistic):**
- `brand` (6 pool: StyleCo, FashionHub, TrendWear, ClassicStyle, ModernFit, ElegantLine)
- `price` (random 19.99–149.99 USD)
- `gtin` (generated GTIN-13)
- `availability` (always `in_stock` for demo)
- `link` (https://store.example.com/...)
- `condition` (always `new`)

**Inferable fields (generated):**
- `title` (structured format)
- `description` (2-3 sentence)
- `gender`, `material`, `pattern`, `color`, `size`, `age_group`
- `google_product_category` (mapped from product type)

## 🛠️ Code & Utilities

**File:** `generate_synthetic_data.py`

Python script to regenerate the test CSVs from source data.

```bash
# Regenerate all CSVs
python3 upgrade/generate_synthetic_data.py

# Output: upgrade/data/{pants,shoes,tops}_google_500.csv
```

## 🗺️ Integration Points

### With Existing Project

- **No modifications** to current modules (`deep-tagger-api`, `deep-tagger-ui`, `img-puller`)
- Reuses CV pipeline: TinyVGG, ResNet, BLIP, K-Means (input images + tags → enrichment)
- Extends the `img-puller` enrichment pattern
- New capability modules add to `deep-tagger-api` or a new `positioner` package

### With Google APIs

- **Merchant API** (replaces deprecated Content API)
- **Google Ads API** (management only; test accounts free, no billing)
- **Google Cloud project** (free to create and enable APIs)

## 🎯 Using This Upgrade

### For understanding the scope:
1. Read `proposal.md` (why/what/impact)
2. Read `requirements.md` (free vs paid setup)
3. Scan `tasks.md` (milestone breakdown)

### For implementation:
1. Pick a milestone (suggest Milestone 1 first — lowest risk, reuses existing models)
2. Read its `specs/*/spec.md`
3. Use `data/pants_google_500.csv` (or shoes/tops) as test input
4. Run the test scenarios; green means you're ready for the next milestone

### For testing against Google:
1. Follow `requirements.md` to set up free test accounts
2. Feed the CSV through each milestone
3. Validate in test/sandbox environments (no cost, no live ads)
4. Once confident, can pivot to production with real data

## 📞 Questions?

- **Why these 4 milestones?** Each is independently valuable and can be tested in isolation.
- **Why separate business vs inferable?** Business data (price, GTIN, link) must come from the user's
  catalog; inferable data (title, color, gender) can be generated by the pipeline. This distinction
  is enforced at Milestone 2 (Feed validation).
- **Can I skip milestones?** No — each builds on the previous. Milestone 2 needs Milestone 1 output;
  Milestone 3 needs Milestone 2 output; etc.
- **How do I know when a milestone is done?** All tests in its `spec.md` → Tests section pass.
