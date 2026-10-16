# UdeSA Computer Vision — Automated Google Ads Positioning

A Computer Vision project at Universidad de San Andrés (UdeSA). The system tags product images with AI-generated metadata and automates end-to-end positioning on Google Ads: from raw image to live campaign.

## Demo

<video src="https://github.com/user-attachments/assets/6f47bd3f-877b-4ed3-bb6f-0f218e168935" controls width="720"></video>

---

## What it does

```
Image URL
   │
   ▼
┌─────────────────────────────────┐
│  Deep Tagger API                │  ← existing
│  TinyVGG · K-Means · BLIP · LLM│
│  product_type, tags, title,     │
│  description                    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Milestone 1 — Product Enrichment│  ← positioner/enrichment/
│  gender, material, pattern,      │
│  google_product_category,        │
│  structured title                │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Milestone 2 — Merchant Feed    │  ← positioner/feed/
│  Merge business fields + enrich │
│  Map to Google Merchant schema  │
│  Validate (GTIN, price format)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Milestone 3 — Publishing       │  ← positioner/publishing/
│  Secure credentials (env vars)  │
│  Upload to Google Merchant API  │
│  Read back approval status      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Milestone 4 — Ads Optimization │  ← positioner/ads/
│  Create Shopping campaigns      │
│  Pull metrics (CTR, ROAS, cost) │
│  Auto-pause/down-bid/flag rules │
└─────────────────────────────────┘
```

---

## Project structure

```
deep-tagger-api/        # FastAPI image-tagging backend
deep-tagger-ui/         # React frontend
img-puller/             # CSV image downloader utility
positioner/             # NEW — Google Ads positioning pipeline
  enrichment/           # M1: inferable field generation
  feed/                 # M2: Merchant schema mapping + validation
  publishing/           # M3: Merchant Center upload
  ads/                  # M4: campaign management + optimization loop
  tests/                # 26 unit tests (no GPU, no network required)
upgrade/                # Specs, proposal, and synthetic test data
  data/                 # 1,500 synthetic products (pants/shoes/tops CSVs)
  specs/                # OpenSpec per milestone
```

---

## Existing components

### Deep Tagger API
FastAPI backend that chains four models on a single image URL:
- **TinyVGG CNN** — classifies product type (tops / pants / shoes)
- **ResNet multilabel** — extracts fit, season, style tags
- **K-Means** — dominant color extraction
- **BLIP transformer** — vision-language title generation
- **Claude LLM** — product description generation

Endpoint: `POST /predict-image` → `{ product_type, title, description, tags }`

See [`deep-tagger-api/README.md`](deep-tagger-api/README.md) for setup.

### Deep Tagger UI
React application to submit image URLs and view AI-generated tags with overlay preview.

See [`deep-tagger-ui/README.md`](deep-tagger-ui/README.md) for setup.

### Image Puller
Python CLI that downloads product images from tagged CSVs and records local paths.

See [`img-puller/README.md`](img-puller/README.md) for setup.

---

## Google Ads Pipeline (`positioner/`)

### Milestone 1 — Product Enrichment

Fills the fields that are **inferable** from the image and existing tags, without requiring business data.

| Field | Source |
|-------|--------|
| `gender` | ResNet multilabel classifier |
| `material` | ResNet multilabel classifier |
| `pattern` | ResNet multilabel classifier |
| `google_product_category` | Mapping table from `product_type` |
| `title` | Structured: `Brand + Gender + Product + Attr + Color + Size` |
| `description` | Claude LLM |

```python
from positioner.enrichment.field_generator import enrich_product

row = {
    "image_url": "https://example.com/product.jpg",
    "brand": "StyleCo",
    "size": "M",
}
result = enrich_product(row)
# result.gender, result.material, result.google_product_category ...
# result.status → "ok" | "partial" | "needs_category_review"
```

### Milestone 2 — Merchant Feed

Merges enriched + business fields, maps them to the Google Merchant product schema, and validates every row.

**Business fields (must come from the user's CSV):** `brand`, `price`, `gtin`, `availability`, `link`, `condition`

**Validations:** GTIN-13 checksum · price format (`"25.32 USD"`) · required fields

```python
from positioner.feed.schema_mapper import map_to_merchant, validate_business_fields
from positioner.feed.validator import validate_row
from positioner.feed.report import build_report

validate_business_fields(business_row)           # raises ValueError if missing
product = map_to_merchant(enriched, business_row, row_id="123")
result  = validate_row(product)                  # ValidationResult(valid, reasons)
report  = build_report([result, ...])            # valid_count + invalid_count == N
```

### Milestone 3 — Merchant Publishing

Uploads the validated feed to Google Merchant Center via the Merchant API. Credentials come exclusively from environment variables — never from the repository.

**Required environment variables:**
```bash
GOOGLE_SERVICE_ACCOUNT_PATH=/path/to/key.json
MERCHANT_ID=123456789
MERCHANT_ENV=test          # or "production"
```

```python
from positioner.publishing.credential_loader import load_credentials
from positioner.publishing.environment import get_environment
from positioner.publishing.merchant_client import MerchantClient

creds  = load_credentials()          # reads env vars, fails fast if missing
env    = get_environment()           # defaults to TEST
client = MerchantClient(creds, env)
results = client.upload_feed(report, products)   # skips invalid rows
statuses = client.get_product_statuses(ids)      # approved/disapproved/pending
```

### Milestone 4 — Ads Optimization

Creates Google Ads campaigns linked to the Merchant Center account, pulls performance metrics, and runs a rule-based optimization loop.

**Optimization rules:**
- CTR < 1% over the window → **pause** product + flag title for rewrite
- Dry-run mode (default) → computes actions without mutating anything

```python
from positioner.ads.campaign_manager import CampaignManager, CampaignConfig
from positioner.ads.metrics_fetcher import MetricsFetcher
from positioner.ads.optimizer import run_optimization, OptimizationConfig

manager = CampaignManager(ads_customer_id="...", environment=env)
result  = manager.create_or_update(CampaignConfig(name="Summer Campaign", merchant_id="..."))

fetcher = MetricsFetcher(ads_customer_id="...", environment=env)
metrics = fetcher.get_metrics(product_ids, date_from="2026-10-01", date_to="2026-10-15")

actions = run_optimization(
    metrics,
    OptimizationConfig(ctr_threshold=0.01, dry_run=True),  # dry_run=True by default
    manager,
)
# actions[i].action → "pause" | "down_bid" | "flag_title"
# actions[i].needs_title_rewrite → True/False
```

---

## Running the tests

No GPU, no network, no credentials required — all external calls are mocked.

```bash
# From the workspace root
python -m unittest positioner.tests.test_enrichment -v   # M1 — 6 tests
python -m unittest positioner.tests.test_feed -v         # M2 — 7 tests
python -m unittest positioner.tests.test_publishing -v   # M3 — 6 tests
python -m unittest positioner.tests.test_ads -v          # M4 — 7 tests

# All 26 at once
python -m unittest discover -s positioner/tests -v
```

---

## Testing against Google (sandbox — free)

Full end-to-end validation is possible at zero cost using Google's test accounts.

**Setup (one time, ~15 min):**
1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Merchant API** and **Google Ads API**
3. Create a **test account** in [Google Merchant Center](https://merchants.google.com)
4. Create a **test account** in [Google Ads](https://ads.google.com)
5. Create a **service account**, download the JSON key
6. Set the environment variables (never commit the key file)

**Test with synthetic data:**
```bash
# 1,500 ready-made products (pants / shoes / tops)
ls upgrade/data/
# pants_google_500.csv  shoes_google_500.csv  tops_google_500.csv

# Regenerate any time
python3 upgrade/generate_synthetic_data.py
```

**Cost:** $0 — test accounts don't serve real ads or charge.

---

## Free vs paid

| | Free (sandbox) | Paid (production) |
|---|---|---|
| Merchant Center | Test account | Real verified account |
| Google Ads | Test account (no real ads) | Active account with billing |
| Metrics | Zero/simulated | Real CTR, ROAS, conversions |
| Data | Synthetic CSVs OK | Real GTINs, prices, active links |

See [`upgrade/requirements.md`](upgrade/requirements.md) for full details.

---

## Authors

Jorge Federico Flores — jfflores90@gmail.com

Nilayvi Barrientos nilayvi@gmail.com

Nicolás Velázquez nicoj.velazquez@gmail.com

Ruben Ortigoza ortigozaruben@hotmail.com
