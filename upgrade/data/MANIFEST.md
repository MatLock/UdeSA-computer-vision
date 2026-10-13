# Data Manifest

## Overview

Synthetic Google Ads product feed data for testing the positioner upgrade milestones.

- **Total files:** 3 CSVs (pants, shoes, tops)
- **Total rows:** 1,500 products (500 per type, plus headers)
- **Total size:** ~664 KB
- **Generated:** 2026-10-11
- **Source:** First 500 rows of each original product-type CSV + synthetic enrichment

## Files

```
upgrade/data/
├── README.md                 # Overview + usage guide
├── MANIFEST.md              # This file
├── pants_google_500.csv     # 500 pants products, complete schema
├── shoes_google_500.csv     # 500 shoes products, complete schema
└── tops_google_500.csv      # 500 tops products, complete schema
```

## Schema (all files)

Common fields across all three files:

| Field | Type | Example | Note |
|---|---|---|---|
| `id` | string | `11876597` | Original product ID |
| `image_url` | URL | `https://asset1.ruecdn.com/...` | Source image |
| `relative_path` | path | `images/pants/11876597.jpg` | Local image path |
| `brand` | string | `ModernFit` | Business field (synthetic) |
| `price` | string | `25.32 USD` | Business field (synthetic) |
| `gtin` | string | `0000011876597` | Business field (synthetic) |
| `availability` | enum | `in_stock` | Business field (always in_stock here) |
| `link` | URL | `https://store.example.com/...` | Business field (synthetic) |
| `condition` | enum | `new` | Business field (always new here) |
| `gender` | enum | `female` | Inferable field (synthetic) |
| `material` | string | `Silk` | Inferable field (synthetic) |
| `pattern` | string | `Striped` | Inferable field (synthetic) |
| `color` | string | `gray` | Inferable field (synthetic) |
| `size` | string | `XS` | Inferable field (synthetic) |
| `age_group` | enum | `adult` | Inferable field (always adult here) |
| `google_product_category` | string | `Apparel & Accessories > Clothing > Pants` | Inferable field (mapped) |
| `title` | string | `ModernFit Women's Gray Silk Loose Fit Pant Trousers` | Inferable field (structured) |
| `description` | string | `Comfortable loose fit...` | Inferable field (generated) |

### Product-type-specific fields

**Pants:**
- `pocket_details` — e.g., `Slant-Pocket`, `Pockets`, `Zipper-Pocket`
- `fit_silhouette` — e.g., `Loose-Fit-Pant`, `Regular-Fit-Pant`, `Slim-Fit-Pant`
- `season` — e.g., `Tropical-Vacation`, `Fall`, `Spring`

**Shoes:**
- `wearing_occasion` — e.g., `Day-Party`, `Casual`, `Formal`
- `style_silhouette` — e.g., `Pumps`, `Sneakers`, `Loafers`
- `season` — e.g., `Tropical-Vacation`, `Fall`, `Spring`

**Tops:**
- `neck_style` — e.g., `V-Neck`, `Crew-Neck`, `Turtleneck`
- `fit_silhouette` — e.g., `Relaxed-Fit`, `Fitted`, `Loose-Fit`
- `season` — e.g., `Fall`, `Spring`, `Tropical-Vacation`

## Data distribution

### Brands
Randomly selected: StyleCo, FashionHub, TrendWear, ClassicStyle, ModernFit, ElegantLine

### Prices
Random between 19.99 and 149.99 USD

### Materials
Cotton, Polyester, Cotton Blend, Denim, Linen, Wool, Silk

### Patterns
Solid, Striped, Checkered, Floral, Abstract, Geometric

### Colors
black, navy blue, white, gray, beige, khaki, blue, red, green

### Sizes
XS, S, M, L, XL, XXL

### Gender
female, male, unisex (randomly selected)

## Usage in milestones

| Milestone | CSV | Task |
|---|---|---|
| 1 — Product Enrichment | All 3 | Validate that enrichment fields are generated correctly |
| 2 — Merchant Feed | All 3 | Validate CSV contract + schema mapping + validation rules |
| 3 — Merchant Publishing | All 3 | Test upload to sandbox Merchant Center + status readback |
| 4 — Ads Optimization | All 3 | Test campaign creation + metrics + optimization loop |

## Regeneration

To regenerate these files (e.g., to create a different sample or update generation logic):

```bash
cd /path/to/project
python3 upgrade/generate_synthetic_data.py
```

The script reads from `img-puller/data/{pants,shoes,tops}_tags.csv` and outputs to `upgrade/data/`.
