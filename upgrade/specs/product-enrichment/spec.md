# Milestone 1 — Product Enrichment

Fills the product fields that are **inferable** from the image and existing tags,
reusing the current CV/LLM pipeline. Prepares each product so it can later be mapped
to the Google Merchant schema.

## ADDED Requirements

### Requirement: Inferable field generation
The system SHALL generate inferable Google-relevant fields for a product from its
image and known tags, without requiring business data.

#### Scenario: Enrich a single product row
- **WHEN** a product row with `image_url` and known tags is submitted for enrichment
- **THEN** the system SHALL produce `gender`, `material`, `pattern`,
  `google_product_category`, an improved `title`, and a `description`

#### Scenario: Missing image is handled gracefully
- **IF** the `image_url` cannot be downloaded
- **THEN** the system SHALL skip image-derived fields and still return the fields
  derivable from text tags, flagging the row as `partial`

### Requirement: Structured title format
The system SHALL build titles following the Google Shopping convention
`Brand + Gender + Product + Attribute + Color + Size` when the inputs are available.

#### Scenario: Title assembled from available parts
- **WHEN** `brand`, `gender`, `product_type`, `color` are available
- **THEN** the generated `title` SHALL include those parts in the specified order

#### Scenario: Title degrades when parts are missing
- **IF** `brand` or `size` is unavailable
- **THEN** the title SHALL be assembled from the remaining available parts without
  inserting placeholder text

### Requirement: Google category mapping
The system SHALL map the internal `product_type` to a valid
`google_product_category` value.

#### Scenario: Known product type maps to a category
- **WHEN** the internal `product_type` is `pants`
- **THEN** `google_product_category` SHALL be set to the corresponding Google
  taxonomy path (e.g. `Apparel & Accessories > Clothing > Pants`)

#### Scenario: Unknown product type is flagged
- **IF** the internal `product_type` has no mapping
- **THEN** the row SHALL be flagged `needs_category_review` and left unmapped

## Tests

- `test_enrich_single_row_returns_inferable_fields`
- `test_enrich_missing_image_returns_partial`
- `test_title_follows_structured_order`
- `test_title_degrades_without_brand_or_size`
- `test_category_mapping_known_type`
- `test_category_mapping_unknown_type_flagged`
