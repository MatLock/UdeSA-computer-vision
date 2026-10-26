# Milestone 1 — Product Enrichment

Fills the product fields that are **inferable** from the image, reusing the current
CV/LLM pipeline, and carries through the fields the brand declares. Prepares each
product so it can later be mapped to the Google Merchant schema.

Inferred from the image: `google_product_category` (from the TinyVGG product type),
color (K-Means), the title attribute (multilabel classifier) and `description`
(Claude). Declared by the brand and never predicted: `gender`, `material`,
`pattern`, `size`. A declared color wins over the predicted one; both are kept.

## ADDED Requirements

### Requirement: Inferable field generation
The system SHALL generate the inferable Google-relevant fields for a product from
its image, and SHALL pass through the fields declared by the brand without
inventing them.

#### Scenario: Enrich a single product row
- **WHEN** a product row with `image_url` and the brand's declared fields is
  submitted for enrichment
- **THEN** the system SHALL produce `google_product_category`, `color`,
  `predicted_color`, a structured `title` and a `description`, and SHALL return
  the declared `gender`, `material` and `pattern` unchanged
- **TEST** `test_enrich_single_row_returns_inferable_fields`

#### Scenario: Missing image is handled gracefully
- **IF** the `image_url` cannot be downloaded
- **THEN** the system SHALL skip image-derived fields and still return the fields
  derivable from text tags, flagging the row as `partial`
- **TEST** `test_enrich_missing_image_returns_partial`

### Requirement: Structured title format
The system SHALL build titles following the Google Shopping convention
`Brand + Gender + Product + Attribute + Color + Size` when the inputs are available.

#### Scenario: Title assembled from available parts
- **WHEN** `brand`, `gender`, `product_type`, `color` are available
- **THEN** the generated `title` SHALL include those parts in the specified order
- **TEST** `test_title_follows_structured_order`

#### Scenario: Title degrades when parts are missing
- **IF** `brand` or `size` is unavailable
- **THEN** the title SHALL be assembled from the remaining available parts without
  inserting placeholder text
- **TEST** `test_title_degrades_without_brand_or_size`

### Requirement: Google category mapping
The system SHALL map the internal `product_type` to a valid
`google_product_category` value.

#### Scenario: Known product type maps to a category
- **WHEN** the internal `product_type` is `pants`
- **THEN** `google_product_category` SHALL be set to the corresponding Google
  taxonomy path (e.g. `Apparel & Accessories > Clothing > Pants`)
- **TEST** `test_category_mapping_known_type`

#### Scenario: Unknown product type is flagged
- **IF** the internal `product_type` has no mapping
- **THEN** the row SHALL be flagged `needs_category_review` and left unmapped
- **TEST** `test_category_mapping_unknown_type_flagged`

## Tests

- `test_enrich_single_row_returns_inferable_fields`
- `test_enrich_missing_image_returns_partial`
- `test_title_follows_structured_order`
- `test_title_degrades_without_brand_or_size`
- `test_category_mapping_known_type`
- `test_category_mapping_unknown_type_flagged`
