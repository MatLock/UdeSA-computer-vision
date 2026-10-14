# Milestone 2 — Merchant Feed

Merges enriched (inferable) fields with user-supplied business fields, maps them to
the Google Merchant product schema, and validates every row before publishing.

## ADDED Requirements

### Requirement: Input CSV contract
The system SHALL define and enforce a CSV contract that separates business fields
(required from the user) from inferable fields (produced by Milestone 1).

#### Scenario: Business fields present
- **WHEN** the user CSV includes `price`, `gtin`, `availability`, `link`, `brand`
- **THEN** the system SHALL accept the row for mapping

#### Scenario: Missing business field rejected
- **IF** a required business field (e.g. `price`) is absent
- **THEN** the row SHALL be rejected with a clear error naming the missing field

### Requirement: Google Merchant schema mapping
The system SHALL map a merged product record to the Google Merchant product schema.

#### Scenario: Full apparel record maps completely
- **WHEN** a merged record has all required apparel fields
- **THEN** the output SHALL contain `id`, `title`, `description`, `link`,
  `image_link`, `price`, `availability`, `brand`, `condition`,
  `google_product_category`, `gender`, `age_group`, `size`, `color`, `material`

### Requirement: Feed validation
The system SHALL validate each mapped row against Google's required-field and format
rules before the feed is considered publishable.

#### Scenario: Valid row passes validation
- **WHEN** a mapped row satisfies all required fields and formats
- **THEN** the row SHALL be marked `valid`

#### Scenario: Invalid GTIN fails validation
- **IF** `gtin` fails checksum/format validation
- **THEN** the row SHALL be marked `invalid` with reason `gtin_format`

#### Scenario: Invalid price format fails validation
- **IF** `price` is not in `<number> <currency>` format
- **THEN** the row SHALL be marked `invalid` with reason `price_format`

### Requirement: Validation report
The system SHALL emit a summary report of valid vs invalid rows after validation.

#### Scenario: Report counts match
- **WHEN** a feed of N rows is validated
- **THEN** the report SHALL report `valid + invalid == N` and list each invalid
  row's reason

## Tests

- `test_csv_contract_accepts_complete_row`
- `test_csv_contract_rejects_missing_price`
- `test_schema_mapping_full_apparel_record`
- `test_validation_valid_row_passes`
- `test_validation_invalid_gtin`
- `test_validation_invalid_price_format`
- `test_validation_report_counts_consistent`
