# Milestone 4 — Ads Optimization

Creates and updates Google Ads campaigns for the published products, pulls performance
metrics, and runs an optimization loop. This is the layer that turns the catalog into
an actual **automated positioner**.

## ADDED Requirements

### Requirement: Campaign creation
The system SHALL create a Shopping or Performance Max campaign linked to the
Merchant Center account via the Google Ads API.

#### Scenario: Create a campaign in the test account
- **WHEN** a published feed and a Google Ads test account are available
- **THEN** the system SHALL create a campaign referencing the Merchant account and
  return its campaign ID

#### Scenario: Reuse existing campaign
- **IF** a campaign with the same name already exists
- **THEN** the system SHALL update it instead of creating a duplicate

### Requirement: Metrics retrieval
The system SHALL retrieve performance metrics for the managed campaigns.

#### Scenario: Pull metrics for a date range
- **WHEN** metrics are requested for a date range
- **THEN** the system SHALL return impressions, clicks, CTR, conversions, and cost
  per campaign/product

#### Scenario: Test account returns no real metrics
- **WHEN** running against a test account
- **THEN** the system SHALL return zero/empty metrics and label them `test`

### Requirement: Optimization loop
The system SHALL apply rule-based optimization actions from the retrieved metrics.

#### Scenario: Pause underperforming product
- **IF** a product's CTR stays below a configured threshold over the window
- **THEN** the system SHALL pause or down-bid that product and record the action

#### Scenario: Flag low-CTR title for rewrite
- **IF** a product has impressions but CTR below threshold
- **THEN** the system SHALL flag its title for regeneration by Milestone 1

### Requirement: Dry-run safety
The system SHALL support a dry-run mode that computes optimization actions without
applying them.

#### Scenario: Dry-run makes no changes
- **WHEN** optimization runs in dry-run mode
- **THEN** the system SHALL output the planned actions and make no API mutations

## Tests

- `test_create_campaign_returns_id`
- `test_create_campaign_reuses_existing`
- `test_metrics_pull_returns_expected_fields`
- `test_metrics_test_account_labeled_test`
- `test_optimization_pauses_underperformer`
- `test_optimization_flags_low_ctr_title`
- `test_dry_run_makes_no_mutations`
