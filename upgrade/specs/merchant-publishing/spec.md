# Milestone 3 — Merchant Publishing

Manages credentials and uploads the validated feed to a Merchant Center account
(test or production) via the Merchant API, then reads back each product's status.

## ADDED Requirements

### Requirement: Credential management
The system SHALL load Google credentials from a secure source and never from the
repository.

#### Scenario: Load service-account credentials
- **WHEN** a service-account key path and Merchant ID are provided via environment
- **THEN** the system SHALL authenticate successfully against the Merchant API

#### Scenario: Missing credentials fail fast
- **IF** no credentials are configured
- **THEN** the system SHALL abort before any network call with a clear error

### Requirement: Environment selection
The system SHALL allow targeting either a test/sandbox or a production Merchant
Center account.

#### Scenario: Default to test environment
- **WHEN** no environment is specified
- **THEN** the system SHALL target the test/sandbox account by default

### Requirement: Feed upload
The system SHALL upload only rows marked `valid` by Milestone 2 to Merchant Center.

#### Scenario: Upload valid rows
- **WHEN** a validated feed with K valid rows is published
- **THEN** the system SHALL submit exactly K products and skip invalid rows

#### Scenario: Partial failure is reported
- **IF** some products are rejected by the API during upload
- **THEN** the system SHALL continue with the rest and return a per-product result
  list

### Requirement: Status readback
The system SHALL read back the approval/disapproval status of each uploaded product.

#### Scenario: Read product statuses
- **WHEN** products have been uploaded
- **THEN** the system SHALL return each product's status
  (`approved` / `disapproved` / `pending`) and any issues reported by Google

## Tests

- `test_credentials_load_from_env`
- `test_credentials_missing_fails_fast`
- `test_environment_defaults_to_test`
- `test_upload_only_valid_rows`
- `test_upload_partial_failure_reported`
- `test_status_readback_returns_statuses`
