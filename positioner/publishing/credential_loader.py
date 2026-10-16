"""Load Google Merchant credentials from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


class CredentialError(Exception):
    """Raised when required credential env vars are missing."""


@dataclass
class Credentials:
    """Google Merchant Center credentials loaded from environment."""
    service_account_path: str
    merchant_id: str


def load_credentials() -> Credentials:
    """Load credentials from environment variables.

    Reads GOOGLE_SERVICE_ACCOUNT_PATH and MERCHANT_ID.
    Raises CredentialError on the first missing variable.
    Makes NO Google auth calls — only loads and validates the config.

    Returns:
        Credentials dataclass populated from env vars.

    Raises:
        CredentialError: if GOOGLE_SERVICE_ACCOUNT_PATH or MERCHANT_ID is unset.
    """
    values: dict[str, str] = {}
    for var_name in ('GOOGLE_SERVICE_ACCOUNT_PATH', 'MERCHANT_ID'):
        value = os.environ.get(var_name)
        if not value:
            raise CredentialError(f"Missing env var: {var_name}")
        values[var_name] = value

    return Credentials(
        service_account_path=values['GOOGLE_SERVICE_ACCOUNT_PATH'],
        merchant_id=values['MERCHANT_ID'],
    )
