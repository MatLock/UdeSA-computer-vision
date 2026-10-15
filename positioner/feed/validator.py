"""Validates each MerchantProduct row against Google format rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from positioner.feed.schema_mapper import MerchantProduct

_PRICE_RE = re.compile(r'^\d+(\.\d+)?\s+[A-Z]{3}$')


@dataclass
class ValidationResult:
    """Outcome of validating a single product row."""
    row_id: str
    valid: bool
    reasons: List[str] = field(default_factory=list)


def validate_gtin(gtin: str) -> bool:
    """Return True if gtin is a well-formed GTIN-13 with a valid checksum.

    Algorithm (GS1 standard):
      - Must be exactly 13 ASCII digits.
      - Weights for first 12 digits alternate 1,3,1,3,... (position 0 → weight 1).
      - check_digit = (10 - (sum % 10)) % 10 must equal digits[12].
    """
    if not gtin or not gtin.isdigit() or len(gtin) != 13:
        return False
    digits = [int(ch) for ch in gtin]
    total = sum(
        d * (1 if i % 2 == 0 else 3)
        for i, d in enumerate(digits[:12])
    )
    expected_check = (10 - (total % 10)) % 10
    return expected_check == digits[12]


def validate_price_format(price: str) -> bool:
    """Return True if price matches '<number> <ISO-4217-currency>' format.

    Accepted examples: '25.32 USD', '10 EUR', '1000.00 GBP'.
    Rejected examples: '$25.32', '25.32usd', '25.32', 'USD 25.32'.
    """
    if not price:
        return False
    return bool(_PRICE_RE.match(price))


def validate_row(product: MerchantProduct) -> ValidationResult:
    """Validate a MerchantProduct and return a ValidationResult.

    Checks:
      - GTIN-13 checksum/format when gtin is present and non-empty.
      - Price format when price is present and non-empty.

    Args:
        product: a MerchantProduct (output of map_to_merchant).

    Returns:
        ValidationResult with valid=True if no issues found, else valid=False
        and reasons list containing one or more of: 'gtin_format', 'price_format'.
    """
    reasons: List[str] = []

    if product.gtin:
        if not validate_gtin(product.gtin):
            reasons.append('gtin_format')

    if product.price:
        if not validate_price_format(product.price):
            reasons.append('price_format')

    return ValidationResult(
        row_id=product.id,
        valid=len(reasons) == 0,
        reasons=reasons,
    )
