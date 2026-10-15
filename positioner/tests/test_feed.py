"""Tests for positioner.feed (Milestone 2 — Merchant Feed).

No mocks needed: the feed sub-package has zero ML/GPU/network dependencies.
Run: python -m unittest positioner.tests.test_feed -v
     (from workspace root c:\\Users\\u645020\\Documents\\MIA\\UdeSA-computer-vision-plus)
"""

# ── 1. sys.path: make positioner importable from workspace root ───────────────
import sys
import os

_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

# ── 2. imports ────────────────────────────────────────────────────────────────
import unittest  # noqa: E402

from positioner.enrichment.models import EnrichedProduct          # noqa: E402
from positioner.feed.schema_mapper import (                       # noqa: E402
    MerchantProduct,
    validate_business_fields,
    map_to_merchant,
)
from positioner.feed.validator import (                           # noqa: E402
    ValidationResult,
    validate_row,
)
from positioner.feed.report import ValidationReport, build_report  # noqa: E402


# ── 3. Helper fixtures ────────────────────────────────────────────────────────

def _complete_business() -> dict:
    """A business dict that satisfies all 6 required fields."""
    return {
        'price': '25.32 USD',
        'gtin': '4006381333931',   # valid GTIN-13
        'availability': 'in stock',
        'link': 'https://example.com/pants/1',
        'brand': 'StyleCo',
        'condition': 'new',
        'image_url': 'https://example.com/pants/1.jpg',
        'size': 'M',
        'color': 'black',
        'age_group': 'adult',
    }


def _complete_enriched() -> EnrichedProduct:
    """An EnrichedProduct with all inferable fields populated."""
    return EnrichedProduct(
        gender='female',
        material='Denim',
        pattern='Solid',
        google_product_category='Apparel & Accessories > Clothing > Pants',
        title='StyleCo female Pants Solid black M',
        description='Great pants.',
        status='ok',
    )


# ── 4. Test class ─────────────────────────────────────────────────────────────
class FeedTests(unittest.TestCase):

    # ------------------------------------------------------------------ #
    #  test_csv_contract_accepts_complete_row                             #
    # ------------------------------------------------------------------ #
    def test_csv_contract_accepts_complete_row(self) -> None:
        """A business dict with all required fields raises no exception."""
        business = _complete_business()
        # Should not raise
        try:
            validate_business_fields(business)
        except ValueError as exc:
            self.fail(f"validate_business_fields raised unexpectedly: {exc}")

    # ------------------------------------------------------------------ #
    #  test_csv_contract_rejects_missing_price                            #
    # ------------------------------------------------------------------ #
    def test_csv_contract_rejects_missing_price(self) -> None:
        """Missing 'price' raises ValueError naming the missing field."""
        business = _complete_business()
        del business['price']
        with self.assertRaises(ValueError) as ctx:
            validate_business_fields(business)
        self.assertIn('price', str(ctx.exception))

    # ------------------------------------------------------------------ #
    #  test_schema_mapping_full_apparel_record                            #
    # ------------------------------------------------------------------ #
    def test_schema_mapping_full_apparel_record(self) -> None:
        """All 15 required Google Merchant apparel fields are present and non-None."""
        enriched = _complete_enriched()
        business = _complete_business()
        product = map_to_merchant(enriched, business, row_id='row_1')

        required_fields = [
            'id', 'title', 'description', 'link', 'image_link',
            'price', 'availability', 'brand', 'condition',
            'google_product_category', 'gender', 'age_group',
            'size', 'color', 'material',
        ]
        for f in required_fields:
            value = getattr(product, f)
            self.assertIsNotNone(value, f"Field '{f}' is None on MerchantProduct")

    # ------------------------------------------------------------------ #
    #  test_validation_valid_row_passes                                   #
    # ------------------------------------------------------------------ #
    def test_validation_valid_row_passes(self) -> None:
        """A row with valid gtin and valid price is marked valid."""
        product = map_to_merchant(_complete_enriched(), _complete_business(), 'r1')
        result = validate_row(product)
        self.assertTrue(result.valid)
        self.assertEqual(result.reasons, [])

    # ------------------------------------------------------------------ #
    #  test_validation_invalid_gtin                                       #
    # ------------------------------------------------------------------ #
    def test_validation_invalid_gtin(self) -> None:
        """A row with a GTIN that fails checksum is marked invalid with 'gtin_format'."""
        business = _complete_business()
        business['gtin'] = '1234567890123'  # wrong check digit (expected 8, got 3)
        product = map_to_merchant(_complete_enriched(), business, 'r2')
        result = validate_row(product)
        self.assertFalse(result.valid)
        self.assertIn('gtin_format', result.reasons)

    # ------------------------------------------------------------------ #
    #  test_validation_invalid_price_format                               #
    # ------------------------------------------------------------------ #
    def test_validation_invalid_price_format(self) -> None:
        """A row whose price does not match '<number> <CURRENCY>' is marked invalid."""
        business = _complete_business()
        business['price'] = '$25.32'   # dollar sign — not accepted
        product = map_to_merchant(_complete_enriched(), business, 'r3')
        result = validate_row(product)
        self.assertFalse(result.valid)
        self.assertIn('price_format', result.reasons)

    # ------------------------------------------------------------------ #
    #  test_validation_report_counts_consistent                           #
    # ------------------------------------------------------------------ #
    def test_validation_report_counts_consistent(self) -> None:
        """valid_count + invalid_count == total; invalid_rows lists reasons."""
        results = [
            ValidationResult(row_id='a', valid=True,  reasons=[]),
            ValidationResult(row_id='b', valid=False, reasons=['gtin_format']),
            ValidationResult(row_id='c', valid=True,  reasons=[]),
            ValidationResult(row_id='d', valid=False, reasons=['price_format']),
        ]
        report = build_report(results)

        self.assertEqual(report.total, 4)
        self.assertEqual(report.valid_count + report.invalid_count, report.total)
        self.assertEqual(report.valid_count, 2)
        self.assertEqual(report.invalid_count, 2)
        self.assertEqual(len(report.invalid_rows), 2)

        invalid_reasons = [r.reasons for r in report.invalid_rows]
        self.assertIn(['gtin_format'], invalid_reasons)
        self.assertIn(['price_format'], invalid_reasons)


if __name__ == '__main__':
    unittest.main()
