"""Tests for positioner.publishing (Milestone 3 — Merchant Publishing).

All network and credential calls are mocked at the method level.
No real Google auth, no real env vars, no real network.

Run: python -m unittest positioner.tests.test_publishing -v
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
from unittest.mock import patch, MagicMock  # noqa: E402

from positioner.publishing.environment import Environment, get_environment  # noqa: E402
from positioner.publishing.credential_loader import (  # noqa: E402
    Credentials,
    CredentialError,
    load_credentials,
)
from positioner.publishing.merchant_client import (  # noqa: E402
    MerchantClient,
    UploadResult,
    ProductStatus,
)
from positioner.feed.schema_mapper import MerchantProduct  # noqa: E402
from positioner.feed.validator import ValidationResult  # noqa: E402
from positioner.feed.report import build_report  # noqa: E402


# ── 3. Test class ─────────────────────────────────────────────────────────────
class PublishingTests(unittest.TestCase):

    # ------------------------------------------------------------------ #
    #  test_credentials_load_from_env                                     #
    # ------------------------------------------------------------------ #
    def test_credentials_load_from_env(self) -> None:
        """Credentials are populated from GOOGLE_SERVICE_ACCOUNT_PATH and MERCHANT_ID."""
        with patch.dict(os.environ, {
            'GOOGLE_SERVICE_ACCOUNT_PATH': '/fake/service_account.json',
            'MERCHANT_ID': 'test-merchant-42',
        }):
            creds = load_credentials()

        self.assertEqual(creds.service_account_path, '/fake/service_account.json')
        self.assertEqual(creds.merchant_id, 'test-merchant-42')

    # ------------------------------------------------------------------ #
    #  test_credentials_missing_fails_fast                                #
    # ------------------------------------------------------------------ #
    def test_credentials_missing_fails_fast(self) -> None:
        """CredentialError is raised before any network call when env vars are absent."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CredentialError):
                load_credentials()

    # ------------------------------------------------------------------ #
    #  test_environment_defaults_to_test                                  #
    # ------------------------------------------------------------------ #
    def test_environment_defaults_to_test(self) -> None:
        """get_environment() returns Environment.TEST when MERCHANT_ENV is not set."""
        with patch.dict(os.environ, {}, clear=True):
            env = get_environment()
        self.assertEqual(env, Environment.TEST)

    # ------------------------------------------------------------------ #
    #  test_upload_only_valid_rows                                        #
    # ------------------------------------------------------------------ #
    def test_upload_only_valid_rows(self) -> None:
        """upload_feed calls _upload_product exactly K times for K valid rows."""
        # 2 valid rows, 1 invalid row
        validation_results = [
            ValidationResult(row_id='r1', valid=True,  reasons=[]),
            ValidationResult(row_id='r2', valid=True,  reasons=[]),
            ValidationResult(row_id='r3', valid=False, reasons=['gtin_format']),
        ]
        report = build_report(validation_results)

        products = [
            MerchantProduct(id='r1'),
            MerchantProduct(id='r2'),
            MerchantProduct(id='r3'),
        ]

        creds = Credentials(
            service_account_path='/fake/key.json',
            merchant_id='merchant-1',
        )
        client = MerchantClient(creds, Environment.TEST)

        with patch.object(
            client,
            '_upload_product',
            return_value=UploadResult(product_id='r1', success=True),
        ) as mock_upload:
            client.upload_feed(report, products)
            self.assertEqual(mock_upload.call_count, 2)

    # ------------------------------------------------------------------ #
    #  test_upload_partial_failure_reported                               #
    # ------------------------------------------------------------------ #
    def test_upload_partial_failure_reported(self) -> None:
        """upload_feed continues after a per-product failure and reports both results."""
        validation_results = [
            ValidationResult(row_id='r1', valid=True, reasons=[]),
            ValidationResult(row_id='r2', valid=True, reasons=[]),
        ]
        report = build_report(validation_results)

        products = [
            MerchantProduct(id='r1'),
            MerchantProduct(id='r2'),
        ]

        creds = Credentials(
            service_account_path='/fake/key.json',
            merchant_id='merchant-1',
        )
        client = MerchantClient(creds, Environment.TEST)

        def _upload_side_effect(product: MerchantProduct) -> UploadResult:
            if product.id == 'r1':
                return UploadResult(product_id='r1', success=True)
            raise Exception("API rejected product")

        with patch.object(client, '_upload_product', side_effect=_upload_side_effect):
            upload_results = client.upload_feed(report, products)

        self.assertEqual(len(upload_results), 2)

        success = next(r for r in upload_results if r.product_id == 'r1')
        failure = next(r for r in upload_results if r.product_id == 'r2')

        self.assertTrue(success.success)
        self.assertFalse(failure.success)
        self.assertIsNotNone(failure.error)

    # ------------------------------------------------------------------ #
    #  test_status_readback_returns_statuses                              #
    # ------------------------------------------------------------------ #
    def test_status_readback_returns_statuses(self) -> None:
        """get_product_statuses returns one ProductStatus per requested ID."""
        creds = Credentials(
            service_account_path='/fake/key.json',
            merchant_id='merchant-1',
        )
        client = MerchantClient(creds, Environment.TEST)

        expected = ProductStatus(
            product_id='p1',
            status='approved',
            issues=[],
        )

        with patch.object(client, '_get_status', return_value=expected) as mock_get:
            results = client.get_product_statuses(['p1'])
            mock_get.assert_called_once_with('p1')

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].product_id, 'p1')
        self.assertEqual(results[0].status, 'approved')


if __name__ == '__main__':
    unittest.main()
