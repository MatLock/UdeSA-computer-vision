"""Tests for positioner.ads (Milestone 4 — Ads Optimization).

All Google Ads API calls are mocked at the method level via
unittest.mock.patch.object or direct attribute assignment.
No real network calls, no real credentials.

Run: python -m unittest positioner.tests.test_ads -v
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
from unittest.mock import MagicMock, patch  # noqa: E402

from positioner.publishing.environment import Environment  # noqa: E402
from positioner.ads.campaign_manager import (  # noqa: E402
    CampaignConfig,
    CampaignManager,
)
from positioner.ads.metrics_fetcher import (  # noqa: E402
    MetricsFetcher,
    ProductMetrics,
)
from positioner.ads.optimizer import (  # noqa: E402
    OptimizationConfig,
    run_optimization,
)


# ── 3. Test class ─────────────────────────────────────────────────────────────
class AdsTests(unittest.TestCase):

    # ------------------------------------------------------------------ #
    #  test_create_campaign_returns_id                                   #
    # ------------------------------------------------------------------ #
    def test_create_campaign_returns_id(self) -> None:
        """_find_campaign returns None → _create_campaign called → created=True."""
        manager = CampaignManager(
            ads_customer_id="cust-001",
            environment=Environment.TEST,
        )
        manager._find_campaign = lambda name: None          # no existing campaign
        manager._create_campaign = lambda config: "camp_123"

        config = CampaignConfig(name="test-camp", merchant_id="merch-1")
        result = manager.create_or_update(config)

        self.assertEqual(result.campaign_id, "camp_123")
        self.assertTrue(result.created)

    # ------------------------------------------------------------------ #
    #  test_create_campaign_reuses_existing                              #
    # ------------------------------------------------------------------ #
    def test_create_campaign_reuses_existing(self) -> None:
        """_find_campaign returns an id → _update_campaign called → created=False."""
        manager = CampaignManager(
            ads_customer_id="cust-001",
            environment=Environment.TEST,
        )
        manager._find_campaign = lambda name: "camp_existing"
        manager._update_campaign = lambda campaign_id, config: None

        config = CampaignConfig(name="existing-camp", merchant_id="merch-1")
        result = manager.create_or_update(config)

        self.assertEqual(result.campaign_id, "camp_existing")
        self.assertFalse(result.created)

    # ------------------------------------------------------------------ #
    #  test_metrics_pull_returns_expected_fields                         #
    # ------------------------------------------------------------------ #
    def test_metrics_pull_returns_expected_fields(self) -> None:
        """_fetch_metrics mocked → get_metrics returns all 5 numeric fields."""
        fetcher = MetricsFetcher(
            ads_customer_id="cust-001",
            environment=Environment.PRODUCTION,
        )
        fake_metric = ProductMetrics(
            product_id="p42",
            impressions=500,
            clicks=25,
            ctr=0.05,
            conversions=3.0,
            cost=12.50,
        )
        fetcher._fetch_metrics = lambda product_ids, date_from, date_to: [fake_metric]

        results = fetcher.get_metrics(["p42"], "2024-01-01", "2024-01-31")

        self.assertEqual(len(results), 1)
        m = results[0]
        self.assertEqual(m.product_id, "p42")
        self.assertEqual(m.impressions, 500)
        self.assertEqual(m.clicks, 25)
        self.assertAlmostEqual(m.ctr, 0.05)
        self.assertAlmostEqual(m.conversions, 3.0)
        self.assertAlmostEqual(m.cost, 12.50)

    # ------------------------------------------------------------------ #
    #  test_metrics_test_account_labeled_test                            #
    # ------------------------------------------------------------------ #
    def test_metrics_test_account_labeled_test(self) -> None:
        """Environment.TEST → zero-filled metrics returned, source='test'."""
        fetcher = MetricsFetcher(
            ads_customer_id="cust-001",
            environment=Environment.TEST,
        )
        # _fetch_metrics is NOT mocked — TEST env must bypass it entirely.
        results = fetcher.get_metrics(["p1"], "2024-01-01", "2024-01-31")

        self.assertEqual(len(results), 1)
        m = results[0]
        self.assertEqual(m.source, "test")
        self.assertEqual(m.impressions, 0)
        self.assertEqual(m.clicks, 0)
        self.assertAlmostEqual(m.ctr, 0.0)
        self.assertAlmostEqual(m.conversions, 0.0)
        self.assertAlmostEqual(m.cost, 0.0)

    # ------------------------------------------------------------------ #
    #  test_optimization_pauses_underperformer                           #
    # ------------------------------------------------------------------ #
    def test_optimization_pauses_underperformer(self) -> None:
        """impressions≥100, ctr<0.01 → one action with action='pause'."""
        metrics = [
            ProductMetrics(
                product_id="p1",
                impressions=200,
                clicks=1,
                ctr=0.005,
                conversions=0.0,
                cost=0.0,
            )
        ]
        config = OptimizationConfig(
            ctr_threshold=0.01,
            min_impressions=100,
            dry_run=True,
        )
        mock_manager = MagicMock()

        actions = run_optimization(metrics, config, mock_manager)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, "pause")

    # ------------------------------------------------------------------ #
    #  test_optimization_flags_low_ctr_title                             #
    # ------------------------------------------------------------------ #
    def test_optimization_flags_low_ctr_title(self) -> None:
        """Same underperformer → needs_title_rewrite is True."""
        metrics = [
            ProductMetrics(
                product_id="p1",
                impressions=200,
                clicks=1,
                ctr=0.005,
                conversions=0.0,
                cost=0.0,
            )
        ]
        config = OptimizationConfig(
            ctr_threshold=0.01,
            min_impressions=100,
            dry_run=True,
        )
        mock_manager = MagicMock()

        actions = run_optimization(metrics, config, mock_manager)

        self.assertTrue(actions[0].needs_title_rewrite)

    # ------------------------------------------------------------------ #
    #  test_dry_run_makes_no_mutations                                   #
    # ------------------------------------------------------------------ #
    def test_dry_run_makes_no_mutations(self) -> None:
        """dry_run=True → CampaignManager is never called."""
        metrics = [
            ProductMetrics(
                product_id="p1",
                impressions=200,
                clicks=1,
                ctr=0.005,
                conversions=0.0,
                cost=0.0,
            )
        ]
        config = OptimizationConfig(dry_run=True)
        mock_manager = MagicMock()

        run_optimization(metrics, config, mock_manager)

        mock_manager.assert_not_called()
        # Verify no attribute on the mock was accessed (no method calls)
        self.assertEqual(len(mock_manager.method_calls), 0)


if __name__ == '__main__':
    unittest.main()
