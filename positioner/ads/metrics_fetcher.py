"""Metrics retrieval: pull impression/click/conversion data from Google Ads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from positioner.publishing.environment import Environment


@dataclass
class ProductMetrics:
    """Performance metrics for a single product."""
    product_id: str
    impressions: int
    clicks: int
    ctr: float
    conversions: float
    cost: float
    source: str = "live"


class MetricsFetcher:
    """Retrieves product-level metrics from the Google Ads API."""

    def __init__(self, ads_customer_id: str, environment: Environment) -> None:
        """Store config. Makes NO network calls.

        Args:
            ads_customer_id: Google Ads customer ID string.
            environment: TEST or PRODUCTION.
        """
        self.ads_customer_id = ads_customer_id
        self.environment = environment

    def get_metrics(
        self,
        product_ids: List[str],
        date_from: str,
        date_to: str,
    ) -> List[ProductMetrics]:
        """Return metrics for the given products over the date window.

        When environment is Environment.TEST: bypasses _fetch_metrics entirely
        and returns zero-filled ProductMetrics with source="test" for every
        product_id in the input list.

        When environment is Environment.PRODUCTION: delegates to _fetch_metrics.

        Args:
            product_ids: list of product IDs to query.
            date_from:   start date string, e.g. "2024-01-01".
            date_to:     end date string,   e.g. "2024-01-31".

        Returns:
            List[ProductMetrics], one entry per product_id.
        """
        if self.environment == Environment.TEST:
            return [
                ProductMetrics(
                    product_id=pid,
                    impressions=0,
                    clicks=0,
                    ctr=0.0,
                    conversions=0.0,
                    cost=0.0,
                    source="test",
                )
                for pid in product_ids
            ]
        return self._fetch_metrics(product_ids, date_from, date_to)

    def _fetch_metrics(
        self,
        product_ids: List[str],
        date_from: str,
        date_to: str,
    ) -> List[ProductMetrics]:
        """Placeholder for the real Google Ads API metrics call.

        Tests mock this method via patch.object or direct assignment.

        Raises:
            NotImplementedError: always — real implementation requires network.
        """
        raise NotImplementedError("Real Google Ads API call not implemented")
