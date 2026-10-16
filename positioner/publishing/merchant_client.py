"""Merchant Center client: upload feed and read back product statuses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from positioner.feed.report import ValidationReport
from positioner.feed.schema_mapper import MerchantProduct
from positioner.publishing.credential_loader import Credentials
from positioner.publishing.environment import Environment


@dataclass
class UploadResult:
    """Outcome of uploading a single product."""
    product_id: str
    success: bool
    error: Optional[str] = None


@dataclass
class ProductStatus:
    """Approval/disapproval status for a single product."""
    product_id: str
    status: str
    issues: List[str] = field(default_factory=list)


class MerchantClient:
    """Orchestrates feed upload and status readback against Merchant Center."""

    def __init__(self, credentials: Credentials, environment: Environment) -> None:
        """Store config. Makes NO network calls.

        Args:
            credentials: loaded service-account path and merchant ID.
            environment: TEST or PRODUCTION.
        """
        self.credentials = credentials
        self.environment = environment

    def upload_feed(
        self,
        report: ValidationReport,
        products: List[MerchantProduct],
    ) -> List[UploadResult]:
        """Upload only valid products from the feed.

        Valid-vs-invalid matching rule:
            invalid_ids = {r.row_id for r in report.invalid_rows}
            A product is uploaded when product.id NOT IN invalid_ids.

        Continues on partial failure; returns a per-product UploadResult list
        covering only the valid products.

        Args:
            report:   ValidationReport produced by build_report().
            products: all MerchantProduct objects corresponding to the report.

        Returns:
            List[UploadResult] — one entry per valid product attempted.
        """
        invalid_ids = {r.row_id for r in report.invalid_rows}
        valid_products = [p for p in products if p.id not in invalid_ids]

        results: List[UploadResult] = []
        for product in valid_products:
            try:
                result = self._upload_product(product)
                results.append(result)
            except Exception as exc:
                results.append(
                    UploadResult(
                        product_id=product.id,
                        success=False,
                        error=str(exc),
                    )
                )
        return results

    def get_product_statuses(self, product_ids: List[str]) -> List[ProductStatus]:
        """Read back approval/disapproval status for each product.

        Args:
            product_ids: list of product IDs to query.

        Returns:
            List[ProductStatus] — one entry per product ID, in input order.
        """
        statuses: List[ProductStatus] = []
        for pid in product_ids:
            status = self._get_status(pid)
            statuses.append(status)
        return statuses

    def _upload_product(self, product: MerchantProduct) -> UploadResult:
        """Placeholder for the real Google Merchant API upload call.

        Tests mock this method to validate upload_feed orchestration.

        Raises:
            NotImplementedError: always — real implementation requires credentials + network.
        """
        raise NotImplementedError("Real Merchant API call not implemented")

    def _get_status(self, product_id: str) -> ProductStatus:
        """Placeholder for the real Google Merchant API status call.

        Tests mock this method to validate get_product_statuses orchestration.

        Raises:
            NotImplementedError: always — real implementation requires credentials + network.
        """
        raise NotImplementedError("Real Merchant API call not implemented")
