"""Rule-based optimization loop: compute or apply actions from product metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from positioner.ads.campaign_manager import CampaignManager
from positioner.ads.metrics_fetcher import ProductMetrics


@dataclass
class OptimizationAction:
    """A single optimization action for one product."""
    product_id: str
    action: str          # "pause" | "down_bid" | "flag_title"
    reason: str
    needs_title_rewrite: bool = False


@dataclass
class OptimizationConfig:
    """Thresholds and safety flags for the optimization loop."""
    ctr_threshold: float = 0.01
    min_impressions: int = 100
    dry_run: bool = True          # defaults True for safety


def run_optimization(
    metrics: List[ProductMetrics],
    config: OptimizationConfig,
    campaign_manager: CampaignManager,
) -> List[OptimizationAction]:
    """Apply rule-based optimization actions derived from product metrics.

    Rules applied per product:
        if impressions >= config.min_impressions and ctr < config.ctr_threshold:
            action = "pause", needs_title_rewrite = True

    Dry-run behaviour:
        config.dry_run=True  → compute and return actions; NO campaign_manager
                               methods are called.
        config.dry_run=False → compute actions AND call campaign_manager to apply
                               them (stub raises NotImplementedError; tests only
                               exercise dry_run=True or inject a mock manager).

    Args:
        metrics:          list of ProductMetrics from MetricsFetcher.get_metrics().
        config:           thresholds and dry-run flag.
        campaign_manager: CampaignManager instance (unused in dry-run mode).

    Returns:
        List[OptimizationAction] — one entry per product that triggered a rule.
    """
    actions: List[OptimizationAction] = []

    for m in metrics:
        if m.impressions >= config.min_impressions and m.ctr < config.ctr_threshold:
            actions.append(
                OptimizationAction(
                    product_id=m.product_id,
                    action="pause",
                    reason=(
                        f"ctr={m.ctr:.4f} below threshold={config.ctr_threshold} "
                        f"with {m.impressions} impressions"
                    ),
                    needs_title_rewrite=True,
                )
            )

    if not config.dry_run:
        # In live mode, apply via campaign_manager (stubs in tests will raise).
        for action in actions:
            campaign_manager.pause_product(action.product_id)  # type: ignore[attr-defined]

    return actions
