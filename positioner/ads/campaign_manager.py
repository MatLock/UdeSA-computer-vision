"""Campaign management: create or update Google Ads Shopping campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from positioner.publishing.environment import Environment


@dataclass
class CampaignConfig:
    """Parameters for a Shopping campaign."""
    name: str
    merchant_id: str
    budget_micros: int = 1_000_000
    campaign_type: str = "SHOPPING"


@dataclass
class CampaignResult:
    """Outcome of create_or_update.

    created=True  → a new campaign was created.
    created=False → an existing campaign was updated (reused).
    """
    campaign_id: str
    created: bool


class CampaignManager:
    """Orchestrates campaign creation and updates via the Google Ads API."""

    def __init__(self, ads_customer_id: str, environment: Environment) -> None:
        """Store config. Makes NO network calls.

        Args:
            ads_customer_id: Google Ads customer ID string.
            environment: TEST or PRODUCTION.
        """
        self.ads_customer_id = ads_customer_id
        self.environment = environment

    def create_or_update(self, config: CampaignConfig) -> CampaignResult:
        """Create a new campaign or update the existing one with the same name.

        Logic:
            existing_id = self._find_campaign(config.name)
            if existing_id is not None:
                self._update_campaign(existing_id, config)
                return CampaignResult(campaign_id=existing_id, created=False)
            else:
                new_id = self._create_campaign(config)
                return CampaignResult(campaign_id=new_id, created=True)

        Args:
            config: campaign parameters.

        Returns:
            CampaignResult with the campaign ID and whether it was newly created.
        """
        existing_id = self._find_campaign(config.name)
        if existing_id is not None:
            self._update_campaign(existing_id, config)
            return CampaignResult(campaign_id=existing_id, created=False)
        new_id = self._create_campaign(config)
        return CampaignResult(campaign_id=new_id, created=True)

    def _find_campaign(self, name: str) -> Optional[str]:
        """Return the campaign ID for an existing campaign with this name, or None.

        Tests mock this method via patch.object or direct assignment.

        Raises:
            NotImplementedError: always — real implementation requires network.
        """
        raise NotImplementedError("Real Google Ads API call not implemented")

    def _create_campaign(self, config: CampaignConfig) -> str:
        """Create a new campaign and return its assigned ID.

        Tests mock this method via patch.object or direct assignment.

        Raises:
            NotImplementedError: always — real implementation requires network.
        """
        raise NotImplementedError("Real Google Ads API call not implemented")

    def _update_campaign(self, campaign_id: str, config: CampaignConfig) -> None:
        """Apply config updates to an existing campaign.

        Tests mock this method via patch.object or direct assignment.

        Raises:
            NotImplementedError: always — real implementation requires network.
        """
        raise NotImplementedError("Real Google Ads API call not implemented")
