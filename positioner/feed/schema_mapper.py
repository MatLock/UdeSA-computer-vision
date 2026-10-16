"""Merges EnrichedProduct (inferable) + business dict (user-supplied) into MerchantProduct."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from positioner.enrichment.models import EnrichedProduct

# Required business fields — presence checked before mapping.
REQUIRED_BUSINESS_FIELDS: list[str] = [
    'price', 'gtin', 'availability', 'link', 'brand', 'condition',
]


@dataclass
class MerchantProduct:
    """Google Merchant Center product record (apparel schema)."""
    # Required identifiers
    id: str = ''
    # Content fields (from EnrichedProduct)
    title: Optional[str] = None
    description: Optional[str] = None
    google_product_category: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    # Business fields (from user CSV)
    link: Optional[str] = None
    image_link: Optional[str] = None
    price: Optional[str] = None
    availability: Optional[str] = None
    brand: Optional[str] = None
    condition: Optional[str] = None
    gtin: Optional[str] = None
    # Bonus apparel field
    pattern: Optional[str] = None


def validate_business_fields(business: dict) -> None:
    """Raise ValueError naming the first missing required business field.

    Args:
        business: dict of user-supplied CSV fields.

    Raises:
        ValueError: if any required field is absent or None.
    """
    for field_name in REQUIRED_BUSINESS_FIELDS:
        if not business.get(field_name):
            raise ValueError(
                f"Missing required business field: '{field_name}'"
            )


def map_to_merchant(
    enriched: EnrichedProduct,
    business: dict,
    row_id: str = '',
) -> MerchantProduct:
    """Merge an EnrichedProduct with business fields into a MerchantProduct.

    Args:
        enriched: output of M1 enrich_product().
        business: dict of user-supplied CSV fields.
        row_id:   identifier string for the row (e.g. CSV 'id' column).

    Returns:
        MerchantProduct with all 15+ Google Merchant apparel fields populated.
    """
    return MerchantProduct(
        id=row_id,
        title=enriched.title,
        description=enriched.description,
        google_product_category=enriched.google_product_category,
        gender=enriched.gender,
        age_group=business.get('age_group') or 'adult',
        size=business.get('size'),
        color=business.get('color'),
        material=enriched.material,
        link=business.get('link'),
        image_link=business.get('image_url'),
        price=business.get('price'),
        availability=business.get('availability'),
        brand=business.get('brand'),
        condition=business.get('condition'),
        gtin=business.get('gtin'),
        pattern=enriched.pattern,
    )
