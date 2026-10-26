"""Data model for a product row after enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnrichedProduct:
    """Represents a product row after enrichment.

    Field origin:
      predicted from the image – google_product_category (from the TinyVGG
                                 product type), predicted_color (K-Means),
                                 title (attribute from the multilabel
                                 classifier) and description (Claude LLM)
      declared by the brand    – gender, material and pattern; no model
                                 predicts them, so they are passed through

    color is the brand's declared color when present and predicted_color
    otherwise; predicted_color always keeps the K-Means output (None when the
    image could not be downloaded), so both can be compared.

    status values:
      'ok'                    – all inferable fields resolved successfully
      'partial'               – image download failed; image-derived fields are None
      'needs_category_review' – product type is not in the Google category map
    """

    gender: Optional[str] = None
    material: Optional[str] = None
    pattern: Optional[str] = None
    color: Optional[str] = None
    predicted_color: Optional[str] = None
    google_product_category: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: str = "ok"
