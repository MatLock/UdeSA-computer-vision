"""Data model for a product row after enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnrichedProduct:
    """Represents a product row after enrichment.

    status values:
      'ok'                    – all inferable fields resolved successfully
      'partial'               – image download failed; image-derived fields are None
      'needs_category_review' – product type is not in the Google category map
    """

    gender: Optional[str] = None
    material: Optional[str] = None
    pattern: Optional[str] = None
    google_product_category: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: str = "ok"
