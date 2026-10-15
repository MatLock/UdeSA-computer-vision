"""Builds a structured product title following the Google Shopping convention.

Order: Brand → Gender → Product → Attribute → Color → Size.
Parts that are None or empty are silently dropped; no placeholder text is inserted.
"""
from __future__ import annotations

from typing import Optional


def build_title(
    brand: Optional[str] = None,
    gender: Optional[str] = None,
    product: Optional[str] = None,
    attribute: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> str:
    """Assemble a structured product title from the supplied parts.

    Args:
        brand:     Brand name (e.g. 'Nike').
        gender:    Target gender (e.g. 'Men', 'Women').
        product:   Product type or name (e.g. 'Pants', 'Shoes').
        attribute: Style/fit qualifier (e.g. 'Slim Fit', 'Striped').
        color:     Primary colour(s) (e.g. 'black', 'navy blue').
        size:      Size label (e.g. 'L', 'XL').

    Returns:
        A single string with the available parts joined by spaces.
        Returns '' if every argument is None or an empty string.
    """
    parts = [brand, gender, product, attribute, color, size]
    filled = [p for p in parts if p]  # drop None and ""
    return " ".join(filled)
