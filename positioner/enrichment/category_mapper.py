"""Maps internal product_type strings to Google Product Taxonomy paths."""
from __future__ import annotations

from typing import Optional, Tuple

# Keys are lowercase FashionMNIST class names and common aliases.
CATEGORY_MAP: dict[str, str] = {
    # FashionMNIST class names (lowercase)
    "t-shirt/top": "Apparel & Accessories > Clothing > Tops",
    "shirt": "Apparel & Accessories > Clothing > Tops",
    "pullover": "Apparel & Accessories > Clothing > Tops",
    "pants": "Apparel & Accessories > Clothing > Pants",
    "dress": "Apparel & Accessories > Clothing > Dresses",
    "coat": "Apparel & Accessories > Clothing > Outerwear",
    "sandal": "Apparel & Accessories > Shoes",
    "shoes": "Apparel & Accessories > Shoes",
    "ankle boot": "Apparel & Accessories > Shoes",
    "bag": "Apparel & Accessories > Handbags, Wallets & Cases",
    # multilabel_classifier normalised aliases
    "tops": "Apparel & Accessories > Clothing > Tops",
}


def map_category(product_type: str) -> Tuple[Optional[str], str]:
    """Map an internal product_type string to a Google Product Taxonomy path.

    Args:
        product_type: raw string from product_type_classifier.predict() or
                      the multilabel key.  Case-insensitive.

    Returns:
        (category_path, status) where status is 'ok' or 'needs_category_review'.
        category_path is None when status is 'needs_category_review'.
    """
    key = product_type.lower()
    category = CATEGORY_MAP.get(key)
    if category is not None:
        return category, "ok"
    return None, "needs_category_review"
