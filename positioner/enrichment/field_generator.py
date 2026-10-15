"""Orchestrates the full product-enrichment pipeline.

Imports deep-tagger-api modules at module level (same style as deep-tagger-api/main.py).
In production the caller must add the deep-tagger-api/ directory to sys.path before
importing this module.  In tests every import below is intercepted via sys.modules
pre-injection so no GPU, network, or model-file access ever occurs.
"""
from __future__ import annotations

# ── deep-tagger-api imports (satisfied via sys.path or sys.modules injection) ──
from aux_functions.auxiliary import download_image  # noqa: F401 (re-exported for mocking)
from machine_learning import k_means
from transformer import blip_transformer
from deep_learning import product_type_classifier
from deep_learning import multilabel_classifier
from llm import claude_client

# ── positioner imports ─────────────────────────────────────────────────────────
from positioner.enrichment.models import EnrichedProduct
from positioner.enrichment.title_builder import build_title
from positioner.enrichment.category_mapper import map_category

# Attribute keys returned by the multilabel classifier, checked in priority order.
_ATTR_KEYS = ["fit_silhouette", "fit", "style", "pocket_details"]


def enrich_product(row: dict) -> EnrichedProduct:
    """Enrich a single product row with image-derived and text-derived fields.

    Args:
        row: dict matching pants_google_500.csv schema.  Required key: 'image_url'.
             Optional keys used: 'brand', 'size', 'color', 'gender',
             'material', 'pattern'.

    Returns:
        EnrichedProduct with status 'ok', 'partial', or 'needs_category_review'.
    """
    # ── 1. Download image ──────────────────────────────────────────────────────
    status = "ok"
    img_array = None
    try:
        img_array = download_image(row.get("image_url", ""))
    except Exception:
        status = "partial"

    # ── 2. Initialise image-derived locals ────────────────────────────────────
    product_type = None
    ml_tags: dict = {}
    color_names: set = set()
    blip_title = None

    # ── 3. Run image-based models (skipped when download failed) ─────────────
    if img_array is not None:
        product_type = product_type_classifier.predict(img_array)
        ml_tags = multilabel_classifier.predict(img_array, product_type)
        color_names, _ = k_means.predict_k_colors(img_array)
        blip_title = blip_transformer.predict_product_title(img_array)

    # ── 4. Resolve final field values (image model wins; row is fallback) ─────
    gender = ml_tags.get("gender") or row.get("gender")
    material = ml_tags.get("material") or row.get("material")
    pattern = ml_tags.get("pattern") or row.get("pattern")
    color = (
        ", ".join(sorted(color_names)) if color_names else None
    ) or row.get("color")

    # ── 5. Attribute for structured title ─────────────────────────────────────
    attribute = next((ml_tags[k] for k in _ATTR_KEYS if k in ml_tags), None)

    # ── 6. Build structured title ─────────────────────────────────────────────
    title = build_title(
        brand=row.get("brand"),
        gender=gender,
        product=product_type or blip_title,
        attribute=attribute,
        color=color,
        size=row.get("size"),
    )

    # ── 7. Map Google category ────────────────────────────────────────────────
    google_category = None
    if product_type is not None:
        google_category, cat_status = map_category(product_type)
        if cat_status == "needs_category_review" and status == "ok":
            status = "needs_category_review"

    # ── 8. Generate description via Claude ────────────────────────────────────
    tags = {k: v for k, v in ml_tags.items() if v is not None}
    if color:
        tags["color"] = color

    description = claude_client.generate_product_description(
        product_type=product_type or "unknown",
        title=title,
        tags=tags,
    )

    # ── 9. Return enriched product ────────────────────────────────────────────
    return EnrichedProduct(
        gender=gender,
        material=material,
        pattern=pattern,
        google_product_category=google_category,
        title=title,
        description=description,
        status=status,
    )
