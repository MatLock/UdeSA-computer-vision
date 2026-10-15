"""Tests for positioner.enrichment.

All heavy calls (GPU models, network, file I/O) are intercepted via sys.modules
pre-injection before any positioner module is imported.  The suite runs under both
`python -m unittest positioner.tests.test_enrichment` and `pytest positioner/tests/`
with no additional dependencies beyond the Python stdlib.
"""

# ── 1. sys.path: make positioner importable from the workspace root ───────────
import sys
import os

_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

# ── 2. sys.modules pre-injection (BEFORE any positioner import) ──────────────
# Python's import machinery checks sys.modules first; if an entry is already
# present it skips loading the file entirely.  This prevents any GPU/model/
# network code in deep-tagger-api from ever executing.
from unittest.mock import MagicMock  # noqa: E402

_MOCK_MODULES = [
    "deep_learning",
    "deep_learning.product_type_classifier",
    "deep_learning.multilabel_classifier",
    "machine_learning",
    "machine_learning.k_means",
    "transformer",
    "transformer.blip_transformer",
    "llm",
    "llm.claude_client",
    "aux_functions",
    "aux_functions.auxiliary",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# ── 3. Now safe to import positioner modules ──────────────────────────────────
import unittest  # noqa: E402

from positioner.enrichment.models import EnrichedProduct  # noqa: E402
from positioner.enrichment.title_builder import build_title  # noqa: E402
from positioner.enrichment.category_mapper import map_category  # noqa: E402
import positioner.enrichment.field_generator as _fg  # noqa: E402


# ── 4. Test class ─────────────────────────────────────────────────────────────
class EnrichmentTests(unittest.TestCase):

    def setUp(self) -> None:
        """Reset all mock state (calls, return_values, side_effects) before each test."""
        for _mod in _MOCK_MODULES:
            # return_value=True and side_effect=True cascade recursively to children,
            # so child mocks like download_image are also fully reset.
            sys.modules[_mod].reset_mock(return_value=True, side_effect=True)

    # ------------------------------------------------------------------ #
    #  test_enrich_single_row_returns_inferable_fields                    #
    # ------------------------------------------------------------------ #
    def test_enrich_single_row_returns_inferable_fields(self) -> None:
        """Happy-path: all image-based models succeed and populate every field."""
        # Arrange
        _fg.download_image.return_value = MagicMock()          # fake img_array
        _fg.product_type_classifier.predict.return_value = "Pants"
        _fg.multilabel_classifier.predict.return_value = {
            "gender": "female",
            "material": "Denim",
            "pattern": "Solid",
        }
        _fg.k_means.predict_k_colors.return_value = ({"black"}, [[0, 0, 0]])
        _fg.blip_transformer.predict_product_title.return_value = "pants"
        _fg.claude_client.generate_product_description.return_value = "Great pants."

        row = {
            "image_url": "http://example.com/img.jpg",
            "brand": "StyleCo",
            "size": "M",
        }

        # Act
        result = _fg.enrich_product(row)

        # Assert — all inferable fields are populated
        self.assertEqual(result.gender, "female")
        self.assertEqual(result.material, "Denim")
        self.assertEqual(result.pattern, "Solid")
        self.assertIsNotNone(result.google_product_category)
        self.assertIsNotNone(result.title)
        self.assertEqual(result.description, "Great pants.")
        self.assertEqual(result.status, "ok")

    # ------------------------------------------------------------------ #
    #  test_enrich_missing_image_returns_partial                          #
    # ------------------------------------------------------------------ #
    def test_enrich_missing_image_returns_partial(self) -> None:
        """When the image cannot be downloaded the row is flagged 'partial'."""
        # Arrange — image download fails
        _fg.download_image.side_effect = Exception("network error")
        _fg.claude_client.generate_product_description.return_value = "fallback"

        row = {
            "image_url": "http://bad.url/img.jpg",
            "brand": "B",
            "gender": "male",
        }

        # Act
        result = _fg.enrich_product(row)

        # Assert
        self.assertEqual(result.status, "partial")
        # gender falls back to the value in the row
        self.assertEqual(result.gender, "male")
        # category not mapped (product_type was never determined)
        self.assertIsNone(result.google_product_category)

    # ------------------------------------------------------------------ #
    #  test_title_follows_structured_order                                #
    # ------------------------------------------------------------------ #
    def test_title_follows_structured_order(self) -> None:
        """Title parts appear in Brand→Gender→Product→Attribute→Color→Size order."""
        title = build_title(
            brand="Nike",
            gender="Men",
            product="Pants",
            attribute="Slim Fit",
            color="black",
            size="L",
        )
        self.assertEqual(title, "Nike Men Pants Slim Fit black L")

    # ------------------------------------------------------------------ #
    #  test_title_degrades_without_brand_or_size                          #
    # ------------------------------------------------------------------ #
    def test_title_degrades_without_brand_or_size(self) -> None:
        """Missing parts are silently dropped; no placeholder text is inserted."""
        title = build_title(gender="Women", product="Shoes", color="white")
        self.assertEqual(title, "Women Shoes white")
        self.assertNotIn("None", title)
        self.assertNotIn("N/A", title)
        self.assertGreater(len(title.strip()), 0)

    # ------------------------------------------------------------------ #
    #  test_category_mapping_known_type                                   #
    # ------------------------------------------------------------------ #
    def test_category_mapping_known_type(self) -> None:
        """A known product type (mixed-case) maps to the correct taxonomy path."""
        cat, status = map_category("Pants")
        self.assertEqual(cat, "Apparel & Accessories > Clothing > Pants")
        self.assertEqual(status, "ok")

    # ------------------------------------------------------------------ #
    #  test_category_mapping_unknown_type_flagged                         #
    # ------------------------------------------------------------------ #
    def test_category_mapping_unknown_type_flagged(self) -> None:
        """An unknown product type leaves category unmapped and flags the row."""
        cat, status = map_category("widget")
        self.assertIsNone(cat)
        self.assertEqual(status, "needs_category_review")


if __name__ == "__main__":
    unittest.main()
