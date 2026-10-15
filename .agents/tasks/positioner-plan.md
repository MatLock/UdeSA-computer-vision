# Implementation Plan — `positioner/` package (Milestone 1: Product Enrichment)

> **Scope**: ADD-ONLY. Zero modifications to `deep-tagger-api/`, `deep-tagger-ui/`,
> `img-puller/`, or `upgrade/`. Every new file lives under `positioner/` at the
> workspace root.

---

## Verification

**Command run** (from `c:\Users\u645020\Documents\MIA\UdeSA-computer-vision-plus`):

```
python -m unittest positioner.tests.test_enrichment -v
```

**Full result:**

```
test_category_mapping_known_type (positioner.tests.test_enrichment.EnrichmentTests.test_category_mapping_known_type)
A known product type (mixed-case) maps to the correct taxonomy path. ... ok
test_category_mapping_unknown_type_flagged (positioner.tests.test_enrichment.EnrichmentTests.test_category_mapping_unknown_type_flagged)
An unknown product type leaves category unmapped and flags the row. ... ok
test_enrich_missing_image_returns_partial (positioner.tests.test_enrichment.EnrichmentTests.test_enrich_missing_image_returns_partial)
When the image cannot be downloaded the row is flagged 'partial'. ... ok
test_enrich_single_row_returns_inferable_fields (positioner.tests.test_enrichment.EnrichmentTests.test_enrich_single_row_returns_inferable_fields)
Happy-path: all image-based models succeed and populate every field. ... ok
test_title_degrades_without_brand_or_size (positioner.tests.test_enrichment.EnrichmentTests.test_title_degrades_without_brand_or_size)
Missing parts are silently dropped; no placeholder text is inserted. ... ok
test_title_follows_structured_order (positioner.tests.test_enrichment.EnrichmentTests.test_title_follows_structured_order)
Title parts appear in Brand→Gender→Product→Attribute→Color→Size order. ... ok
----------------------------------------------------------------------
Ran 6 tests in 0.002s
OK
```

All 6 tests passed. Exit confirmed via `&&` chain printing `EXIT_OK`.

---

## Source inventory (read during exploration)

### Confirmed function signatures

All symbols below were verified against the actual source files.

| Symbol | File (relative to `deep-tagger-api/`) | Actual signature | Return type |
|---|---|---|---|
| `predict` | `deep_learning/product_type_classifier.py` | `predict(img_array: np.ndarray) -> str` | One of 10 CLASS_NAMES: `'T-shirt/top'`, `'Pants'`, `'Pullover'`, `'Dress'`, `'Coat'`, `'Sandal'`, `'Shirt'`, `'Shoes'`, `'Bag'`, `'Ankle boot'` |
| `predict` | `deep_learning/multilabel_classifier.py` | `predict(img_array: np.ndarray, product_type: str) -> Dict[str, str]` | Attribute dict e.g. `{'gender': 'female', 'material': 'Silk', 'pattern': 'Striped'}` |
| `predict_k_colors` | `machine_learning/k_means.py` | `predict_k_colors(img_array: np.ndarray) -> tuple[set, list]` | `(set_of_color_name_strings, list_of_rgb_lists)` e.g. `({'black', 'navy blue'}, [[0,0,0]])` |
| `predict_product_title` | `transformer/blip_transformer.py` | `predict_product_title(img_array: np.ndarray) -> str` | Caption string e.g. `"pants"` |
| `generate_product_description` | `llm/claude_client.py` | `generate_product_description(product_type: str, title: str, tags: dict) -> str` | Description string |
| `download_image` | `aux_functions/auxiliary.py` | `download_image(image_url: str) -> np.ndarray` | RGBA numpy array; raises `requests.HTTPError` or any `Exception` on network/HTTP failure |

**No name discrepancies found.** All symbols named in the task prompt match the real source exactly.

**Mock patch paths** (how tests reference each symbol after field_generator imports them):

| What to mock | Patch target string |
|---|---|
| Image download | `positioner.enrichment.field_generator.download_image` |
| Product type classifier | `positioner.enrichment.field_generator.product_type_classifier` (module object) |
| Multilabel classifier | `positioner.enrichment.field_generator.multilabel_classifier` (module object) |
| K-means color extractor | `positioner.enrichment.field_generator.k_means` (module object) |
| BLIP title generator | `positioner.enrichment.field_generator.blip_transformer` (module object) |
| Claude description | `positioner.enrichment.field_generator.claude_client` (module object) |

### CSV columns (`pants_google_500.csv` — 21 columns)

```
id, image_url, pocket_details, fit_silhouette, season, relative_path,
brand, price, gtin, availability, link, condition,
gender, material, pattern, color, size, age_group,
google_product_category, title, description
```

Fields consumed by the enrichment pipeline:
- **Inputs from the row**: `image_url`, `brand`, `size`, `color` (used as fallback if image-based color fails)
- **Fields to generate/enrich**: `gender`, `material`, `pattern`, `google_product_category`, `title`, `description`
- `pocket_details`, `fit_silhouette` are also returned by the multilabel classifier for some product types and map to the `attribute` slot in the structured title.

### Dependency check (`requirements.txt`)

- `pydantic==2.13.2` ✓ present, but **NOT used** for `EnrichedProduct` (see design decision #1)
- `pytest` ✗ **NOT in `requirements.txt`** → tests must use `unittest.TestCase`; this is recorded as a discrepancy and handled in item 6
- `numpy`, `torch`, `torchvision`, `scikit-learn`, `transformers`, `anthropic`, `requests`, `Pillow` — all present; none imported directly by `positioner/` since the deep-tagger-api modules are mocked entirely in tests

---

## Design decisions

**1. `EnrichedProduct` as stdlib `@dataclass`, not `pydantic.BaseModel`.**
Pydantic is available in the project venv but positioner is required to be importable standalone. A plain `@dataclass` (Python 3.7+ stdlib) carries the same type-annotated interface with zero external deps, and it is the safer choice for a new, decoupled package.

**2. Module-level imports in `field_generator.py` + `sys.modules` pre-injection in tests.**
`field_generator.py` imports from deep-tagger-api at module level (matching the pattern in `main.py`). Tests pre-populate `sys.modules` with `MagicMock()` objects for every deep-tagger-api namespace *before* importing `field_generator`. Because Python's import machinery consults `sys.modules` first and skips file loading entirely when a module is already registered, no GPU, no model weights, and no network code ever executes during tests. This is simpler and more reliable than lazy-import gymnastics.

**3. Single `unittest.TestCase` class with all 6 required test methods.**
All 6 test methods go in one class so `python -m unittest positioner.tests.test_enrichment` discovers them in one command. They also work under `pytest positioner/tests/test_enrichment.py` without changes. `pytest` is only available if installed separately; tests never import it.

**4. Status priority: `needs_category_review` > `partial` > `ok`.**
A row starts at `ok`. Image download failure → `partial`. Unknown category → `needs_category_review` (only overrides `ok`; `partial` is preserved if category mapping is also unavailable because the image failed and product_type was never resolved). The spec's two failure scenarios are independent tests, so both outcomes are reachable in isolation.

**5. `attribute` slot in `build_title` is resolved from `ml_tags`.**
The multilabel classifier returns group-named attributes (e.g. `fit_silhouette`, `pocket_details`). The `attribute` parameter is populated by checking keys in priority order: `['fit_silhouette', 'fit', 'style', 'pocket_details']`. First key found in `ml_tags` wins; `None` if none present.

**6. `category_mapper` normalises to lowercase for lookup.**
`product_type_classifier.predict()` returns mixed-case strings like `'T-shirt/top'`, `'Ankle boot'`. `map_category()` calls `.lower()` on its input so callers never need to normalise.

---

## Directory layout to create

```
positioner/
├── __init__.py
├── enrichment/
│   ├── __init__.py
│   ├── models.py
│   ├── field_generator.py
│   ├── title_builder.py
│   └── category_mapper.py
└── tests/
    ├── __init__.py
    └── test_enrichment.py
```

---

## Implementation Plan

- [ ] 1. **Create the package skeleton** — three empty `__init__.py` files that make `positioner`,
      `positioner.enrichment`, and `positioner.tests` importable Python packages.

      Files:
      - `positioner/__init__.py` (empty)
      - `positioner/enrichment/__init__.py` (empty)
      - `positioner/tests/__init__.py` (empty)

      Verify: from the workspace root run
      ```
      python -c "import positioner; import positioner.enrichment; import positioner.tests; print('ok')"
      ```
      Expected output: `ok` with exit code 0.

---

- [ ] 2. **Create `positioner/enrichment/models.py`** — define `EnrichedProduct` as a stdlib
      `@dataclass`. All six content fields are `Optional[str]` defaulting to `None`.
      `status` is a plain `str` defaulting to `'ok'`; its docstring documents the three
      allowed values (`'ok'`, `'partial'`, `'needs_category_review'`).

      Exact class definition to implement:
      ```python
      from __future__ import annotations
      from dataclasses import dataclass, field
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
          status: str = 'ok'
      ```

      Files:
      - `positioner/enrichment/models.py`

      Verify:
      ```
      python -c "
      from positioner.enrichment.models import EnrichedProduct
      e = EnrichedProduct()
      assert e.status == 'ok'
      assert e.gender is None
      e2 = EnrichedProduct(status='partial', gender='female')
      assert e2.status == 'partial' and e2.gender == 'female'
      print('ok')
      "
      ```
      Expected output: `ok`.

---

- [ ] 3. **Create `positioner/enrichment/title_builder.py`** — implement `build_title()`.
      Parts are assembled strictly in the order `brand → gender → product → attribute → color → size`.
      Any part that is `None` or an empty string is silently dropped.
      Parts are joined with a single space. If all parts are absent the function returns `''`.
      No placeholder text is ever inserted.

      Exact public signature:
      ```python
      def build_title(
          brand: Optional[str] = None,
          gender: Optional[str] = None,
          product: Optional[str] = None,
          attribute: Optional[str] = None,
          color: Optional[str] = None,
          size: Optional[str] = None,
      ) -> str:
      ```

      Files:
      - `positioner/enrichment/title_builder.py`

      Verify:
      ```
      python -c "
      from positioner.enrichment.title_builder import build_title
      assert build_title(brand='Nike', gender='Men', product='Pants', attribute='Slim Fit', color='black', size='L') == 'Nike Men Pants Slim Fit black L'
      assert build_title(gender='Women', product='Shoes', color='white') == 'Women Shoes white'
      assert build_title(brand='Nike', color='black') == 'Nike black'
      assert build_title() == ''
      assert build_title(brand='', size=None, color='red') == 'red'
      print('ok')
      "
      ```
      Expected output: `ok`.

---

- [ ] 4. **Create `positioner/enrichment/category_mapper.py`** — implement `map_category()`.
      Uses an internal `CATEGORY_MAP` dict (all keys lowercase). Normalises the input to
      lowercase before lookup. Returns `(category_str, 'ok')` on a hit and
      `(None, 'needs_category_review')` on a miss.

      `CATEGORY_MAP` must include at minimum:

      ```python
      CATEGORY_MAP = {
          # FashionMNIST class names (lowercase)
          't-shirt/top':  'Apparel & Accessories > Clothing > Tops',
          'shirt':        'Apparel & Accessories > Clothing > Tops',
          'pullover':     'Apparel & Accessories > Clothing > Tops',
          'pants':        'Apparel & Accessories > Clothing > Pants',
          'dress':        'Apparel & Accessories > Clothing > Dresses',
          'coat':         'Apparel & Accessories > Clothing > Outerwear',
          'sandal':       'Apparel & Accessories > Shoes',
          'shoes':        'Apparel & Accessories > Shoes',
          'ankle boot':   'Apparel & Accessories > Shoes',
          'bag':          'Apparel & Accessories > Handbags, Wallets & Cases',
          # multilabel_classifier normalised aliases
          'tops':         'Apparel & Accessories > Clothing > Tops',
      }
      ```

      Exact public signature:
      ```python
      def map_category(product_type: str) -> tuple:
          """Map an internal product_type string to a Google Product Taxonomy path.

          Args:
              product_type: raw string from product_type_classifier.predict() or
                            the multilabel key. Case-insensitive.

          Returns:
              (category_path, status) where status is 'ok' or 'needs_category_review'.
              category_path is None when status is 'needs_category_review'.
          """
      ```

      Files:
      - `positioner/enrichment/category_mapper.py`

      Verify:
      ```
      python -c "
      from positioner.enrichment.category_mapper import map_category
      cat, s = map_category('Pants')
      assert cat == 'Apparel & Accessories > Clothing > Pants' and s == 'ok', cat
      cat2, s2 = map_category('pants')   # already lowercase
      assert s2 == 'ok'
      cat3, s3 = map_category('widget')
      assert cat3 is None and s3 == 'needs_category_review'
      print('ok')
      "
      ```
      Expected output: `ok`.

---

- [ ] 5. **Create `positioner/enrichment/field_generator.py`** — implement `enrich_product()`.
      Imports at module level (same style as `deep-tagger-api/main.py`):

      ```python
      from aux_functions.auxiliary import download_image
      from machine_learning import k_means
      from transformer import blip_transformer
      from deep_learning import product_type_classifier
      from deep_learning import multilabel_classifier
      from llm import claude_client

      from positioner.enrichment.models import EnrichedProduct
      from positioner.enrichment.title_builder import build_title
      from positioner.enrichment.category_mapper import map_category
      ```

      Public function signature:
      ```python
      def enrich_product(row: dict) -> EnrichedProduct:
          """Enrich a single product row with image-derived and text-derived fields.

          Args:
              row: dict matching pants_google_500.csv schema. Required key: 'image_url'.
                   Optional keys used: 'brand', 'size', 'color', 'gender',
                   'material', 'pattern'.

          Returns:
              EnrichedProduct with status 'ok', 'partial', or 'needs_category_review'.
          """
      ```

      **Full pipeline logic** (implement in this exact order):

      ```
      1.  status = 'ok'
          img_array = None
          try:
              img_array = download_image(row.get('image_url', ''))
          except Exception:
              status = 'partial'

      2.  product_type = None
          ml_tags = {}
          color_names = set()
          blip_title = None

      3.  if img_array is not None:
              product_type = product_type_classifier.predict(img_array)
              ml_tags = multilabel_classifier.predict(img_array, product_type)
              color_names, _ = k_means.predict_k_colors(img_array)
              blip_title = blip_transformer.predict_product_title(img_array)

      4.  # Resolve final field values (image model wins over row, row is fallback)
          gender   = ml_tags.get('gender')   or row.get('gender')
          material = ml_tags.get('material') or row.get('material')
          pattern  = ml_tags.get('pattern')  or row.get('pattern')
          color    = (', '.join(sorted(color_names)) if color_names
                      else None) or row.get('color')

      5.  # Attribute for title (fit/style qualifier from multilabel output)
          _ATTR_KEYS = ['fit_silhouette', 'fit', 'style', 'pocket_details']
          attribute = next((ml_tags[k] for k in _ATTR_KEYS if k in ml_tags), None)

      6.  # Build structured title
          title = build_title(
              brand=row.get('brand'),
              gender=gender,
              product=product_type or blip_title,
              attribute=attribute,
              color=color,
              size=row.get('size'),
          )

      7.  # Map Google category
          google_category = None
          if product_type is not None:
              google_category, cat_status = map_category(product_type)
              if cat_status == 'needs_category_review' and status == 'ok':
                  status = 'needs_category_review'

      8.  # Build tags dict for Claude (filter None values)
          tags = {k: v for k, v in ml_tags.items() if v is not None}
          if color:
              tags['color'] = color

          description = claude_client.generate_product_description(
              product_type=product_type or 'unknown',
              title=title,
              tags=tags,
          )

      9.  return EnrichedProduct(
              gender=gender,
              material=material,
              pattern=pattern,
              google_product_category=google_category,
              title=title,
              description=description,
              status=status,
          )
      ```

      Files:
      - `positioner/enrichment/field_generator.py`

      Verify: runs as part of item 6 (the tests import and exercise this module).

---

- [ ] 6. **Create `positioner/tests/test_enrichment.py`** — all 6 required tests.

      **Critical constraint**: `pytest` is NOT in `requirements.txt`. Use only
      `unittest.TestCase`. Running `python -m unittest positioner.tests.test_enrichment`
      from the workspace root must discover and pass all 6 tests without pytest installed.

      **File structure** (implement in this exact order):

      ```python
      # ── 1. sys.path: make positioner importable from workspace root ──────────────
      import sys
      import os
      _WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
      if _WS_ROOT not in sys.path:
          sys.path.insert(0, _WS_ROOT)

      # ── 2. sys.modules pre-injection (BEFORE any positioner import) ──────────────
      # Prevents real deep-tagger-api modules (and their GPU/model/network code)
      # from ever being loaded. Python's import machinery uses sys.modules as a
      # cache: if the entry already exists, the file is never executed.
      from unittest.mock import MagicMock

      _MOCK_MODULES = [
          'deep_learning',
          'deep_learning.product_type_classifier',
          'deep_learning.multilabel_classifier',
          'machine_learning',
          'machine_learning.k_means',
          'transformer',
          'transformer.blip_transformer',
          'llm',
          'llm.claude_client',
          'aux_functions',
          'aux_functions.auxiliary',
      ]
      for _mod in _MOCK_MODULES:
          sys.modules.setdefault(_mod, MagicMock())

      # ── 3. Now safe to import positioner modules ──────────────────────────────────
      import unittest
      from positioner.enrichment.models import EnrichedProduct
      from positioner.enrichment.title_builder import build_title
      from positioner.enrichment.category_mapper import map_category
      import positioner.enrichment.field_generator as _fg


      # ── 4. Test class ─────────────────────────────────────────────────────────────
      class EnrichmentTests(unittest.TestCase):

          def setUp(self):
              """Reset mock call history before each test."""
              for _mod in _MOCK_MODULES:
                  sys.modules[_mod].reset_mock()

          # ------------------------------------------------------------------ #
          #  test_enrich_single_row_returns_inferable_fields                    #
          # ------------------------------------------------------------------ #
          def test_enrich_single_row_returns_inferable_fields(self):
              # Arrange
              _fg.download_image.return_value = MagicMock()           # fake img_array
              _fg.product_type_classifier.predict.return_value = 'Pants'
              _fg.multilabel_classifier.predict.return_value = {
                  'gender': 'female', 'material': 'Denim', 'pattern': 'Solid'
              }
              _fg.k_means.predict_k_colors.return_value = ({'black'}, [[0, 0, 0]])
              _fg.blip_transformer.predict_product_title.return_value = 'pants'
              _fg.claude_client.generate_product_description.return_value = 'Great pants.'

              row = {'image_url': 'http://example.com/img.jpg',
                     'brand': 'StyleCo', 'size': 'M'}

              # Act
              result = _fg.enrich_product(row)

              # Assert – all inferable fields are populated
              self.assertEqual(result.gender, 'female')
              self.assertEqual(result.material, 'Denim')
              self.assertEqual(result.pattern, 'Solid')
              self.assertIsNotNone(result.google_product_category)
              self.assertIsNotNone(result.title)
              self.assertEqual(result.description, 'Great pants.')
              self.assertEqual(result.status, 'ok')

          # ------------------------------------------------------------------ #
          #  test_enrich_missing_image_returns_partial                          #
          # ------------------------------------------------------------------ #
          def test_enrich_missing_image_returns_partial(self):
              # Arrange – image download fails
              _fg.download_image.side_effect = Exception('network error')
              _fg.claude_client.generate_product_description.return_value = 'fallback'

              row = {'image_url': 'http://bad.url/img.jpg',
                     'brand': 'B', 'gender': 'male'}

              # Act
              result = _fg.enrich_product(row)

              # Assert
              self.assertEqual(result.status, 'partial')
              # gender falls back to the value in the row
              self.assertEqual(result.gender, 'male')
              # category not mapped (product_type was never determined)
              self.assertIsNone(result.google_product_category)

          # ------------------------------------------------------------------ #
          #  test_title_follows_structured_order                                #
          # ------------------------------------------------------------------ #
          def test_title_follows_structured_order(self):
              title = build_title(
                  brand='Nike', gender='Men', product='Pants',
                  attribute='Slim Fit', color='black', size='L'
              )
              self.assertEqual(title, 'Nike Men Pants Slim Fit black L')

          # ------------------------------------------------------------------ #
          #  test_title_degrades_without_brand_or_size                          #
          # ------------------------------------------------------------------ #
          def test_title_degrades_without_brand_or_size(self):
              title = build_title(gender='Women', product='Shoes', color='white')
              self.assertEqual(title, 'Women Shoes white')
              # No placeholder text, no 'None' literals
              self.assertNotIn('None', title)
              self.assertNotIn('N/A', title)
              self.assertGreater(len(title.strip()), 0)

          # ------------------------------------------------------------------ #
          #  test_category_mapping_known_type                                   #
          # ------------------------------------------------------------------ #
          def test_category_mapping_known_type(self):
              cat, status = map_category('Pants')      # mixed-case input
              self.assertEqual(cat, 'Apparel & Accessories > Clothing > Pants')
              self.assertEqual(status, 'ok')

          # ------------------------------------------------------------------ #
          #  test_category_mapping_unknown_type_flagged                         #
          # ------------------------------------------------------------------ #
          def test_category_mapping_unknown_type_flagged(self):
              cat, status = map_category('widget')
              self.assertIsNone(cat)
              self.assertEqual(status, 'needs_category_review')


      if __name__ == '__main__':
          unittest.main()
      ```

      Files:
      - `positioner/tests/test_enrichment.py`

      Verify:
      ```
      python -m unittest positioner.tests.test_enrichment
      ```
      Run from the workspace root (`c:\Users\u645020\Documents\MIA\UdeSA-computer-vision-plus`).
      Expected output: `Ran 6 tests in <Xs> OK` with no failures or errors.

---

## Implementation order

Items must be implemented in sequence because each item depends on the previous:

```
1 (skeleton) → 2 (models) → 3 (title_builder) → 4 (category_mapper)
→ 5 (field_generator, imports 2+3+4) → 6 (tests, imports 2+3+4+5)
```

After item 6 passes, the positioner package is complete for Milestone 1.

---

## Known constraints & discrepancies

| Constraint | Detail |
|---|---|
| `pytest` not in `requirements.txt` | Tests use `unittest.TestCase` only. If pytest is installed in the venv, `pytest positioner/tests/` also works, but it is not required. |
| `positioner/` standalone importability | `field_generator.py` imports from `deep_learning`, `machine_learning`, `transformer`, `llm`, and `aux_functions` — all relative to `deep-tagger-api/`. In production the caller must add `deep-tagger-api/` to `sys.path`. In tests this is bypassed entirely via `sys.modules` injection. |
| No new pip dependencies | `positioner/` uses only Python stdlib (`dataclasses`, `typing`, `unittest`) plus modules already in `requirements.txt`. No new packages are introduced. |
| Tests are GPU/network/file free | Every call to a model or network function is intercepted by a `MagicMock`. No `.pth` checkpoint files, no HTTP calls, no GPU needed. |
