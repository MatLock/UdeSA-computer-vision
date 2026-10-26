"""Regression tests for the multilabel checkpoint metadata (*.labels.json).

multilabel_classifier.predict() slices the logits with the `start`/`end` of each
group and silently skips a product type when its checkpoint is missing. These
tests read only the JSON files, so they run without torch, a GPU or network:

    python -m unittest discover -s deep-tagger-api/tests
"""
import json
import unittest
from pathlib import Path

TORCH_STATE_DIR = Path(__file__).resolve().parent.parent / "deep_learning" / "torch_state" / "baseline_v1"

# Attribute groups documented in deep-tagger-api/README.md.
EXPECTED_GROUPS = {
    "tops": ["neck_style", "fit_silhouette", "season"],
    "shoes": ["wearing_occasion", "style_silhouette", "season"],
    "pants": ["pocket_details", "fit_silhouette", "season"],
}


def load_meta(product_type: str) -> dict:
    path = TORCH_STATE_DIR / f"multilabel_classifier_{product_type}_v1.labels.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class MultilabelLabelsTests(unittest.TestCase):

    def test_checkpoint_and_labels_exist(self):
        """Every product type has its .pth and .labels.json, so predict() does not skip it."""
        for product_type in EXPECTED_GROUPS:
            with self.subTest(product_type=product_type):
                self.assertTrue((TORCH_STATE_DIR / f"multilabel_classifier_{product_type}_v1.pth").is_file())
                self.assertTrue((TORCH_STATE_DIR / f"multilabel_classifier_{product_type}_v1.labels.json").is_file())

    def test_product_type_matches_file(self):
        for product_type in EXPECTED_GROUPS:
            with self.subTest(product_type=product_type):
                self.assertEqual(load_meta(product_type)["product_type"], product_type)

    def test_groups_are_the_documented_attributes(self):
        for product_type, expected in EXPECTED_GROUPS.items():
            with self.subTest(product_type=product_type):
                names = [group["name"] for group in load_meta(product_type)["groups"]]
                self.assertEqual(names, expected)

    def test_groups_tile_the_logits(self):
        """Groups are contiguous, start at 0 and end at num_logits, one logit per class."""
        for product_type in EXPECTED_GROUPS:
            with self.subTest(product_type=product_type):
                meta = load_meta(product_type)
                position = 0
                for group in meta["groups"]:
                    self.assertEqual(group["start"], position, group["name"])
                    self.assertEqual(group["end"] - group["start"], len(group["classes"]), group["name"])
                    position = group["end"]
                self.assertEqual(position, meta["num_logits"])

    def test_classes_are_unique_within_each_group(self):
        for product_type in EXPECTED_GROUPS:
            for group in load_meta(product_type)["groups"]:
                with self.subTest(product_type=product_type, group=group["name"]):
                    self.assertEqual(len(group["classes"]), len(set(group["classes"])))

    def test_rare_classes_were_dropped_at_twenty(self):
        """The Hito 2 sample README relies on the taxonomy dropping classes under 20 examples."""
        for product_type in EXPECTED_GROUPS:
            with self.subTest(product_type=product_type):
                self.assertEqual(load_meta(product_type)["min_samples_per_class"], 20)


if __name__ == "__main__":
    unittest.main()
