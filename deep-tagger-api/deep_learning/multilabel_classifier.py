from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

TORCH_STATE_DIR = Path(__file__).resolve().parent / "torch_state/baseline_v1"
PRODUCT_TYPES = ["tops", "shoes", "pants"]

# Map product_type_classifier class names → multilabel checkpoint keys
_PRODUCT_TYPE_MAP: Dict[str, str] = {
    "T-shirt/top": "tops",
    "Shirt": "tops",
    "Pullover": "tops",
    "Pants": "pants",
    "Shoes": "shoes",
    "Sandal": "shoes",
    "Ankle boot": "shoes",
}


class MultilabelClassifier(nn.Module):
    def __init__(self, n_logits: int):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _build_transform(mean: List[float], std: List[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _load_model(product_type: str):
    """Load a single product-type checkpoint + label metadata. Returns None if files are missing."""
    ckpt_path = TORCH_STATE_DIR / f"multilabel_classifier_{product_type}_v1.pth"
    labels_path = TORCH_STATE_DIR / f"multilabel_classifier_{product_type}_v1.labels.json"

    if not ckpt_path.exists() or not labels_path.exists():
        print(f"[multilabel] WARNING: checkpoint not found for '{product_type}', skipping")
        return None

    with labels_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model = MultilabelClassifier(n_logits=meta["num_logits"])
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()

    transform = _build_transform(meta["normalize_mean"], meta["normalize_std"])

    return {"model": model, "meta": meta, "transform": transform}


# Eagerly load all available checkpoints at import time.
_models: Dict[str, dict] = {}
for _pt in PRODUCT_TYPES:
    _loaded = _load_model(_pt)
    if _loaded is not None:
        _models[_pt] = _loaded


def predict(img_array: np.ndarray, product_type: str) -> Dict[str, str]:
    """Return ``{attribute_name: predicted_class}`` for the given product type.

    Falls back to an empty dict if no checkpoint is available.
    """
    mapped_type = _PRODUCT_TYPE_MAP.get(product_type, product_type)
    entry = _models.get(mapped_type)
    if entry is None:
        return {}

    model = entry["model"]
    meta = entry["meta"]
    transform = entry["transform"]

    image = Image.fromarray(img_array).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.inference_mode():
        logits = model(tensor)

    predictions: Dict[str, str] = {}
    for group in meta["groups"]:
        group_logits = logits[0, group["start"]:group["end"]]
        predicted_idx = group_logits.argmax().item()
        predictions[group["name"]] = group["classes"][predicted_idx]

    return predictions