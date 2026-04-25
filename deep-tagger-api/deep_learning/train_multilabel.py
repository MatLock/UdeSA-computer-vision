"""Train a multi-label CNN classifier for tops / shoes / pants.

One architecture (ResNet18 from scratch, no pretrained weights), three
independent training runs — one checkpoint per product type.

Usage:
    python train_multilabel.py --product-type tops
    python train_multilabel.py --product-type shoes --device cuda
    python train_multilabel.py --product-type pants --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Schema: grouped attributes per product type.
# ---------------------------------------------------------------------------
SCHEMA: Dict[str, List[str]] = {
    "tops":  ["neck_style", "fit_silhouette", "season"],
    "shoes": ["wearing_occasion", "style_silhouette", "season"],
    "pants": ["pocket_details", "fit_silhouette", "season"],
}

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_BASE = REPO_ROOT / "img-puller" / "data"
CSV_PATHS = {
    "tops":  REPO_ROOT / "img-puller" / "data" / "tops_tags.csv",
    "shoes": REPO_ROOT / "img-puller" / "data" / "shoes_tags.csv",
    "pants": REPO_ROOT / "img-puller" / "data" / "pants_tags.csv",
}
OUT_DIR = Path(__file__).resolve().parent / "torch_state"
MIN_SAMPLES_PER_CLASS = 20


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
@dataclass
class LabelGroup:
    name: str
    classes: List[str]
    start: int  # slice start in the full multi-hot vector

    @property
    def end(self) -> int:
        return self.start + len(self.classes)


def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_exists(rel_path: str) -> bool:
    if not isinstance(rel_path, str) or not rel_path:
        return False
    p = IMG_BASE / rel_path.replace("\\", "/")
    return p.is_file()


def load_and_filter(
    product_type: str,
    max_samples: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[LabelGroup]]:
    """Load CSV, drop rows without images, drop rare classes, cap dataset size."""
    csv_path = CSV_PATHS[product_type]
    attrs = SCHEMA[product_type]

    # Use a dedup so the pants CSV's duplicated `relative_path` header doesn't break us.
    df = pd.read_csv(csv_path, usecols=lambda c: c in {"id", "relative_path", *attrs})
    df = df.loc[:, ~df.columns.duplicated()]
    total_rows = len(df)

    # Normalize path separators for Windows-authored CSVs.
    df["relative_path"] = df["relative_path"].astype(str).str.replace("\\", "/", regex=False)

    # Drop rows whose image file is missing on disk.
    mask_img = df["relative_path"].map(_image_exists)
    dropped_missing = int((~mask_img).sum())
    df = df[mask_img].reset_index(drop=True)
    rows_with_image = len(df)

    # Drop rare classes per attribute (replace rare value with NaN, then drop rows
    # that lost their only label in that attribute).
    groups: List[LabelGroup] = []
    cursor = 0
    nan_counts: Dict[str, int] = {}
    for attr in attrs:
        counts = df[attr].value_counts(dropna=True)
        keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
        keep.sort()
        nan_before = int(df[attr].isna().sum())
        df.loc[~df[attr].isin(keep), attr] = np.nan
        nan_counts[attr] = int(df[attr].isna().sum()) - nan_before
        groups.append(LabelGroup(name=attr, classes=keep, start=cursor))
        cursor += len(keep)

    # Drop rows that end up with no label at all.
    df = df.dropna(subset=attrs, how="all").reset_index(drop=True)
    rows_after_rare = len(df)
    dropped_rare = rows_with_image - rows_after_rare

    # Shuffle + cap.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    if max_samples and max_samples < len(df):
        df = df.iloc[:max_samples].reset_index(drop=True)
    rows_used = len(df)
    dropped_cap = rows_after_rare - rows_used

    print(f"[data] === {product_type} dataset summary ===")
    print(f"[data]   total rows in CSV                       : {total_rows}")
    print(f"[data]   dropped (image missing on disk)         : {dropped_missing}")
    print(f"[data]   dropped (all attrs were rare classes)   : {dropped_rare}")
    print(f"[data]   dropped (--max-samples cap, cap={max_samples}) : {dropped_cap}")
    print(f"[data]   rows used (will be split into train/val): {rows_used}")
    for attr, n in nan_counts.items():
        print(f"[data]   attr '{attr}': {n} cells NaN'd by rare-class filter")
    for g in groups:
        print(f"[data]   group {g.name}: {len(g.classes)} classes")
    return df, groups


class MultilabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        groups: List[LabelGroup],
        transform: transforms.Compose,
    ):
        self.df = df.reset_index(drop=True)
        self.groups = groups
        self.transform = transform
        self.n_logits = sum(len(g.classes) for g in groups)
        # Pre-compute the class -> global index map for speed.
        self._class_idx: Dict[str, Dict[str, int]] = {
            g.name: {c: g.start + i for i, c in enumerate(g.classes)} for g in groups
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = IMG_BASE / row["relative_path"]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        y = torch.zeros(self.n_logits, dtype=torch.float32)
        for g in self.groups:
            v = row[g.name]
            if isinstance(v, str) and v in self._class_idx[g.name]:
                y[self._class_idx[g.name][v]] = 1.0
        return x, y


def _build_transforms(train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultilabelClassifier(nn.Module):
    """ResNet18 backbone with a multi-label linear head of `n_logits` outputs.

    By default we start from ImageNet-pretrained weights (one-time download to
    ~/.cache/torch on first run). The transferred features make the small
    fashion dataset much more tractable than training from scratch — expect a
    sizeable lift in val F1 and subset accuracy. Pass `pretrained=False` to
    train from scratch instead.

    The head produces raw logits; apply `torch.sigmoid` at inference time and
    threshold at 0.5 (or argmax per attribute group) to recover predictions.
    """

    def __init__(self, n_logits: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    groups: List[LabelGroup],
) -> Dict[str, object]:
    model.eval()
    losses: List[float] = []
    all_logits: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.int8)
        all_logits.append(logits.cpu().numpy())
        all_pred.append(pred)
        all_true.append(y.cpu().numpy().astype(np.int8))
    y_logits = np.concatenate(all_logits, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    # Per-group top-1 accuracy: for each row, take argmax of the logits inside
    # the group and compare to argmax of the true one-hot block. Rows whose
    # group is fully zero (rare-class filter NaN'd them) are skipped.
    top1_per_group: Dict[str, float] = {}
    for g in groups:
        true_block = y_true[:, g.start:g.end]
        logit_block = y_logits[:, g.start:g.end]
        has_label = true_block.sum(axis=1) > 0
        if has_label.sum() == 0:
            top1_per_group[g.name] = float("nan")
            continue
        true_idx = true_block[has_label].argmax(axis=1)
        pred_idx = logit_block[has_label].argmax(axis=1)
        top1_per_group[g.name] = float((true_idx == pred_idx).mean())
    valid_top1 = [v for v in top1_per_group.values() if not np.isnan(v)]
    top1_avg = float(np.mean(valid_top1)) if valid_top1 else float("nan")

    return {
        "val_loss": float(np.mean(losses)),
        "val_f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "val_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_subset_acc": float((y_pred == y_true).all(axis=1).mean()),
        "val_top1_per_group": top1_per_group,
        "val_top1_avg": top1_avg,
    }


def train_one(
    product_type: str,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_samples: int,
    num_workers: int,
    seed: int,
    pretrained: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    df, groups = load_and_filter(product_type, max_samples, seed)
    n_logits = sum(len(g.classes) for g in groups)

    # 90 / 10 split (the df is already shuffled with a fixed seed).
    n_val = max(1, int(0.1 * len(df)))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    total = len(train_df) + len(val_df)
    print(
        f"[split] train={len(train_df)}  val={len(val_df)}  "
        f"total={total}  n_logits={n_logits}"
    )

    train_ds = MultilabelDataset(train_df, groups, _build_transforms(train=True))
    val_ds = MultilabelDataset(val_df, groups, _build_transforms(train=False))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    model = MultilabelClassifier(n_logits, pretrained=pretrained).to(device)
    print(f"[model] resnet18  pretrained={pretrained}  n_logits={n_logits}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"multilabel_classifier_{product_type}_v1.pth"
    labels_path = OUT_DIR / f"multilabel_classifier_{product_type}_v1.labels.json"

    best_f1 = -1.0
    best_epoch = -1
    best_metrics: Dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}")
        scheduler.step()
        train_loss = running / max(seen, 1)

        metrics = evaluate(model, val_loader, criterion, device, groups)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"f1_micro={metrics['val_f1_micro']:.4f}  "
            f"f1_macro={metrics['val_f1_macro']:.4f}  "
            f"subset_acc={metrics['val_subset_acc']:.4f}  "
            f"top1_avg={metrics['val_top1_avg']:.4f}"
        )

        if metrics["val_f1_macro"] > best_f1:
            best_f1 = float(metrics["val_f1_macro"])
            best_epoch = epoch
            best_metrics = metrics
            torch.save(model.state_dict(), ckpt_path)
            with labels_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "product_type": product_type,
                        "num_logits": n_logits,
                        "image_size": 224,
                        "normalize_mean": [0.485, 0.456, 0.406],
                        "normalize_std": [0.229, 0.224, 0.225],
                        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
                        "groups": [
                            {"name": g.name, "start": g.start, "end": g.end, "classes": g.classes}
                            for g in groups
                        ],
                        "best_epoch": best_epoch,
                        "best_val": metrics,
                    },
                    f,
                    indent=2,
                )
            print(
                f"[ckpt] saved {ckpt_path.name} "
                f"(epoch={best_epoch}, val_f1_macro={best_f1:.4f})"
            )

    print(f"[done] {product_type}: best val_f1_macro={best_f1:.4f} at epoch {best_epoch}/{epochs}")
    print(f"[done] best epoch metrics:")
    print(f"[done]   val_loss      = {best_metrics['val_loss']:.4f}")
    print(f"[done]   val_f1_micro  = {best_metrics['val_f1_micro']:.4f}")
    print(f"[done]   val_f1_macro  = {best_metrics['val_f1_macro']:.4f}")
    print(f"[done]   val_subset_acc= {best_metrics['val_subset_acc']:.4f}")
    print(f"[done]   val_top1_avg  = {best_metrics['val_top1_avg']:.4f}")
    print(f"[done] best epoch top-1 per attribute:")
    for attr, acc in best_metrics["val_top1_per_group"].items():
        print(f"[done]   {attr:<20} = {acc:.4f}")
    print(f"[done] checkpoint: {ckpt_path}")
    print(f"[done] labels:     {labels_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--product-type", required=True, choices=list(SCHEMA.keys()))
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--max-samples", type=int, default=10_000,
                   help="Cap on rows after filtering. 0 disables the cap.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate. Default 3e-4 is tuned for the pretrained "
                        "backbone; if you pass --no-pretrained, 1e-3 works better.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Default 0 for Windows safety.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                   help="Use ImageNet-pretrained ResNet18 weights (default).")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="Train ResNet18 from scratch instead.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"[env] device={device}  torch={torch.__version__}")
    train_one(
        args.product_type,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
