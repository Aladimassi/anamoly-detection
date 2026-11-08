"""Data loading, preprocessing and splitting for the RoadAnomaly dataset.

Dataset layout (Lis et al., ICCV 2019):
    RoadAnomaly/frame_list.json
    RoadAnomaly/frames/<name>.webp                         RGB image, 1280x720
    RoadAnomaly/frames/<name>.labels/labels_semantic.png   0 = background, 2 = anomaly

IMPORTANT: the anomaly class is encoded with value **2**, not 1. Masks are
remapped to {0, 1} here so that every downstream component works with a
binary target.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Working resolution: keeps the 16:9 aspect ratio (approximately) and is divisible by 32,
# which encoder-decoder networks with 5 down-sampling stages require.
IMG_H, IMG_W = 256, 448
# Images are cached at a slightly larger size so random crops/zoom have room to work.
CACHE_H, CACHE_W = 360, 640


def category_of(name: str) -> str:
    """'animals17_E6_near...' -> 'animals', 'lost_and_found03' -> 'lost'"""
    return re.match(r"[a-zA-Z]+", name).group(0)


@dataclass
class Sample:
    name: str
    category: str
    image: np.ndarray  # HxWx3 uint8 (cached size)
    mask: np.ndarray  # HxW uint8 in {0,1}
    image_full: np.ndarray | None = None  # original resolution (for display)
    mask_full: np.ndarray | None = None


def load_dataset(root: str, keep_full: bool = False) -> list[Sample]:
    """Load every frame + binary mask, checking that image/mask correspond."""
    frames = json.load(open(os.path.join(root, "frame_list.json")))
    out = []
    for f in frames:
        stem = f.rsplit(".", 1)[0]
        img = Image.open(os.path.join(root, "frames", f)).convert("RGB")
        lab = Image.open(os.path.join(root, "frames", stem + ".labels", "labels_semantic.png"))
        lab_np = np.array(lab)
        assert img.size == lab.size, f"size mismatch for {f}: {img.size} vs {lab.size}"
        assert set(np.unique(lab_np)) <= {0, 2}, f"unexpected label values in {f}"
        mask_full = (lab_np == 2).astype(np.uint8)
        img_c = np.array(img.resize((CACHE_W, CACHE_H), Image.BILINEAR))
        mask_c = np.array(Image.fromarray(mask_full * 255).resize((CACHE_W, CACHE_H), Image.NEAREST)) // 255
        out.append(Sample(stem, category_of(stem), img_c, mask_c.astype(np.uint8),
                          np.array(img) if keep_full else None, mask_full if keep_full else None))
    return out


def stratified_split(samples, val_frac=0.17, test_frac=0.17, seed=42):
    """Split per category so every subset contains animals, cones, obstacles, ..."""
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_cat.setdefault(s.category, []).append(i)
    train, val, test = [], [], []
    for cat, idx in sorted(by_cat.items()):
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        test += idx[:n_test]
        val += idx[n_test:n_test + n_val]
        train += idx[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


class RoadAnomalyDataset(Dataset):
    """Returns (image tensor normalised with ImageNet stats, mask tensor 1xHxW float)."""

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        H, W = img.shape[:2]
        # random zoom / crop (scale 0.6-1.0 of the cached frame, aspect kept)
        s = random.uniform(0.6, 1.0)
        ch, cw = int(H * s), int(W * s)
        y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
        img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
        if random.random() < 0.5:  # horizontal flip (road scenes are left/right symmetric)
            img, mask = img[:, ::-1], mask[:, ::-1]
        img = img.astype(np.float32)
        # photometric jitter: brightness, contrast, saturation
        img = img * random.uniform(0.75, 1.25)
        mean = img.mean()
        img = (img - mean) * random.uniform(0.75, 1.25) + mean
        gray = img.mean(axis=2, keepdims=True)
        img = (img - gray) * random.uniform(0.7, 1.3) + gray
        return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        s = self.samples[i]
        img, mask = s.image, s.mask
        if self.augment:
            img, mask = self._augment(img, mask)
        img = np.array(Image.fromarray(np.ascontiguousarray(img)).resize((IMG_W, IMG_H), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask * 255).resize((IMG_W, IMG_H), Image.NEAREST)) // 255
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(mask[None].astype(np.float32))


def denormalize(x: torch.Tensor) -> np.ndarray:
    """CHW normalised tensor -> HxWx3 float image in [0,1] for plotting."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)
