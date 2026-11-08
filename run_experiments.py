"""Train every model of the study and save checkpoints + training histories to ./checkpoints.

Usage:  python run_experiments.py --data /path/to/RoadAnomaly [--only unet_r34_imagenet ...]
The notebook can either call this or simply load the checkpoints it produces.
"""
import argparse
import os

import torch

from roadanomaly.data import load_dataset, stratified_split
from roadanomaly.models import ConvAutoencoder, PatchANN, PatchCNN, UNetScratch, deeplab_resnet34, unet_resnet34
from roadanomaly.train import seed_everything, train_autoencoder, train_patch_classifier, train_segmenter

EXPERIMENTS = {
    # name: (constructor, kwargs for train_segmenter)
    "unet_scratch": (lambda: UNetScratch(), dict(epochs=80, lr=1e-3)),
    "unet_r34_random": (lambda: unet_resnet34(pretrained=False), dict(epochs=60, lr=1e-3)),
    "unet_r34_imagenet": (lambda: unet_resnet34(pretrained=True), dict(epochs=60, lr=1e-3, encoder_lr=1e-4)),
    "deeplabv3p_r34_imagenet": (lambda: deeplab_resnet34(pretrained=True), dict(epochs=60, lr=1e-3, encoder_lr=1e-4)),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/RoadAnomaly")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count())
    os.makedirs("checkpoints", exist_ok=True)
    S = load_dataset(a.data)
    tr, va, te = stratified_split(S)
    train_s, val_s = [S[i] for i in tr], [S[i] for i in va]
    for name, (ctor, kw) in EXPERIMENTS.items():
        if a.only and name not in a.only:
            continue
        print(f"\n===== {name} =====", flush=True)
        seed_everything(42)
        train_segmenter(ctor(), train_s, val_s, f"checkpoints/{name}.pt", **kw)
    for name, ctor in {"patch_ann": PatchANN, "patch_cnn": PatchCNN}.items():
        if a.only and name not in a.only:
            continue
        print(f"\n===== {name} =====", flush=True)
        seed_everything(42)
        train_patch_classifier(ctor(), train_s, val_s, f"checkpoints/{name}.pt", epochs=40)
    if not a.only or "autoencoder" in a.only:
        print("\n===== autoencoder =====", flush=True)
        seed_everything(42)
        train_autoencoder(ConvAutoencoder(), train_s, "checkpoints/autoencoder.pt", epochs=150)
