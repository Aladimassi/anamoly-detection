"""Training / inference loops (device-agnostic: runs on CUDA if available, else CPU)."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import IMAGENET_MEAN, IMAGENET_STD, RoadAnomalyDataset
from .metrics import BCEDiceLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _cosine_with_warmup(opt, total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def val_iou(model, loader):
    model.eval(); tp = fp = fn = 0; loss = 0.0; crit = BCEDiceLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        lo = model(x); loss += crit(lo, y).item() * len(x)
        p = lo > 0
        tp += (p & (y > 0.5)).sum().item(); fp += (p & (y < 0.5)).sum().item(); fn += (~p & (y > 0.5)).sum().item()
    return tp / (tp + fp + fn + 1e-9), loss / len(loader.dataset)


def train_segmenter(model, train_samples, val_samples, ckpt_path, epochs=60, lr=1e-3, encoder_lr=None,
                    batch_size=4, weight_decay=1e-4, log_every=5):
    """Train a binary segmentation network with BCE+Dice; keep the checkpoint with the best val IoU.

    encoder_lr: if given, parameters under `model.encoder` use this (smaller) learning rate —
    standard practice when fine-tuning a pretrained backbone (discriminative learning rates).
    """
    model.to(DEVICE)
    tl = DataLoader(RoadAnomalyDataset(train_samples, augment=True), batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(RoadAnomalyDataset(val_samples), batch_size=batch_size)
    if encoder_lr is not None and hasattr(model, "encoder"):
        enc = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc}
        groups = [{"params": enc, "lr": encoder_lr}, {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": lr}]
    else:
        groups = [{"params": model.parameters(), "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = _cosine_with_warmup(opt, epochs * len(tl), warmup=2 * len(tl))
    crit = BCEDiceLoss()
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": []}
    best, t0 = -1.0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(x)
        iou, vloss = val_iou(model, vl)
        hist["epoch"].append(ep); hist["train_loss"].append(run / (len(tl) * batch_size))
        hist["val_loss"].append(vloss); hist["val_iou"].append(iou)
        if iou > best:
            best = iou; torch.save(model.state_dict(), ckpt_path)
        if ep % log_every == 0 or ep == 1:
            print(f"ep {ep:3d} | train {hist['train_loss'][-1]:.4f} | val {vloss:.4f} | val IoU {iou:.4f} "
                  f"| best {best:.4f} | {time.time() - t0:.0f}s", flush=True)
    json.dump(hist, open(ckpt_path.replace(".pt", "_history.json"), "w"))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return hist


@torch.no_grad()
def predict_probs(model, samples, out_hw=(720, 1280), tta=True):
    """Anomaly probability maps, upsampled to the original resolution for evaluation."""
    model.to(DEVICE).eval(); out = []
    ds = RoadAnomalyDataset(samples)
    for i in range(len(ds)):
        x = ds[i][0][None].to(DEVICE)
        p = torch.sigmoid(model(x))
        if tta:  # horizontal-flip test-time augmentation
            p = 0.5 * (p + torch.sigmoid(model(x.flip(-1))).flip(-1))
        p = F.interpolate(p, size=out_hw, mode="bilinear", align_corners=False)
        out.append(p[0, 0].cpu().numpy())
    return out


# ----------------------------------------------------------------------------- patch classifiers (ANN / CNN)
# The image (448x256) is cut into a grid of 8x8 "cells" (56x32 cells). For each cell we feed the classifier the
# 32x32 patch centred on it (context around the cell) and the network predicts the fraction of anomalous pixels in
# that cell. Sliding the window with stride 8 gives a coarse anomaly map. Same input for ANN and CNN => fair comparison.
CELL, PAD = 8, 12  # (32 - 8) / 2 = 12 pixels of context on each side


def extract_patches(x, y=None):
    """x: Bx3xHxW -> (B*nh*nw)x3x32x32 patches; y (optional): matching cell-level anomaly fractions."""
    B, C, H, W = x.shape
    xp = F.pad(x, (PAD, PAD, PAD, PAD), mode="reflect")
    p = F.unfold(xp, kernel_size=32, stride=CELL)  # B x (3*32*32) x L
    p = p.transpose(1, 2).reshape(-1, C, 32, 32)
    if y is None:
        return p
    t = F.avg_pool2d(y, CELL).reshape(-1)  # fraction of anomaly pixels in every 8x8 cell
    return p, t


def train_patch_classifier(model, train_samples, val_samples, ckpt_path, epochs=40, lr=1e-3, batch_images=4,
                           n_pos=192, n_neg=192, log_every=5):
    """Trains on balanced mini-batches of patches (anomalous cells are rare, ~10%)."""
    model.to(DEVICE)
    tl = DataLoader(RoadAnomalyDataset(train_samples, augment=True), batch_size=batch_images, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = _cosine_with_warmup(opt, epochs * len(tl), warmup=len(tl))
    hist = {"epoch": [], "train_loss": [], "val_iou": []}; best, t0 = -1.0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for x, y in tl:
            p, t = extract_patches(x, y)
            pos = torch.nonzero(t > 0).flatten(); neg = torch.nonzero(t == 0).flatten()
            sel = torch.cat([pos[torch.randperm(len(pos))[:n_pos]], neg[torch.randperm(len(neg))[:n_neg]]])
            p, t = p[sel].to(DEVICE), t[sel].to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(p).squeeze(1), t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run += loss.item()
        maps = predict_patch_maps(model, val_samples, out_hw=None)
        tp = fp = fn = 0
        for m, s in zip(maps, RoadAnomalyDataset(val_samples)):
            g = s[1][0].numpy() > 0.5; pr = m >= 0.5
            tp += (pr & g).sum(); fp += (pr & ~g).sum(); fn += (~pr & g).sum()
        iou = tp / (tp + fp + fn + 1e-9)
        hist["epoch"].append(ep); hist["train_loss"].append(run / len(tl)); hist["val_iou"].append(float(iou))
        if iou > best:
            best = iou; torch.save(model.state_dict(), ckpt_path)
        if ep % log_every == 0 or ep == 1:
            print(f"ep {ep:3d} | train BCE {hist['train_loss'][-1]:.4f} | val IoU {iou:.4f} | best {best:.4f} | {time.time() - t0:.0f}s", flush=True)
    json.dump(hist, open(ckpt_path.replace(".pt", "_history.json"), "w"))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return hist


@torch.no_grad()
def predict_patch_maps(model, samples, out_hw=(720, 1280)):
    """Sliding-window inference: one score per 8x8 cell, upsampled (bilinear) to out_hw (or 256x448 if None)."""
    model.to(DEVICE).eval(); ds = RoadAnomalyDataset(samples); out = []
    for i in range(len(ds)):
        x = ds[i][0][None]
        s = torch.sigmoid(model(extract_patches(x).to(DEVICE))).reshape(1, 1, x.shape[2] // CELL, x.shape[3] // CELL)
        s = F.interpolate(s, size=out_hw or x.shape[-2:], mode="bilinear", align_corners=False)
        out.append(s[0, 0].cpu().numpy())
    return out


# ----------------------------------------------------------------------------- autoencoder
def train_autoencoder(ae, train_samples, ckpt_path, epochs=150, lr=2e-3, batch_size=4, mask_anomalies=True,
                      log_every=10):
    """Reconstruction training (L1). With mask_anomalies=True the loss ignores annotated anomaly
    pixels, so the AE only ever learns to reproduce *normal* content (road, sky, vegetation...).
    Note: the AE never sees the masks as a target - it is not a segmentation model."""
    ae.to(DEVICE)
    tl = DataLoader(RoadAnomalyDataset(train_samples, augment=True), batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-5)
    sched = _cosine_with_warmup(opt, epochs * len(tl), warmup=2 * len(tl))
    hist = {"epoch": [], "train_loss": []}; t0 = time.time()
    for ep in range(1, epochs + 1):
        ae.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            err = (ae(x) - x).abs()
            w = (1 - y) if mask_anomalies else torch.ones_like(y)
            loss = (err * w).sum() / (w.sum() * 3 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        hist["epoch"].append(ep); hist["train_loss"].append(run / len(tl))
        if ep % log_every == 0 or ep == 1:
            print(f"ep {ep:3d} | recon L1 {hist['train_loss'][-1]:.4f} | {time.time() - t0:.0f}s", flush=True)
    torch.save(ae.state_dict(), ckpt_path)
    json.dump(hist, open(ckpt_path.replace(".pt", "_history.json"), "w"))
    return hist


def _gauss_blur(t, sigma=4.0):
    k = int(2 * round(3 * sigma) + 1); ax = torch.arange(k, device=t.device) - k // 2
    g = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


@torch.no_grad()
def autoencoder_scores(ae, extractor, samples, out_hw=(720, 1280)):
    """Two anomaly maps per image:
       * 'pixel'      : blurred mean absolute RGB reconstruction error
       * 'perceptual' : 1 - cosine similarity between ResNet features of input and reconstruction
    Also returns the reconstructions for visualisation."""
    ae.to(DEVICE).eval(); extractor.to(DEVICE)
    ds = RoadAnomalyDataset(samples); pix, perc, recs = [], [], []
    for i in range(len(ds)):
        x = ds[i][0][None].to(DEVICE); r = ae(x)
        e = _gauss_blur((r - x).abs().mean(1, keepdim=True))
        fx, fr = extractor(x), extractor(r)
        d = sum(F.interpolate(1 - F.cosine_similarity(a, b, dim=1)[:, None], size=x.shape[-2:], mode="bilinear",
                              align_corners=False) for a, b in zip(fx, fr)) / len(fx)
        d = _gauss_blur(d)
        pix.append(F.interpolate(e, size=out_hw, mode="bilinear")[0, 0].cpu().numpy())
        perc.append(F.interpolate(d, size=out_hw, mode="bilinear")[0, 0].cpu().numpy())
        rec = r[0].cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        recs.append(np.clip(rec, 0, 1))
    return pix, perc, recs
