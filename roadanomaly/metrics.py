"""Losses and evaluation metrics for binary anomaly segmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


class BCEDiceLoss(nn.Module):
    """BCE (pixel-wise calibration) + soft Dice (robust to the ~10% foreground imbalance)."""

    def __init__(self, pos_weight=2.0, dice_w=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_w = dice_w

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(self.pos_weight, device=logits.device))
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return bce + self.dice_w * dice.mean()


def confusion(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt); tn = np.sum(~pred & ~gt)
    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)  # identical to Dice for binary masks
    iou = tp / (tp + fp + fn + eps)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, dice=f1)


def evaluate(scores: list[np.ndarray], gts: list[np.ndarray], threshold: float, n_auc_pixels=2_000_000, seed=0):
    """scores: per-image anomaly maps (same size as gt). Returns dataset-level metrics.

    * precision / recall / F1 / IoU / Dice are computed on the pixels of the whole subset
      (global confusion matrix), the convention used in segmentation benchmarks;
    * mIoU_img / mDice_img average the per-image values (every image weighs the same,
      so small anomalies count as much as large ones);
    * AUROC, AP and FPR@95%TPR are threshold-free metrics used in the anomaly-segmentation
      literature (Lis et al. 2019; Chan et al. 2021, SegmentMeIfYouCan).
    """
    TP = FP = FN = TN = 0
    per_img_iou, per_img_dice = [], []
    for s, g in zip(scores, gts):
        tp, fp, fn, tn = confusion(s >= threshold, g)
        TP += tp; FP += fp; FN += fn; TN += tn
        m = scores_from_counts(tp, fp, fn, tn)
        per_img_iou.append(m["iou"]); per_img_dice.append(m["dice"])
    res = scores_from_counts(TP, FP, FN, TN)
    res["mIoU_img"] = float(np.mean(per_img_iou))
    res["mDice_img"] = float(np.mean(per_img_dice))
    # threshold-free metrics on a random pixel subsample (memory)
    s_all = np.concatenate([s.ravel() for s in scores]); g_all = np.concatenate([g.ravel() for g in gts])
    if len(s_all) > n_auc_pixels:
        idx = np.random.default_rng(seed).choice(len(s_all), n_auc_pixels, replace=False)
        s_all, g_all = s_all[idx], g_all[idx]
    res["auroc"] = roc_auc_score(g_all, s_all)
    res["ap"] = average_precision_score(g_all, s_all)
    fpr, tpr, _ = roc_curve(g_all, s_all)
    res["fpr95"] = float(fpr[np.searchsorted(tpr, 0.95)])
    res["threshold"] = threshold
    res["per_image_iou"] = per_img_iou
    return {k: (float(v) if np.isscalar(v) else v) for k, v in res.items()}


def best_f1_threshold(scores, gts, n_steps=99):
    """Choose the decision threshold maximising pixel F1 (used on the VALIDATION set only)."""
    lo = min(float(s.min()) for s in scores); hi = max(float(s.max()) for s in scores)
    best = (-1, 0.5)
    for t in np.linspace(lo, hi, n_steps + 2)[1:-1]:
        TP = FP = FN = 0
        for s, g in zip(scores, gts):
            tp, fp, fn, _ = confusion(s >= t, g); TP += tp; FP += fp; FN += fn
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]
