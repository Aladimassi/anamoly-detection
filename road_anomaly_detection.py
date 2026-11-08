# %% [markdown]
# # Road Anomaly Detection on the RoadAnomaly dataset
# **Deep Learning module project**
#
# **Goal.** Given a road-scene image, produce a pixel-level map of *unexpected* objects on or near the road
# (animals, rocks, traffic cones, lost cargo, unusual vehicles…). This is the problem studied by
# Lis, Nakka, Fua & Salzmann, *Detecting the Unexpected via Image Resynthesis*, ICCV 2019, for which the
# RoadAnomaly dataset was released.
#
# **Plan of the notebook**
# 1. Dataset exploration — structure, annotations, image/mask correspondence, statistics
# 2. Preprocessing — label remapping, resizing, stratified train/val/test split, augmentation
# 3. Models, from the simplest to the most advanced — all built from the ANN/CNN building blocks seen in class:
#    (1) ANN (MLP) on image patches, (2) classic CNN classifier on the same patches, (3) U-Net CNN from scratch,
#    (4) U-Net with ResNet-34 encoder, random init, (5) same with ImageNet weights (transfer learning),
#    (6) DeepLabV3+ (transfer learning), (7) convolutional autoencoder (unsupervised, reconstruction error)
# 4. Evaluation — precision, recall, F1, IoU, Dice (+ AUROC, AP, FPR@95%TPR), bootstrap confidence intervals
# 5. Results analysis — qualitative predictions, error maps, per-category and per-size analysis, failure cases
#
# Code is split between this notebook (experiments, figures, discussion) and the documented package
# `roadanomaly/` (`data.py`, `models.py`, `metrics.py`, `train.py`). Everything runs on CPU or GPU.

# %%
import json, os, sys, time, warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from roadanomaly.data import (CACHE_H, CACHE_W, IMG_H, IMG_W, RoadAnomalyDataset, denormalize,
                              load_dataset, stratified_split)
from roadanomaly.metrics import best_f1_threshold, evaluate
from roadanomaly.models import (ConvAutoencoder, PatchANN, PatchCNN, PerceptualExtractor, UNetScratch,
                                count_params, deeplab_resnet34, unet_resnet34)
from roadanomaly.train import (DEVICE, autoencoder_scores, predict_patch_maps, predict_probs, seed_everything,
                               train_autoencoder, train_patch_classifier, train_segmenter)

DATA_ROOT = os.environ.get("ROADANOMALY_ROOT", "data/RoadAnomaly")
TRAIN = False          # True: retrain every model (≈2.5 h on CPU, ≈10 min on a GPU); False: load ./checkpoints
FIG = "figures"; os.makedirs(FIG, exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count())
seed_everything(42)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("device:", DEVICE, "| torch", torch.__version__)

# %% [markdown]
# ## 1. Dataset exploration
# ### 1.1 Structure
# ```
# RoadAnomaly/
#   frame_list.json                      list of the 60 frame names
#   credits.txt                          image sources (web images, various licences)
#   frames/<name>.webp                   RGB image
#   frames/<name>.labels/labels_semantic.png        per-pixel label: 0 = background, 2 = anomaly
#   frames/<name>.labels/labels_semantic_color.png  colour visualisation of the same label
#   frames/<name>.labels/labels_instance.png        instance ids (only 46/60 frames)
#   frames/<name>.labels/instance_XXX_gc_*          LabelGrab (GrabCut) editor state
# ```

# %%
frames = json.load(open(os.path.join(DATA_ROOT, "frame_list.json")))
print(len(frames), "frames, e.g.", frames[:3])
lab_dirs = [os.path.join(DATA_ROOT, "frames", f.rsplit(".", 1)[0] + ".labels") for f in frames]
print("frames with labels_semantic.png :", sum(os.path.exists(os.path.join(d, "labels_semantic.png")) for d in lab_dirs))
print("frames with labels_instance.png :", sum(os.path.exists(os.path.join(d, "labels_instance.png")) for d in lab_dirs))

# raw inspection of image sizes and label values (before any preprocessing)
sizes, values = Counter(), Counter()
for f, d in zip(frames, lab_dirs):
    im = Image.open(os.path.join(DATA_ROOT, "frames", f)); lab = np.array(Image.open(os.path.join(d, "labels_semantic.png")))
    sizes[(im.size, lab.shape[::-1])] += 1; values.update(np.unique(lab).tolist())
print("(image size, mask size) ->", dict(sizes))
print("label values present     ->", dict(values), " => anomaly is encoded as 2, NOT 1")

# %% [markdown]
# **Image / mask correspondence.** All 60 images and masks are 1280×720 and share the same file stem, so the
# pairing is unambiguous. Two pitfalls are worth stating explicitly:
# * the anomaly class is stored with value **2** (the value 1 is reserved for "road" in the authors' tooling and
#   never appears here). Treating the mask as `mask > 0` or `mask == 2` is correct; `mask == 1` would give an
#   empty target and a model that silently learns nothing;
# * masks must be resized with **nearest-neighbour** interpolation, otherwise interpolated values appear at the
#   borders.
#
# `load_dataset` asserts both points for every frame.

# %%
samples = load_dataset(DATA_ROOT, keep_full=True)
cats = Counter(s.category for s in samples)
frac = np.array([s.mask_full.mean() for s in samples])
inst = []
for s in samples:
    p = os.path.join(DATA_ROOT, "frames", s.name + ".labels", "labels_instance.png")
    inst.append(len(np.unique(np.array(Image.open(p)))) - 1 if os.path.exists(p) else np.nan)
eda = pd.DataFrame({"name": [s.name for s in samples], "category": [s.category for s in samples],
                    "anomaly_frac": frac, "n_instances": inst})
display(eda.groupby("category").agg(n=("name", "size"), mean_anomaly_pct=("anomaly_frac", lambda x: 100 * x.mean()),
                                    min_pct=("anomaly_frac", lambda x: 100 * x.min()),
                                    max_pct=("anomaly_frac", lambda x: 100 * x.max()),
                                    mean_instances=("n_instances", "mean")).round(2))
print(f"anomaly pixels overall: {100 * frac.mean():.2f}%  (median per image {100 * np.median(frac):.2f}%)")

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
cc = eda.category.value_counts(); ax[0].bar(cc.index, cc.values, color="#4C72B0"); ax[0].set_title("Frames per category")
ax[1].hist(100 * frac, bins=20, color="#DD8452"); ax[1].set_xlabel("% of image covered by anomaly"); ax[1].set_title("Anomaly size distribution")
prior = np.mean([s.mask for s in samples], axis=0)
im = ax[2].imshow(prior, cmap="magma"); ax[2].set_title("Spatial prior: P(anomaly) per pixel"); ax[2].axis("off")
plt.colorbar(im, ax=ax[2], fraction=0.03); plt.tight_layout(); plt.savefig(f"{FIG}/eda_stats.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# Observations:
# * **Very small dataset** (60 images) dominated by animals (31). Cones, lost cargo and unusual vehicles have
#   3–8 examples each — any learned model will see very few examples of these.
# * **Strong class imbalance**: anomalies cover ≈10% of pixels on average, but the median is ≈6.6% and some
#   objects cover <0.5% of the frame (distant cones, small animals). Pixel accuracy would be meaningless
#   (an all-background prediction scores ≈90%), hence the use of IoU/F1/Dice and threshold-free metrics.
# * **Spatial prior**: anomalies concentrate in a central horizontal band (on the road, around and below the
#   horizon) and almost never touch the top of the frame (sky) or its lateral edges.
#   A CNN can exploit this context; it is legitimate for the task but also a possible shortcut — we therefore
#   include a "spatial prior only" reference that predicts this average map for every image.
# * Annotation semantics: people, cyclists and ordinary cars are background; for animal-drawn carts the whole
#   cart + load + animal is labelled anomalous. The "lost" frames are dash-cam video stills with watermarks.
# * Images are web photos with heterogeneous viewpoints, weather and quality — very different from the
#   car-mounted cameras of Cityscapes.

# %%
def overlay(img, mask, color=(1, 0, 0), alpha=0.45):
    img = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.copy()
    out = img.copy(); m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out

show = [eda[eda.category == c].index[0] for c in cats] + list(eda.sort_values("anomaly_frac").index[[0, -1]])
fig, axes = plt.subplots(3, len(show), figsize=(3.2 * len(show), 6.4))
for j, i in enumerate(show):
    s = samples[i]
    color = np.array(Image.open(os.path.join(DATA_ROOT, "frames", s.name + ".labels", "labels_semantic_color.png")).convert("RGB"))
    for r, im_ in enumerate([s.image_full, color, overlay(s.image_full, s.mask_full)]):
        axes[r, j].imshow(im_); axes[r, j].axis("off")
    axes[0, j].set_title(f"{s.category}\n{100 * frac[i]:.1f}% anomaly", fontsize=9)
axes[0, 0].text(-80, 360, "image", rotation=90, va="center"); axes[1, 0].text(-80, 360, "label (colour)", rotation=90, va="center")
axes[2, 0].text(-80, 360, "binary mask overlay", rotation=90, va="center")
plt.tight_layout(); plt.savefig(f"{FIG}/eda_examples.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# The overlays confirm that the masks are aligned with the objects (one example per category, plus the
# smallest and the largest anomaly). Note that the annotation marks *only* the anomalous objects; the rest
# (road, cars, people, buildings) is all "background". People and ordinary cars are **not** anomalies here.

# %% [markdown]
# ## 2. Preprocessing
# * **Label remapping**: `mask = (labels_semantic == 2)` → binary {0,1}.
# * **Resizing**: images cached at 640×360, network input 448×256 (16:9 kept, divisible by 32 as required by
#   5-stage encoders). Bilinear for images, nearest for masks. Predictions are **upsampled back to 1280×720
#   and evaluated against the original full-resolution masks**.
# * **Normalisation**: ImageNet mean/std (needed by the pretrained encoders; harmless for the others).
# * **Split**: stratified by category (seed 42) so every subset contains animals, cones, obstacles, lost cargo
#   and vehicles: 38 train / 11 validation / 11 test. The validation set is used for checkpoint selection and
#   threshold choice; the test set is touched **once**, at the end.
# * **Augmentation** (train only — vital with 38 images): random zoom-crop (60–100%), horizontal flip,
#   brightness/contrast/saturation jitter. No vertical flips or rotations (road geometry has a fixed "up").

# %%
tr_idx, va_idx, te_idx = stratified_split(samples)
train_s, val_s, test_s = ([samples[i] for i in ix] for ix in (tr_idx, va_idx, te_idx))
split = pd.DataFrame({"split": ["train"] * len(tr_idx) + ["val"] * len(va_idx) + ["test"] * len(te_idx),
                      "category": [samples[i].category for i in tr_idx + va_idx + te_idx]})
display(pd.crosstab(split.category, split.split, margins=True)[["train", "val", "test", "All"]])
print("test frames:", [s.name for s in test_s])

ds_aug = RoadAnomalyDataset(train_s[:1], augment=True)
fig, axes = plt.subplots(1, 5, figsize=(17, 2.6))
for k, ax in enumerate(axes):
    x, y = ds_aug[0] if k else RoadAnomalyDataset(train_s[:1])[0]
    ax.imshow(overlay(denormalize(x), y[0].numpy())); ax.axis("off"); ax.set_title("original (resized)" if k == 0 else f"augmentation {k}")
plt.tight_layout(); plt.savefig(f"{FIG}/augmentations.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 3. Models
#
# ### 3.1 From ANN to CNN to segmentation network
# The models form a progression; each step adds one idea to the previous one.
#
# | # | Model | Building blocks (course vocabulary) | Why it is included |
# |---|---|---|---|
# | 1 | **ANN on patches** (1.6 M params) | Flatten → Dense(512) → Dense(128) → Dense(1, sigmoid) | Simplest neural baseline. The image is cut into 8×8-pixel cells; for each cell the ANN sees the surrounding 32×32 patch (flattened to 3072 numbers) and predicts how much of the cell is anomalous. Sliding it over the image gives a coarse anomaly map. |
# | 2 | **CNN classifier on patches** (0.36 M params) | 3× [Conv 3×3 → ReLU → MaxPool] → Flatten → Dense(128) → Dense(1) | *Exactly the same input and target as the ANN*: isolates the benefit of convolutions (local receptive fields, weight sharing, translation invariance) with 4.6× fewer parameters. |
# | 3 | **U-Net from scratch** (7.8 M) | Conv/ReLU/MaxPool **encoder** (like the CNN above, deeper) + **decoder** of transposed convolutions (learned up-sampling) + **skip connections** | Instead of classifying patches one by one, a *fully convolutional* network (no Dense layer) outputs a full-resolution mask in one pass. Skip connections give back the fine details lost by pooling. |
# | 4 | **U-Net, ResNet-34 encoder, random init** | Encoder = ResNet-34, a 34-layer CNN with *residual* (shortcut) connections, which make deep CNNs trainable | Same architecture as 5 but no pretraining → isolates the effect of transfer learning. |
# | 5 | **U-Net, ResNet-34, ImageNet weights** (24.6 M) | Same as 4, encoder initialised with weights learnt on 1.2 M ImageNet photos | **Transfer learning**: ImageNet features already encode "animal", "rock", "cone"… — what we need with only 38 training images. Encoder fine-tuned with a 10× smaller learning rate so the pretrained filters are not destroyed. |
# | 6 | **DeepLabV3+, ResNet-34, ImageNet** (22.4 M) | Same pretrained CNN encoder; decoder uses *dilated* ("atrous") convolutions = convolutions with holes, which see a wider area without pooling | Alternative segmentation decoder: looks at several scales at once (anomalies range from 0.4% to 42% of the frame). |
# | 7 | **Convolutional autoencoder** (1.3 M) | CNN encoder (strided Conv) → small bottleneck → CNN decoder (transposed Conv) | Unsupervised family: trained only to *reconstruct* normal pixels (anomaly pixels are masked out of the loss). At test time, what it fails to reconstruct is flagged as anomalous — same idea as the "resynthesis" of Lis et al. |
#
# **Why no RNN?** RNNs (and LSTM/GRU) process *sequences* — time series, text, video frames over time — by
# carrying a hidden state from one step to the next. RoadAnomaly contains 60 independent still photos: there is
# no temporal order to exploit, and the useful structure is *spatial* (neighbouring pixels), which is exactly what
# convolutions model. An RNN would only become relevant with video (e.g. a CNN per frame + an LSTM over time to
# make detections temporally consistent) — mentioned as future work.
#
# ### 3.2 Training set-up
# **Loss** (models 3–6): BCE (pos_weight 2) + soft Dice — BCE gives well-calibrated per-pixel gradients, Dice counters
# the 1:9 class imbalance and directly optimises the overlap metric.
# **Optimiser**: AdamW, weight decay 1e-4, 2-epoch warm-up + cosine decay, batch 4. 60 epochs (80 for the
# scratch U-Net, which converges more slowly). The checkpoint with the best validation IoU is kept.
#
# **Important framing remark.** Models A–D are *supervised*: they learn what the annotated anomaly classes look
# like (the same is true for models 1–2). This is what the assignment asks for and what the 60 pixel-level labels allow, but it differs from the
# paper's setting, where the detector is trained on Cityscapes (no anomalies) and RoadAnomaly is used only as a
# test set: true anomalies are by definition *unknown* at training time. Model E is closer to that philosophy.
# We come back to this in the discussion.

# %%
MODELS = {
    "U-Net (scratch)": ("unet_scratch", lambda: UNetScratch(), dict(epochs=80, lr=1e-3)),
    "U-Net R34 (random init)": ("unet_r34_random", lambda: unet_resnet34(pretrained=False), dict(epochs=60, lr=1e-3)),
    "U-Net R34 (ImageNet)": ("unet_r34_imagenet", lambda: unet_resnet34(pretrained=True), dict(epochs=60, lr=1e-3, encoder_lr=1e-4)),
    "DeepLabV3+ R34 (ImageNet)": ("deeplabv3p_r34_imagenet", lambda: deeplab_resnet34(pretrained=True), dict(epochs=60, lr=1e-3, encoder_lr=1e-4)),
}
nets, hists = {}, {}
# models 1-2: patch classifiers (ANN vs CNN on identical inputs)
PATCH_MODELS = {"ANN (patches)": ("patch_ann", PatchANN), "CNN classifier (patches)": ("patch_cnn", PatchCNN)}
patch_nets = {}
for label, (key, ctor) in PATCH_MODELS.items():
    ckpt = f"checkpoints/{key}.pt"; seed_everything(42); net = ctor()
    if TRAIN or not os.path.exists(ckpt):
        print(f"--- training {label}"); train_patch_classifier(net, train_s, val_s, ckpt, epochs=40)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)); patch_nets[label] = net.to(DEVICE).eval()
    hists[label] = json.load(open(ckpt.replace(".pt", "_history.json")))
    print(f"{label:28s} params={count_params(net) / 1e6:5.2f}M  best val IoU={max(hists[label]['val_iou']):.3f} "
          f"(epoch {int(np.argmax(hists[label]['val_iou'])) + 1})")
# models 3-6: segmentation networks
for label, (key, ctor, kw) in MODELS.items():
    ckpt = f"checkpoints/{key}.pt"
    seed_everything(42); net = ctor()
    if TRAIN or not os.path.exists(ckpt):
        print(f"--- training {label}"); train_segmenter(net, train_s, val_s, ckpt, **kw)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)); nets[label] = net.to(DEVICE).eval()
    hists[label] = json.load(open(ckpt.replace(".pt", "_history.json")))
    print(f"{label:28s} params={count_params(net) / 1e6:5.1f}M  best val IoU={max(hists[label]['val_iou']):.3f} "
          f"(epoch {int(np.argmax(hists[label]['val_iou'])) + 1})")

ae = ConvAutoencoder()
if TRAIN or not os.path.exists("checkpoints/autoencoder.pt"):
    seed_everything(42); train_autoencoder(ae, train_s, "checkpoints/autoencoder.pt", epochs=150)
ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=DEVICE)); ae.eval()
ae_hist = json.load(open("checkpoints/autoencoder_history.json"))
print(f"Autoencoder params={count_params(ae) / 1e6:.1f}M  final recon L1={ae_hist['train_loss'][-1]:.4f}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
for label, h in hists.items():
    if "val_loss" in h:
        ax[0].plot(h["epoch"], h["train_loss"], label=label); ax[1].plot(h["epoch"], h["val_loss"], label=label)
    ax[2].plot(h["epoch"], h["val_iou"], label=label, ls="--" if "patches" in label else "-")
ax[0].set_title("segmentation nets: train loss (BCE+Dice)"); ax[1].set_title("segmentation nets: validation loss")
ax[2].set_title("validation IoU (thr 0.5), all supervised models")
for a in ax: a.set_xlabel("epoch")
ax[2].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/learning_curves.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4. Evaluation
# Protocol:
# 1. compute anomaly maps at 1280×720 for validation and test images (flip test-time augmentation for A–D);
# 2. pick, **on the validation set only**, the decision threshold maximising pixel F1 (for the supervised models
#    we also report the default 0.5; the autoencoder scores are not probabilities, so a tuned threshold is
#    mandatory);
# 3. report on the test set: precision, recall, F1, IoU, Dice (from the global pixel confusion matrix),
#    per-image mean IoU/Dice, and the threshold-free AUROC, AP and FPR@95%TPR.
#
# Note that for binary masks **F1 and Dice are the same quantity** (2TP / (2TP+FP+FN)); IoU = Dice / (2 − Dice).

# %%
gt_val = [s.mask_full for s in val_s]; gt_test = [s.mask_full for s in test_s]
maps_val, maps_test = {}, {}
for label, net in patch_nets.items():
    maps_val[label] = predict_patch_maps(net, val_s); maps_test[label] = predict_patch_maps(net, test_s)
for label, net in nets.items():
    maps_val[label] = predict_probs(net, val_s); maps_test[label] = predict_probs(net, test_s)

extractor = PerceptualExtractor()
pv, qv, _ = autoencoder_scores(ae, extractor, val_s)
pt, qt, recon_test = autoencoder_scores(ae, extractor, test_s)
maps_val["Autoencoder (pixel error)"], maps_test["Autoencoder (pixel error)"] = pv, pt
maps_val["Autoencoder (perceptual error)"], maps_test["Autoencoder (perceptual error)"] = qv, qt

# trivial reference: the spatial prior learnt from training masks (no image content at all)
prior_full = np.array(Image.fromarray((np.mean([s.mask for s in train_s], 0) * 255).astype(np.uint8)).resize((1280, 720), Image.BILINEAR)) / 255.0
maps_val["Spatial prior (no image)"] = [prior_full] * len(val_s); maps_test["Spatial prior (no image)"] = [prior_full] * len(test_s)

rows, results = [], {}
for label in maps_test:
    thr, f1v = best_f1_threshold(maps_val[label], gt_val)
    r = evaluate(maps_test[label], gt_test, thr); results[label] = r
    rows.append({"model": label, "thr (val)": thr, **{k: r[k] for k in ["precision", "recall", "f1", "iou", "dice", "mIoU_img", "mDice_img", "auroc", "ap", "fpr95"]}})
    if label in nets or label in patch_nets:
        r5 = evaluate(maps_test[label], gt_test, 0.5); results[label + " @0.5"] = r5
        rows.append({"model": label + " @0.5", "thr (val)": 0.5, **{k: r5[k] for k in ["precision", "recall", "f1", "iou", "dice", "mIoU_img", "mDice_img", "auroc", "ap", "fpr95"]}})
table = pd.DataFrame(rows).set_index("model")
display(table.style.format("{:.3f}").background_gradient(subset=["f1", "iou", "ap"], cmap="Greens"))
table.to_csv("results_test.csv")

# %% [markdown]
# ### 4.1 How reliable are these numbers? Bootstrap over test images
# The test set has only 11 images, so a single number hides a lot of variance. We resample the test images
# with replacement (2000 times) and recompute the global IoU to get 95% confidence intervals.

# %%
def bootstrap_iou(maps, gts, thr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.array([[np.sum((m >= thr) & (g > 0)), np.sum((m >= thr) & (g == 0)), np.sum((m < thr) & (g > 0))] for m, g in zip(maps, gts)])
    vals = []
    for _ in range(n):
        tp, fp, fn = c[rng.integers(0, len(c), len(c))].sum(0); vals.append(tp / (tp + fp + fn + 1e-9))
    return np.percentile(vals, [2.5, 97.5])

ci = {}
for label in maps_test:
    lo, hi = bootstrap_iou(maps_test[label], gt_test, results[label]["threshold"]); ci[label] = (lo, hi)
    print(f"{label:32s} IoU {results[label]['iou']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

# %%
from sklearn.metrics import precision_recall_curve, roc_curve
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
rng = np.random.default_rng(0)
g_all = np.concatenate([g.ravel() for g in gt_test]); sub = rng.choice(len(g_all), 1_000_000, replace=False)
for label in maps_test:
    s_all = np.concatenate([m.ravel() for m in maps_test[label]])[sub]
    p, r, _ = precision_recall_curve(g_all[sub], s_all); f, t, _ = roc_curve(g_all[sub], s_all)
    ls = "--" if "Autoencoder" in label or "prior" in label else "-"
    ax[0].plot(r, p, ls, label=f"{label} (AP={results[label]['ap']:.2f})"); ax[1].plot(f, t, ls, label=f"{label} (AUROC={results[label]['auroc']:.2f})")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("Precision–recall (test pixels)")
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate"); ax[1].set_title("ROC (test pixels)")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/pr_roc.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 5. Results analysis
# ### 5.1 Qualitative predictions
# Green = true positive, red = false positive, blue = missed anomaly (false negative).

# %%
def error_map(img, pred, gt):
    out = img.astype(np.float32) / 255 * 0.6
    out[pred & gt] = [0.1, 0.85, 0.1]; out[pred & ~gt] = [0.95, 0.1, 0.1]; out[~pred & gt] = [0.1, 0.35, 1.0]
    return out

show_models = ["ANN (patches)", "CNN classifier (patches)", "U-Net (scratch)", "U-Net R34 (ImageNet)", "DeepLabV3+ R34 (ImageNet)", "Autoencoder (perceptual error)"]
fig, axes = plt.subplots(len(test_s), 2 + len(show_models), figsize=(3.0 * (2 + len(show_models)), 1.75 * len(test_s)))
for i, s in enumerate(test_s):
    axes[i, 0].imshow(s.image_full); axes[i, 1].imshow(overlay(s.image_full, s.mask_full, (0.1, 0.85, 0.1)))
    for j, label in enumerate(show_models):
        pred = maps_test[label][i] >= results[label]["threshold"]; g = s.mask_full.astype(bool)
        iou = (pred & g).sum() / ((pred | g).sum() + 1e-9)
        axes[i, 2 + j].imshow(error_map(s.image_full, pred, g)); axes[i, 2 + j].set_title(f"IoU {iou:.2f}", fontsize=8, pad=2)
    axes[i, 0].set_ylabel(s.category, fontsize=8)
for ax_ in axes.ravel(): ax_.set_xticks([]); ax_.set_yticks([])
for j, t in enumerate(["image", "ground truth"] + show_models):
    axes[0, j].set_title((t + "\n" + axes[0, j].get_title()).strip(), fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/qualitative_test.png", bbox_inches="tight", dpi=90); plt.show()

# %%
best_label = max(nets, key=lambda k: results[k]["iou"])
fig, axes = plt.subplots(4, 4, figsize=(17, 8))
for j, i in enumerate(range(0, len(test_s), max(1, len(test_s) // 4))[:4]):
    s = test_s[i]
    axes[0, j].imshow(s.image_full); axes[0, j].set_title(s.name[:30], fontsize=8)
    axes[1, j].imshow(maps_test[best_label][i], cmap="inferno", vmin=0, vmax=1); axes[1, j].set_title(f"{best_label}: probability", fontsize=8)
    axes[2, j].imshow(recon_test[i]); axes[2, j].set_title("autoencoder reconstruction", fontsize=8)
    axes[3, j].imshow(maps_test["Autoencoder (perceptual error)"][i], cmap="inferno"); axes[3, j].set_title("AE perceptual error", fontsize=8)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/heatmaps_and_reconstructions.png", bbox_inches="tight", dpi=90); plt.show()

# %% [markdown]
# ### 5.2 Where do the models fail? Per-category and per-size analysis
# Test-set per-image IoU for each test frame, against the size of the anomaly.

# %%
per = pd.DataFrame({"frame": [s.name[:28] for s in test_s], "category": [s.category for s in test_s],
                    "anomaly_%": [100 * s.mask_full.mean() for s in test_s]})
for label in list(patch_nets) + list(nets) + ["Autoencoder (perceptual error)"]:
    per[label] = results[label]["per_image_iou"]
display(per.sort_values("anomaly_%").style.format({c: "{:.2f}" for c in per.columns[2:]}).background_gradient(subset=list(per.columns[3:]), cmap="RdYlGn", vmin=0, vmax=1))

fig, ax = plt.subplots(figsize=(7, 4))
for label in ["CNN classifier (patches)"] + list(nets):
    ax.scatter(per["anomaly_%"], per[label], label=label, s=30, alpha=0.8)
ax.set_xscale("log"); ax.set_xlabel("anomaly size (% of image, log)"); ax.set_ylabel("per-image IoU"); ax.legend(fontsize=7)
ax.set_title("Per-image IoU vs. anomaly size (test)"); plt.tight_layout(); plt.savefig(f"{FIG}/iou_vs_size.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ### 5.3 Sanity check on the validation set
# The threshold and checkpoint were selected on validation, so validation metrics are optimistic — but a large
# val→test gap would reveal over-fitting to the selection.

# %%
vrows = []
for label in maps_val:
    thr = results[label]["threshold"]; r = evaluate(maps_val[label], gt_val, thr)
    vrows.append({"model": label, "val IoU": r["iou"], "test IoU": results[label]["iou"], "val AP": r["ap"], "test AP": results[label]["ap"]})
display(pd.DataFrame(vrows).set_index("model").round(3))

json.dump({k: {m: v for m, v in r.items() if m != "per_image_iou"} | {"iou_ci95": list(ci.get(k, (np.nan, np.nan)))}
           for k, r in results.items()}, open("results_test.json", "w"), indent=1, default=float)

# %% [markdown]
# ## 6. Discussion and conclusions
# *(The numbered findings below are written from the outputs above; see the technical report for the full
# discussion.)*
#
# 1. **ANN vs CNN** — on identical patches, compare rows "ANN (patches)" and "CNN classifier (patches)" in the table above.
# 2. **Patch classification vs segmentation** — compare the patch CNN with the U-Nets.
# 3. **Transfer learning** — compare "U-Net R34 (random init)" with "U-Net R34 (ImageNet)".
# 4. **Supervised vs unsupervised** — compare the segmentation networks with the autoencoder rows.
# 5. **Limits** — 11 test images (see the bootstrap confidence intervals), supervised models only recognise anomaly *types* seen in training, small and distant objects are the main failure cases.
