"""Assemble a single self-contained Colab notebook from the package modules + experiment script.
The module source is inlined verbatim, so the notebook runs exactly the code used for the reported results."""
import re
import sys

import jupytext
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

exp = jupytext.read("road_anomaly_detection.py")
cells = exp.cells
title = cells[0]
title.source += """

---
### ▶ How to run this notebook in Google Colab
1. **Runtime → Change runtime type → GPU (T4)**. Training everything takes ≈15–20 min on a T4 (≈3 h on CPU).
2. Put `RoadAnomaly.zip` in your Google Drive, in `MyDrive/` (or change `DATA_ZIP` below).
3. **Runtime → Run all**. Drive is mounted; checkpoints, figures and result tables are saved in
   `MyDrive/RoadAnomaly_project/`, so a second run reloads the trained models instead of retraining.
   Set `TRAIN = True` to force retraining.
"""

setup = code('''# ===== 0. Environment set-up (Colab or local Jupyter) =====
import os, sys, zipfile
IN_COLAB = "google.colab" in sys.modules
%pip install -q segmentation-models-pytorch==0.5.0

DATA_ZIP = "/content/drive/MyDrive/RoadAnomaly.zip"       # <- where you put the dataset zip
WORK_DIR = "/content/drive/MyDrive/RoadAnomaly_project"   # <- checkpoints / figures / results go here
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
else:  # local execution: run next to the data folder
    WORK_DIR = os.environ.get("WORK_DIR", os.getcwd())
os.makedirs(WORK_DIR, exist_ok=True); os.chdir(WORK_DIR)

DATA_ROOT = os.environ.get("ROADANOMALY_ROOT", "/content/RoadAnomaly" if IN_COLAB else "data/RoadAnomaly")
if not os.path.exists(os.path.join(DATA_ROOT, "frame_list.json")):
    if not os.path.exists(DATA_ZIP) and IN_COLAB:   # fallback: upload the zip by hand
        from google.colab import files
        DATA_ZIP = list(files.upload().keys())[0]
    with zipfile.ZipFile(DATA_ZIP) as z:
        z.extractall(os.path.dirname(DATA_ROOT))
print("data:", DATA_ROOT, "| working dir:", os.getcwd())''')

module_md = {
    "data": "## Code — data loading and preprocessing\n`load_dataset` reads every frame, **remaps the anomaly label 2 → 1**, checks image/mask sizes, and caches a 640×360 copy; `RoadAnomalyDataset` applies augmentation and ImageNet normalisation.",
    "models": "## Code — models\nAll models of the study: patch ANN, patch CNN, U-Net from scratch, U-Net / DeepLabV3+ on a ResNet-34 encoder (optionally ImageNet-pretrained) and the convolutional autoencoder.",
    "metrics": "## Code — loss and evaluation metrics\nBCE + Dice loss; precision, recall, F1, IoU, Dice from the pixel confusion matrix; AUROC / AP / FPR@95%TPR; validation-based threshold selection.",
    "train": "## Code — training and inference loops\nGeneric training loop (AdamW, warm-up + cosine schedule, best-validation checkpointing), sliding-window inference for the patch models, and autoencoder training/scoring.",
}
module_cells = []
for name, desc in module_md.items():
    src = open(f"roadanomaly/{name}.py").read()
    src = re.sub(r"^from __future__ import annotations\n", "", src, flags=re.M)
    src = re.sub(r"^from \.\w+ import \([^)]*\)\n|^from \.\w+ import [^\n]*\n", "", src, flags=re.M)
    module_cells += [md(desc), code(src.strip())]

# experiment cells: drop the local-package import cell, replace with plain config
imports = code('''import json, time, warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display
warnings.filterwarnings("ignore")

TRAIN = False   # False: reuse checkpoints found in WORK_DIR/checkpoints (models without a checkpoint are trained)
                # True : retrain every model from scratch
FIG = "figures"; os.makedirs(FIG, exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count())
seed_everything(42)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("device:", DEVICE, "| torch", torch.__version__)''')
first_code = next(i for i, c in enumerate(cells) if c.cell_type == "code")
assert "from roadanomaly" in cells[first_code].source
body = cells[1:first_code] + [imports] + cells[first_code + 1:]

nb = nbf.v4.new_notebook()
nb.cells = [title, setup] + module_cells + body
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "accelerator": "GPU",
               "colab": {"provenance": [], "gpuType": "T4"}}
out = sys.argv[1] if len(sys.argv) > 1 else "RoadAnomaly_Colab.ipynb"
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
