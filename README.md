# Road Anomaly Detection

Pixel-level detection of unexpected obstacles on the road (animals, rocks, traffic cones, lost cargo,
unusual vehicles) on the **RoadAnomaly** dataset, a Deep Learning module project at ESPRIT (2026).

Seven deep-learning approaches are compared, from a fully connected network to transfer-learning
segmentation networks and an autoencoder. The best model, **DeepLabV3+ with an ImageNet-pretrained
ResNet-34 encoder, reaches IoU 0.75 / F1 0.86 / AUROC 0.98** on held-out test images. It comes with
two interfaces: a Streamlit app and a browser-only web page.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aladimassi/anamoly-detection/blob/main/notebooks/RoadAnomaly_Detection.ipynb)

**Live demo:** https://aladimassi.github.io/anamoly-detection/ (runs entirely in the browser)

## Results (test set, 11 images never used for training)

| # | Model | Precision | Recall | F1 = Dice | IoU [95 % CI] | AUROC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ANN on 32×32 patches | 0.20 | 0.32 | 0.25 | 0.14 [0.07–0.25] | 0.77 | 0.19 |
| 2 | CNN classifier on the same patches | 0.22 | 0.22 | 0.22 | 0.12 [0.05–0.22] | 0.76 | 0.20 |
| 3 | U-Net from scratch | 0.48 | 0.38 | 0.42 | 0.27 [0.14–0.41] | 0.88 | 0.44 |
| 4 | U-Net, ResNet-34, random init | 0.31 | 0.39 | 0.34 | 0.21 [0.11–0.36] | 0.84 | 0.37 |
| 5 | U-Net, ResNet-34, **ImageNet** | **0.90** | 0.60 | 0.72 | 0.57 [0.43–0.74] | 0.95 | 0.81 |
| 6 | **DeepLabV3+, ResNet-34, ImageNet** | 0.88 | **0.84** | **0.86** | **0.75 [0.60–0.84]** | **0.98** | **0.93** |
| 7 | Convolutional autoencoder (perceptual error) | 0.12 | 0.37 | 0.18 | 0.10 [0.08–0.12] | 0.62 | 0.14 |
| – | Reference: spatial prior, ignores the image | 0.23 | 0.78 | 0.36 | 0.22 [0.12–0.32] | 0.83 | 0.23 |

Pixel-level metrics on the original 1280×720 masks, decision threshold chosen on the validation set,
95 % intervals from a bootstrap over test images.

**Main findings**
- **Transfer learning is decisive:** the same U-Net goes from IoU 0.21 (random weights) to 0.57 (ImageNet weights).
  Without pretraining, no model clearly beats the no-image spatial prior with only 38 training images.
- **ANN vs CNN:** on identical patches they tie (IoU 0.14 vs 0.12). The CNN gets there with 4.6× fewer parameters;
  a 32×32 patch lacks the context to tell a rock on the road from rocks on the verge.
- **The autoencoder fails** (AUROC 0.45 in RGB space, 0.62 in feature space): its reconstruction error tracks
  texture, not novelty, which is why Lis et al. condition their resynthesis on semantic labels.
- **Limitation:** models 1–6 are supervised, so they learn the anomaly *types* seen in training.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/RoadAnomaly_Detection.ipynb` | Self-contained Colab notebook: exploration, preprocessing, all 7 models, evaluation, figures |
| `roadanomaly/` | Python package: `data.py` (loading, label remap, split, augmentation), `models.py`, `metrics.py`, `train.py` |
| `run_experiments.py` | Trains every model from the command line |
| `road_anomaly_detection.py` / `build_colab.py` | Source of the notebook and the script that assembles the Colab version |
| `export_onnx.py` | Exports DeepLabV3+ to ONNX, quantizes it to int8 (90 → 23 MB) and checks the test metrics are unchanged |
| `streamlit_app/` | Streamlit interface (int8 ONNX model, 11 test photos with ground truth) |
| `docs/` | Static web version served by GitHub Pages (ONNX Runtime Web, WebAssembly) |

## Dataset

RoadAnomaly: K. Lis, K. Nakka, P. Fua, M. Salzmann, *Detecting the Unexpected via Image Resynthesis*, ICCV 2019.
60 images (1280×720) with pixel masks. **Anomalies are encoded as 2 in `labels_semantic.png`, not 1**, so the code
remaps `mask == 2` to 1. The dataset is not included here: download it from the authors' page and place
`RoadAnomaly.zip` in Google Drive (Colab) or unzip it to `data/RoadAnomaly/` (local).
The example photos in `streamlit_app/` and `docs/` come from this dataset and are provided for research purposes.

## How to run

**Notebook (recommended):** open it in Colab with the badge above, choose *Runtime → T4 GPU*, put
`RoadAnomaly.zip` in `MyDrive/`, then *Run all* (about 15–20 min).

**Training from the command line:**
```bash
pip install -r requirements.txt
python run_experiments.py --data data/RoadAnomaly
python export_onnx.py data/RoadAnomaly onnx/
```

**Streamlit app:**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
```
On Windows you can double-click `streamlit_app/run.bat`, which creates a virtual environment on first run.

**Web version:** serve `docs/` with any static server (`python -m http.server` inside `docs/`), or enable
GitHub Pages on the `docs/` folder. Opening `index.html` directly from disk does not work.

## Method in brief

- Stratified split per category (seed 42): 38 train / 11 validation / 11 test.
- Images resized to 448×256 (16:9, divisible by 32); predictions upsampled back to 1280×720 for evaluation.
- Augmentation: random zoom-crop, horizontal flip, brightness/contrast/saturation jitter.
- Loss: BCE + soft Dice; AdamW with warm-up and cosine decay; best checkpoint on validation IoU;
  pretrained encoders fine-tuned with a 10× smaller learning rate.
- No RNN: the data are independent still images, with no sequence to model.
