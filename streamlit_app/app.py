"""Road Anomaly Detector — Streamlit interface.

Run:  streamlit run app.py      (then open http://localhost:8501)

Model: DeepLabV3+ with a ResNet-34 encoder pretrained on ImageNet, fine-tuned on the RoadAnomaly
dataset (Lis et al., ICCV 2019), exported to ONNX and quantized to int8 (23 MB).
The ONNX graph takes a 1x3x256x448 RGB image in [0, 1] and returns a 1x1x256x448 anomaly probability;
ImageNet normalisation and the sigmoid are inside the graph.
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageFilter

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "deeplab_int8.onnx"
NET_W, NET_H = 448, 256
DEFAULT_THRESHOLD = 0.51  # chosen on the validation set (maximises pixel F1)

AMBER = np.array([232, 119, 14], dtype=np.float32)
TP_C, FP_C, FN_C = (np.array(c, dtype=np.float32) for c in ([31, 164, 90], [224, 53, 43], [45, 108, 223]))

SAMPLES = {  # test-set photos (never used for training): id -> (label, true anomaly area in %)
    "animals28": ("Moose", 8.02), "animals25": ("Boar family", 13.52), "cones06": ("Traffic cones", 4.99),
    "obstacles10": ("Boulder", 19.05), "vehicle03": ("Animal carts", 7.68), "animals15": ("Horse tram", 10.38),
    "animals20": ("Young boar", 4.32), "obstacles11": ("Boulder in snow", 22.72), "obstacles08": ("Distant boulder", 1.35),
    "animals11": ("Pheasant", 0.92), "lost": ("Lost tyre", 0.37),
}
COMPARISON = [("DeepLabV3+ R34 (ImageNet)", 0.748), ("U-Net R34 (ImageNet)", 0.568), ("U-Net (from scratch)", 0.268),
              ("Spatial prior, ignores the image", 0.216), ("U-Net R34 (random init)", 0.208), ("ANN on patches", 0.143),
              ("CNN on patches", 0.124), ("Autoencoder (perceptual error)", 0.100)]
# (plain Markdown tables: no pandas, whose compiled DLL Windows Smart App Control can block)


# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the network…")
def load_session():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])


def predict(img: Image.Image):
    """Returns the probability map at the photo's own resolution, and the inference time in ms."""
    x = np.asarray(img.convert("RGB").resize((NET_W, NET_H), Image.BILINEAR), dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    t0 = time.perf_counter()
    prob = load_session().run(None, {"image": x})[0][0, 0]
    ms = (time.perf_counter() - t0) * 1000
    prob_up = np.asarray(Image.fromarray(prob.astype(np.float32), mode="F").resize(img.size, Image.BILINEAR))
    return np.clip(prob_up, 0, 1), ms


# ----------------------------------------------------------------------------- rendering
def inferno(t):
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                      [250, 193, 39], [252, 255, 164]], dtype=np.float32)
    s = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(len(stops) - 2, np.floor(s).astype(int))
    f = (s - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def blend(base, mask, color, alpha):
    base[mask] = base[mask] * (1 - alpha) + color * alpha
    return base


def render(img, prob, thr, view, gt=None):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32).copy()
    pred = prob >= thr
    if view == "Detection":
        rgb[~pred] *= 0.8
        rgb = blend(rgb, pred, AMBER, 0.55)
        edge = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)) > 0
        rgb[edge] = AMBER
    elif view == "Probability":
        a = (0.25 + 0.6 * prob)[..., None]
        rgb = rgb * (1 - a) + inferno(prob) * a
    elif view == "vs. ground truth" and gt is not None:
        rgb[~pred & ~gt] *= 0.6
        rgb = blend(rgb, pred & gt, TP_C, 0.6)
        rgb = blend(rgb, pred & ~gt, FP_C, 0.6)
        rgb = blend(rgb, ~pred & gt, FN_C, 0.6)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


@st.cache_data(show_spinner=False)
def load_sample(sid):
    img = Image.open(ROOT / "samples" / f"{sid}.jpg").convert("RGB")
    gt = np.asarray(Image.open(ROOT / "samples" / f"{sid}_gt.png").convert("L").resize(img.size, Image.NEAREST)) > 127
    return img, gt


@st.cache_data(show_spinner=False, max_entries=20)
def cached_predict(key, _img):
    return predict(_img)


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Road Anomaly Detector", page_icon="🚧", layout="wide")
st.title("Road Anomaly Detector")
st.caption("Finds unexpected objects on the road (animals, rocks, cones, lost cargo) pixel by pixel. "
           "Deep Learning module project, ESPRIT — DeepLabV3+ trained on the RoadAnomaly dataset.")

with st.sidebar:
    st.header("Photo")
    source = st.radio("Source", ["Test-set example", "Upload my photo"], label_visibility="collapsed")
    if source == "Test-set example":
        sid = st.selectbox("Example (never used for training)", list(SAMPLES),
                           format_func=lambda k: f"{SAMPLES[k][0]} — true anomaly {SAMPLES[k][1]} %")
        img, gt = load_sample(sid)
        key, label = f"sample:{sid}", f"Test photo · {SAMPLES[sid][0]}"
    else:
        up = st.file_uploader("Road photo (JPEG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"])
        if up is None:
            st.info("Upload a photo to analyse it. Wide 16:9 road photos work best.")
            st.stop()
        img = Image.open(up).convert("RGB")
        if max(img.size) > 1600:  # keep the display fast
            img.thumbnail((1600, 1600))
        gt, key, label = None, f"upload:{up.name}:{up.size}", f"Your photo · {up.name}"

    st.header("Settings")
    thr = st.slider("Decision threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01,
                    help="A pixel is flagged when its probability is at least this value. "
                         "0.51 was chosen on the validation set.")
    views = ["Detection", "Probability", "vs. ground truth", "Photo"] if gt is not None else ["Detection", "Probability", "Photo"]
    view = st.radio("View", views)

prob, ms = cached_predict(key, img)
pred = prob >= thr
area = 100 * pred.mean()

col_img, col_res = st.columns([3, 1], gap="large")
with col_img:
    st.image(render(img, prob, thr, view, gt), caption=label, use_container_width=True)
    legends = {
        "Detection": "🟧 Flagged as anomaly (probability ≥ threshold)",
        "Probability": "Colour = anomaly probability, from dark (0) to yellow (1)",
        "vs. ground truth": "🟩 Correct detection · 🟥 False alarm · 🟦 Missed",
        "Photo": "Original photo",
    }
    st.caption(legends[view])

with col_res:
    if pred.any():
        st.error(f"**Anomaly detected** — {area:.1f} % of the frame" if area >= 0.1 else "**Anomaly detected** — under 0.1 % of the frame")
    else:
        st.success(f"**Road looks clear** — nothing above {thr:.2f}")
    st.metric("Flagged area", f"{area:.1f} %")
    st.metric("Peak probability", f"{prob.max():.2f}")
    if gt is not None:
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        st.metric("IoU vs. ground truth", f"{inter / union:.2f}" if union else "–")
    st.metric("Inference (CPU)", f"{ms:.0f} ms")

st.divider()
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.subheader("The model")
    st.markdown("**DeepLabV3+** with a **ResNet-34** encoder pretrained on ImageNet, fine-tuned on 38 annotated "
                "photos. Compressed to 8-bit integers (90 MB → 23 MB) with no loss on the test set.")
    st.markdown("| Test set, 11 photos | Value |\n|---|---:|\n| IoU | 0.73 |\n| F1 / Dice | 0.84 |\n"
                "| Precision · Recall | 0.88 · 0.81 |\n| AUROC · AP | 0.97 · 0.89 |")
with c2:
    st.subheader("Why this model")
    st.markdown("Seven approaches compared on the same test photos. Transfer learning made the difference.")
    st.markdown("| Model | Test IoU |\n|---|---:|\n" + "\n".join(
        f"| {'**' + m + '**' if i == 0 else m} | {'**%.2f**' % v if i == 0 else '%.2f' % v} |" for i, (m, v) in enumerate(COMPARISON)))
with c3:
    st.subheader("Limits")
    st.markdown("- Learned the anomaly *types* seen in training (animals, rocks, cones, tyres, carts); a new kind of obstacle may be missed.\n"
                "- People, cyclists and ordinary cars count as normal.\n"
                "- Small, distant objects (under 1 % of the frame) are the hardest.\n"
                "- Photos are resized to 448×256 for the network.")
st.caption("Example photos and annotations: RoadAnomaly dataset (Lis, Nakka, Fua, Salzmann, *Detecting the Unexpected "
           "via Image Resynthesis*, ICCV 2019), provided for research purposes.")
