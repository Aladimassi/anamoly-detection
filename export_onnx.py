"""Export the trained DeepLabV3+ (ResNet-34, ImageNet init) to ONNX for in-browser inference,
quantize it to int8 (static, QDQ, calibrated on training images), and check that the compressed
model gives the same results as PyTorch on the test set."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from roadanomaly.data import IMAGENET_MEAN, IMAGENET_STD, IMG_H, IMG_W, RoadAnomalyDataset, load_dataset, stratified_split
from roadanomaly.metrics import best_f1_threshold, evaluate
from roadanomaly.models import deeplab_resnet34

DATA = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count())


class Deployable(nn.Module):
    """Input: 1x3x256x448 RGB in [0,1]. Output: 1x1x256x448 anomaly probability.
    ImageNet normalisation and the sigmoid are inside the graph, so the browser only resizes."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.net((x - self.mean) / self.std))


net = deeplab_resnet34(pretrained=False)
net.load_state_dict(torch.load("checkpoints/deeplabv3p_r34_imagenet.pt", map_location="cpu"))
model = Deployable(net).eval()

S = load_dataset(DATA, keep_full=True)
tr, va, te = stratified_split(S)


def raw(samples):  # [0,1] RGB tensors at network resolution, no normalisation
    ds = RoadAnomalyDataset(samples)
    return [torch.from_numpy(((ds[i][0].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN)
                              .transpose(2, 0, 1)).astype(np.float32))[None] for i in range(len(ds))]


def up(p):
    return F.interpolate(torch.as_tensor(p), size=(720, 1280), mode="bilinear", align_corners=False)[0, 0].numpy()


# ---- PyTorch reference (no flip TTA, same as the browser) ----
with torch.no_grad():
    pt_val = [up(model(x)) for x in raw([S[i] for i in va])]
    pt_test = [up(model(x)) for x in raw([S[i] for i in te])]
gt_val = [S[i].mask_full for i in va]; gt_test = [S[i].mask_full for i in te]
thr, _ = best_f1_threshold(pt_val, gt_val)
r_pt = evaluate(pt_test, gt_test, thr)

# ---- ONNX export ----
fp32 = os.path.join(OUT, "deeplab_fp32.onnx")
torch.onnx.export(model, torch.rand(1, 3, IMG_H, IMG_W), fp32, input_names=["image"], output_names=["prob"],
                  opset_version=17, dynamo=False)

# ---- int8 static quantization (QDQ), calibrated on training images ----
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

pre = os.path.join(OUT, "deeplab_pre.onnx")
quant_pre_process(fp32, pre)
calib = raw([S[i] for i in tr])


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter([{"image": x.numpy()} for x in calib])

    def get_next(self):
        return next(self.it, None)


int8 = os.path.join(OUT, "deeplab_int8.onnx")
quantize_static(pre, int8, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True)

import onnxruntime as ort

res = {"threshold_val": thr, "pytorch": {k: r_pt[k] for k in ["precision", "recall", "f1", "iou", "auroc", "ap"]}}
for name, path in [("onnx_fp32", fp32), ("onnx_int8", int8)]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    maps = [up(sess.run(None, {"image": x.numpy()})[0]) for x in raw([S[i] for i in te])]
    r = evaluate(maps, gt_test, thr)
    res[name] = {k: r[k] for k in ["precision", "recall", "f1", "iou", "auroc", "ap"]}
    res[name]["size_mb"] = os.path.getsize(path) / 1e6
    res[name]["max_abs_diff_vs_pytorch"] = float(max(np.abs(a - b).max() for a, b in zip(maps, pt_test)))
json.dump(res, open(os.path.join(OUT, "export_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
