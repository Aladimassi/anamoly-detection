"""Model zoo used in the project.

1. UNetScratch      - classic U-Net (Ronneberger et al., 2015), randomly initialised.
                     Baseline CNN: tests what can be learnt from 38 images alone.
2. unet_resnet34    - U-Net decoder on an ImageNet-pretrained ResNet-34 encoder (transfer learning).
3. deeplab_resnet34 - DeepLabV3+ (Chen et al., 2018) with the same pretrained encoder:
                     atrous spatial pyramid pooling gives multi-scale context.
4. ConvAutoencoder  - unsupervised/complementary approach: learns to reconstruct *normal* road
                     content; anomalies are expected to be reconstructed poorly.
"""
from __future__ import annotations

import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet ResNet-34 weights (timm release, torchvision-compatible key names)
RESNET34_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth"


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNetScratch(nn.Module):
    def __init__(self, base=32, depth=4, dropout=0.2):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]  # 32, 64, 128, 256, 512
        self.downs = nn.ModuleList([_conv_block(3, chs[0])] + [_conv_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_conv_block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.drop(skips.pop())
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def load_imagenet_resnet34(encoder: nn.Module, cache_dir="weights"):
    os.makedirs(cache_dir, exist_ok=True)
    sd = torch.hub.load_state_dict_from_url(RESNET34_URL, model_dir=cache_dir, map_location="cpu", progress=False)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    own = encoder.state_dict()
    assert set(sd) <= set(own), set(sd) - set(own)
    own.update(sd)
    nn.Module.load_state_dict(encoder, own, strict=True)
    assert torch.equal(encoder.conv1.weight.detach().cpu(), sd["conv1.weight"])  # weights really loaded
    return encoder


def unet_resnet34(pretrained=True):
    m = smp.Unet("resnet34", encoder_weights=None, classes=1, decoder_attention_type="scse")
    if pretrained:
        load_imagenet_resnet34(m.encoder)
    return m


def deeplab_resnet34(pretrained=True):
    m = smp.DeepLabV3Plus("resnet34", encoder_weights=None, classes=1)
    if pretrained:
        load_imagenet_resnet34(m.encoder)
    return m


class ConvAutoencoder(nn.Module):
    """Under-complete convolutional AE: 3x256x448 -> 128x16x28 bottleneck -> 3x256x448."""

    def __init__(self, ch=(32, 64, 128, 128), bottleneck=128):
        super().__init__()
        enc, cin = [], 3
        for c in ch:
            enc += [nn.Conv2d(cin, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True)]
            cin = c
        enc += [nn.Conv2d(cin, bottleneck, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        dec, cin = [], bottleneck
        for c in reversed(ch):
            dec += [nn.ConvTranspose2d(cin, c, 4, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            cin = c
        dec += [nn.Conv2d(cin, 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PerceptualExtractor(nn.Module):
    """Frozen ImageNet ResNet-34 stem + layer1 + layer2, used to measure the discrepancy
    between an image and its reconstruction in feature space (idea borrowed from the
    discrepancy network of Lis et al., 2019)."""

    def __init__(self):
        super().__init__()
        enc = smp.Unet("resnet34", encoder_weights=None).encoder
        load_imagenet_resnet34(enc)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.l1, self.l2 = enc.layer1, enc.layer2
        self.eval().requires_grad_(False)

    def forward(self, x):
        f1 = self.l1(self.stem(x))
        f2 = self.l2(f1)
        return [f1, f2]


PATCH = 32  # patch size (pixels at 448x256) used by the two "course-level" baselines below


class PatchANN(nn.Module):
    """Fully-connected network (multilayer perceptron) on a flattened 32x32x3 patch.
    Input 3072 -> 512 -> 128 -> 1 (probability that the patch is anomalous).
    It has no notion of spatial neighbourhood: every pixel position is a separate input."""

    def __init__(self, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, cin = [nn.Flatten()], 3 * PATCH * PATCH
        for h in hidden:
            layers += [nn.Linear(cin, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            cin = h
        layers += [nn.Linear(cin, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchCNN(nn.Module):
    """Classic CNN classifier (as seen in class): 3 x [Conv3x3 - BN - ReLU - MaxPool] then Dense layers.
    32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 -> Flatten(2048) -> Dense(128) -> Dense(1).
    Weight sharing + local receptive fields make it translation-equivariant, unlike the ANN."""

    def __init__(self, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
