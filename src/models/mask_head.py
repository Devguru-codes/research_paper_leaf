"""Own leaf auto-masker (for cross-domain P7 and background corruptions).

ConvNeXt-T (ImageNet-1k, torchvision) encoder + light FPN decoder -> leaf probability at 1/4
resolution, upsampled. Trained only on our segmented leaves composited onto textures (DTD) and
procedural backgrounds; no third-party segmentation model is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class LeafMasker(nn.Module):
    def __init__(self, pretrained: bool = True, ch: int = 96):
        super().__init__()
        m = tvm.convnext_tiny(weights="DEFAULT" if pretrained else None)
        self.s1, self.s2, self.s3, self.s4 = m.features[:2], m.features[2:4], m.features[4:6], m.features[6:8]
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in (96, 192, 384, 768)])
        self.out = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(), nn.Conv2d(ch, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        p = self.lat[3](f4)
        for f, lat in ((f3, self.lat[2]), (f2, self.lat[1]), (f1, self.lat[0])):
            p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        return F.interpolate(self.out(p), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
