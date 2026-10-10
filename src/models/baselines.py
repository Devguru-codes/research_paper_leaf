"""Reproductions of prior medicinal-leaf pipelines (P1).

DualBranch: RGB MobileNetV2 + vein-skeleton DenseNet121 with late feature fusion (after the
dual-branch appearance/vein-morphology design, Frontiers in AI 2026). The skeleton is the
thinned classical ridge-filter vein map, computed from the image (deterministic preprocessing).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class DualBranch(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        w = "DEFAULT" if pretrained else None
        self.rgb = tvm.mobilenet_v2(weights=w).features          # 1280
        self.skel = tvm.densenet121(weights=w).features          # 1024
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(1280 + 1024, num_classes))

    def forward(self, img, mask, skel=None):
        a = self.rgb(img).mean((2, 3))
        s = torch.relu(self.skel(skel[:, None].expand(-1, 3, -1, -1))).mean((2, 3))
        return self.head(torch.cat([a, s], 1)), None
