"""ES-Former and its comparison variants (identical backbone, vein head and training).

variant:
  plain     backbone global token -> classifier
  global    + GlobalTopoFusion (Delta chi grid, globally fused)
  strip     + strip pooling over image rows/columns (control)
  maskpool  + mask-weighted mean of patch tokens over the leaf (control: leaf location only)
  slabstat  + slab attention with non-topological slab statistics (control)
  esa       + Euler-Sweep Attention (proposed)
All variants train the same vein head with the same auxiliary loss, so differences are due
to how (and whether) structure enters the representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.esa import EulerSweepAttention, GlobalTopoFusion, MaskPoolFusion, StripPoolFusion


class VeinHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


class ESFormer(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "convnext_tiny", variant: str = "esa",
                 pretrained: bool = True, img_size: int = 384, grid: int = 96, d: int = 192,
                 K: int = 32, T: int = 16, R: float = 2.0, kappa: float = 4.0, nbins: int = 256,
                 heads: int = 4, depth: int = 2, weights: str = "learned", attn: str = "slab",
                 use_blade: bool = True, use_vein: bool = True, drop: float = 0.2, token: str | None = None):
        super().__init__()
        self.variant, self.grid = variant, grid
        self.backbone = build_backbone(backbone, pretrained, img_size)
        dims = self.backbone.dims
        self.vein_head = VeinHead(dims["early"])
        if variant == "plain":
            self.fusion = None
        elif variant == "global":
            self.fusion = GlobalTopoFusion(dims["mid"], d, K, T, R, kappa, nbins, weights, token or "euler")
        elif variant == "strip":
            self.fusion = StripPoolFusion(dims["patch"], d, heads, depth)
        elif variant == "maskpool":
            self.fusion = MaskPoolFusion(dims["patch"], d)
        elif variant in ("esa", "slabstat"):
            self.fusion = EulerSweepAttention(dims["patch"], dims["mid"], d, K, T, R, kappa, nbins, heads, depth,
                                              token=token or ("euler" if variant == "esa" else "stats"), attn=attn,
                                              weights=weights, use_blade=use_blade, use_vein=use_vein)
        else:
            raise ValueError(variant)
        fdim = dims["glob"] + (self.fusion.out_dim if self.fusion is not None else 0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(fdim, num_classes))
        self.feat_dim = fdim

    def forward(self, img, mask, return_aux: bool = False):
        """img (B,3,H,W) normalised; mask (B,H,W) binary leaf mask at input resolution."""
        f = self.backbone(img)
        G = self.grid
        blade = (F.interpolate(mask[:, None].float(), size=(G, G), mode="area")[:, 0] > 0.5).float()
        vein_logit = self.vein_head(f["early"])
        if vein_logit.shape[-1] != G:
            vein_logit = F.interpolate(vein_logit[:, None], size=(G, G), mode="bilinear", align_corners=False)[:, 0]
        vein = torch.sigmoid(vein_logit.float()) * blade
        z = f["glob"]
        extra = {}
        if self.fusion is not None:
            pm = F.interpolate(mask[:, None].float(), size=f["hw"], mode="area").flatten(1)
            out = self.fusion(f["patch"], f["hw"], f["mid"], blade, vein, pm, return_attn=return_aux)
            if return_aux:
                out, extra = out
            z = torch.cat([z, out.to(z.dtype)], 1)
        logits = self.head(z)
        if return_aux:
            return logits, dict(vein_logit=vein_logit, blade=blade, embedding=z, **extra)
        return logits, vein_logit
