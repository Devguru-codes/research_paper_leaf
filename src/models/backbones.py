"""Appearance backbones exposing multi-scale features.

Every backbone returns a dict:
  early  (B, Ce, H/4,  W/4)   -> vein head
  mid    (B, Cm, h_m,  w_m)   -> structural weight MLP g
  patch  (B, P,  Cp) + hw     -> ESA keys/values
  global (B, Cg)              -> classifier
Origins (per the project's tooling constraints): ConvNeXt (Meta), EfficientNet (Google), ViT AugReg (Google),
DINOv2 (Meta), BioCLIP 2 (Imageomics / Ohio State).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


class Renorm(nn.Module):
    """Inputs arrive ImageNet-normalised; convert to the backbone's own pre-training statistics."""

    def __init__(self, mean, std):
        super().__init__()
        im, istd = torch.tensor(_IMNET_MEAN).view(1, 3, 1, 1), torch.tensor(_IMNET_STD).view(1, 3, 1, 1)
        m, sd = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("a", istd / sd, persistent=False)
        self.register_buffer("b", (im - m) / sd, persistent=False)

    def forward(self, x):
        return x * self.a.to(x.dtype) + self.b.to(x.dtype)


class ConvNeXtTV(nn.Module):
    """torchvision ConvNeXt-T/S (ImageNet-1k weights)."""

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        fn = getattr(tvm, name)
        weights = "DEFAULT" if pretrained else None
        m = fn(weights=weights)
        self.stem_s1 = m.features[:2]   # stride 4,  96
        self.s2 = m.features[2:4]       # stride 8,  192
        self.s3 = m.features[4:6]       # stride 16, 384
        self.s4 = m.features[6:8]       # stride 32, 768
        self.norm = m.classifier[0]     # LayerNorm2d
        c = [96, 192, 384, 768] if "tiny" in name or "small" in name else [128, 256, 512, 1024]
        self.dims = dict(early=c[0], mid=c[1], patch=c[2], glob=c[3])

    def forward(self, x):
        e = self.stem_s1(x)
        m = self.s2(e)
        p = self.s3(m)
        g = self.norm(self.s4(p)).mean((2, 3))
        B, C, h, w = p.shape
        return dict(early=e, mid=m, patch=p.flatten(2).transpose(1, 2), hw=(h, w), glob=g)


class EfficientNetTV(nn.Module):
    """torchvision EfficientNet-B0 (Google; ImageNet-1k weights): second CNN family."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        m = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        f = m.features
        self.early = f[:3]   # stride 4,  24
        self.mid = f[3]      # stride 8,  40
        self.patch = f[4:6]  # stride 16, 112
        self.top = f[6:9]    # stride 32, 1280
        self.dims = dict(early=24, mid=40, patch=112, glob=1280)

    def forward(self, x):
        e = self.early(x)
        m = self.mid(e)
        p = self.patch(m)
        g = self.top(p).mean((2, 3))
        B, C, h, w = p.shape
        return dict(early=e, mid=m, patch=p.flatten(2).transpose(1, 2), hw=(h, w), glob=g)


class TimmViT(nn.Module):
    """timm ViT (AugReg ViT-S/16 or DINOv2 ViT-S/14); multi-scale from intermediate blocks."""

    def __init__(self, name: str = "vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained: bool = True,
                 img_size: int = 384, early_block: int = 3, mid_block: int = 6):
        super().__init__()
        import timm
        kw = dict(pretrained=pretrained, num_classes=0)
        if "dinov2" in name:
            kw["img_size"] = img_size
        self.m = timm.create_model(name, **kw)
        cfg = self.m.pretrained_cfg
        self.renorm = Renorm(cfg.get("mean", _IMNET_MEAN), cfg.get("std", _IMNET_STD))
        self.eb, self.mb = early_block, mid_block
        d = self.m.embed_dim
        self.dims = dict(early=d, mid=d, patch=d, glob=d)
        self.npre = self.m.num_prefix_tokens

    def _grid(self, t, hw):
        B, N, C = t.shape
        return t[:, self.npre:].transpose(1, 2).reshape(B, C, *hw)

    def forward(self, x):
        m = self.m
        H, W = x.shape[-2:]
        ps = m.patch_embed.patch_size
        hw = (H // ps[0], W // ps[1])
        t = m.patch_embed(self.renorm(x))
        t = m._pos_embed(t)
        t = m.norm_pre(t)
        early = mid = None
        for i, blk in enumerate(m.blocks):
            t = blk(t)
            if i == self.eb:
                early = self._grid(t, hw)
            if i == self.mb:
                mid = self._grid(t, hw)
        t = m.norm(t)
        glob = t[:, 0] if self.npre else t.mean(1)
        early = F.interpolate(early, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        return dict(early=early, mid=mid, patch=t[:, self.npre:], hw=hw, glob=glob)


class BioCLIP2(nn.Module):
    """BioCLIP 2 visual tower (ViT-L/14, open_clip). Frozen except the last `train_blocks` blocks."""

    def __init__(self, train_blocks: int = 2, early_block: int = 5, mid_block: int = 11):
        super().__init__()
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        self.v = model.visual
        norm = [t for t in getattr(preprocess, "transforms", []) if hasattr(t, "mean")]
        mean, std = (norm[0].mean, norm[0].std) if norm else ((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))
        self.renorm = Renorm(mean, std)
        self.lnd = not getattr(self.v.transformer, "batch_first", True)  # older open_clip: (L, N, D)
        for p in self.v.parameters():
            p.requires_grad = False
        blocks = self.v.transformer.resblocks
        for blk in blocks[len(blocks) - train_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.v.ln_post.parameters():
            p.requires_grad = True
        self.eb, self.mb = early_block, mid_block
        d = self.v.transformer.width
        self.dims = dict(early=d, mid=d, patch=d, glob=d)

    def forward(self, x):
        v = self.v
        H, W = x.shape[-2:]
        t = v.conv1(self.renorm(x))  # (B, C, h, w)
        hw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(t.dtype) + torch.zeros(t.shape[0], 1, t.shape[-1], dtype=t.dtype, device=t.device)
        t = torch.cat([cls, t], 1) + v.positional_embedding.to(t.dtype)
        t = v.ln_pre(t)
        early = mid = None
        if self.lnd:
            t = t.transpose(0, 1)
        for i, blk in enumerate(v.transformer.resblocks):
            t = blk(t)
            tb = t.transpose(0, 1) if self.lnd else t
            if i == self.eb:
                early = tb[:, 1:].transpose(1, 2).reshape(tb.shape[0], -1, *hw)
            if i == self.mb:
                mid = tb[:, 1:].transpose(1, 2).reshape(tb.shape[0], -1, *hw)
        t = v.ln_post(t.transpose(0, 1) if self.lnd else t)
        early = F.interpolate(early, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        return dict(early=early, mid=mid, patch=t[:, 1:], hw=tuple(hw), glob=t[:, 0])


def build_backbone(name: str, pretrained: bool = True, img_size: int = 384) -> nn.Module:
    if name.startswith("convnext"):
        return ConvNeXtTV(name, pretrained)
    if name == "efficientnet_b0":
        return EfficientNetTV(pretrained)
    if name == "vit_small":
        return TimmViT("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained, img_size)
    if name == "dinov2_small":
        return TimmViT("vit_small_patch14_dinov2.lvd142m", pretrained, img_size)
    if name == "bioclip2":
        return BioCLIP2()
    raise ValueError(name)
