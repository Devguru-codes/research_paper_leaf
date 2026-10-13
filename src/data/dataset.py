"""Leaf dataset over the preprocessing cache (448px leaf-centred crops + masks + vein maps)."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset

from src.paths import cache_root

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_item(stem: str, root=None):
    root = root or cache_root()
    img = cv2.cvtColor(cv2.imread(str(root / "img" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(root / "mask" / f"{stem}.png"), 0)
    vein = cv2.imread(str(root / "vein" / f"{stem}.png"), 0)
    return img, mask, vein


def to_tensors(img: np.ndarray, mask: np.ndarray, vein: np.ndarray):
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    m = torch.from_numpy((mask > 127).astype(np.float32))[None]
    v = torch.from_numpy(vein.astype(np.float32) / 255.0)[None]
    return x, m, v


def geometric(x, m, v, angle: float, scale: float, translate=(0, 0), hflip=False):
    """Apply the same rigid transform to image (white fill), mask (nearest) and vein (bilinear)."""
    if hflip:
        x, m, v = TF.horizontal_flip(x), TF.horizontal_flip(m), TF.horizontal_flip(v)
    kw = dict(angle=angle, translate=list(translate), scale=scale, shear=[0.0, 0.0])
    x = TF.affine(x, **kw, interpolation=TF.InterpolationMode.BILINEAR, fill=1.0)
    m = TF.affine(m, **kw, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
    v = TF.affine(v, **kw, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    return x, m, v


def finalize(x, m, v, size: int):
    if x.shape[-1] != size:
        x = TF.resize(x, [size, size], antialias=True)
        m = (TF.resize(m, [size, size], antialias=True) > 0.5).float()
        v = TF.resize(v, [size, size], antialias=True)
    return (x - MEAN) / STD, m[0], v[0]


class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices, size: int = 384, train: bool = False, root=None,
                 skeleton: bool = False, raw_uint8: bool = False):
        self.df, self.idx, self.size, self.train, self.skeleton = df, np.asarray(indices), size, train, skeleton
        self.raw_uint8 = raw_uint8  # return un-augmented uint8 tensors (GPU augmentation path)
        self.root = root or cache_root()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.df.loc[self.idx[i]]
        if self.raw_uint8:
            img, mask, vein = load_item(r.stem, self.root)
            return dict(img=torch.from_numpy(img).permute(2, 0, 1).contiguous(), mask=torch.from_numpy(mask),
                        vein=torch.from_numpy(vein), label=int(r.label), index=int(self.idx[i]))
        x, m, v = to_tensors(*load_item(r.stem, self.root))
        if self.train:
            W = x.shape[-1]
            x, m, v = geometric(x, m, v, angle=random.uniform(-180, 180), scale=random.uniform(0.85, 1.15),
                                translate=(random.uniform(-0.05, 0.05) * W, random.uniform(-0.05, 0.05) * W),
                                hflip=random.random() < 0.5)
            x = TF.adjust_brightness(x, random.uniform(0.8, 1.2))
            x = TF.adjust_contrast(x, random.uniform(0.8, 1.2))
            x = TF.adjust_saturation(x, random.uniform(0.8, 1.2))
            x = TF.adjust_hue(x, random.uniform(-0.02, 0.02)).clamp(0, 1)
        x, m, v = finalize(x, m, v, self.size)
        out = dict(img=x, mask=m, vein=v, label=int(r.label), index=int(self.idx[i]))
        if self.skeleton:  # classical vein skeleton (input of the dual-branch baseline)
            from src.data.venation import vein_skeleton
            out["skel"] = torch.from_numpy(vein_skeleton(v.numpy()).astype(np.float32))
        return out
