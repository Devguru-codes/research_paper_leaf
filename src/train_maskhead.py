"""Train the own leaf masker on composites: our segmented leaves pasted on DTD textures / procedural
backgrounds with random scale, rotation, position, shadows and colour jitter.

  python -m src.train_maskhead --epochs 12 --out results/maskhead
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.corruptions import _procedural_bg
from src.data.dataset import MEAN, STD, load_item
from src.models.mask_head import LeafMasker
from src.paths import cache_root, results_root

EXCLUDE = {"veined", "paisley", "frilly", "lacelike"}  # leaf-like DTD categories
SIZE = 448


def dtd_files():
    fs = glob.glob("/kaggle/input/**/dtd/images/*/*.jpg", recursive=True) or \
         glob.glob(os.environ.get("DTD_ROOT", "__none__") + "/**/*.jpg", recursive=True)
    return [f for f in fs if Path(f).parent.name not in EXCLUDE]


def composite(img, mask, bg, rng):
    """Paste the leaf (random scale/rotation/position) on background bg (SIZE x SIZE)."""
    s = rng.uniform(0.35, 0.95)
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, s)
    M[:, 2] += rng.uniform(-0.2, 0.2, 2) * SIZE
    li = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=(255, 255, 255))
    lm = cv2.warpAffine(mask.astype(np.float32), M, (SIZE, SIZE))
    if rng.random() < 0.5:  # cast shadow
        sh = cv2.GaussianBlur(np.roll(lm, rng.integers(3, 15, 2), (0, 1)), (0, 0), 6)[..., None]
        bg = (bg * (1 - 0.4 * sh)).astype(np.uint8)
    soft = cv2.GaussianBlur(lm, (0, 0), 0.8)[..., None]
    out = (soft * li + (1 - soft) * bg).astype(np.float32)
    out *= rng.uniform(0.7, 1.2)
    out += rng.normal(0, rng.uniform(0, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), (lm > 0.5).astype(np.float32)


class Composites(Dataset):
    def __init__(self, stems, bgs, n, seed):
        self.stems, self.bgs, self.n, self.seed = stems, bgs, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i + random.randint(0, 10 ** 9))
        img, m, _ = load_item(self.stems[rng.integers(len(self.stems))])
        if self.bgs and rng.random() < 0.8:
            b = cv2.cvtColor(cv2.imread(self.bgs[rng.integers(len(self.bgs))]), cv2.COLOR_BGR2RGB)
            y = rng.integers(0, max(1, b.shape[0] - SIZE)); x = rng.integers(0, max(1, b.shape[1] - SIZE))
            b = cv2.resize(b[y:y + SIZE * 2, x:x + SIZE * 2], (SIZE, SIZE))
        else:
            b = _procedural_bg((SIZE, SIZE, 3), int(rng.integers(1, 4)), rng)
        x, y = composite(img, (m > 127).astype(np.float32), b, rng)
        t = (torch.from_numpy(x).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return t, torch.from_numpy(y)


def iou(logit, y):
    p = logit > 0
    inter = (p & (y > 0.5)).sum((1, 2)).float()
    union = (p | (y > 0.5)).sum((1, 2)).float().clamp_min(1)
    return (inter / union).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=str(results_root() / "maskhead"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cache_root() / "index.csv")
    df = df[df.keep]
    stems = df.stem.tolist()
    rng = np.random.default_rng(0)
    rng.shuffle(stems)
    val_stems, tr_stems = stems[:150], stems[150:]
    bgs = dtd_files()
    rng.shuffle(bgs)
    nb = len(bgs)
    tr_bg, va_bg = bgs[: int(0.9 * nb)], bgs[int(0.9 * nb):]
    print("DTD backgrounds:", nb, flush=True)
    tr = DataLoader(Composites(tr_stems, tr_bg, a.steps * a.bs, 1), batch_size=a.bs, num_workers=4, drop_last=True)
    va = DataLoader(Composites(val_stems, va_bg, 300, 2), batch_size=a.bs, num_workers=4)
    model = LeafMasker().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.epochs * len(tr))
    scaler = torch.amp.GradScaler()
    log = []
    for ep in range(a.epochs):
        model.train()
        t = time.time()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x)
            p = torch.sigmoid(lg.float())
            loss = F.binary_cross_entropy_with_logits(lg.float(), y) + \
                (1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in va:
                with torch.autocast("cuda", dtype=torch.float16):
                    ious.append(iou(model(x.to(dev)).float(), y.to(dev)))
        log.append(dict(epoch=ep, loss=loss.item(), val_iou=float(np.mean(ious)), sec=time.time() - t))
        print(json.dumps(log[-1]), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "maskhead.pt")
    (out / "maskhead_log.json").write_text(json.dumps(dict(log=log, n_dtd=nb, val_composite_iou=log[-1]["val_iou"]), indent=1))


if __name__ == "__main__":
    main()
