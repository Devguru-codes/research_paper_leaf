"""Parameters, GFLOPs and latency of every backbone x model (reviewer request).

  python -m src.cost --out results/cost.csv
GFLOPs are counted with torch.utils.flop_counter (multiply-add = 2 FLOPs) for one image at the training
resolution. Latency: median of 30 forward passes, batch 1, after 5 warm-ups, on the local device.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.esformer import ESFormer

MODELS = {"plain": dict(variant="plain"), "global": dict(variant="global", token="ap"), "maskpool": dict(variant="maskpool"),
          "strip": dict(variant="strip"),
          "slabonly": dict(variant="esa", token="none"), "sva": dict(variant="slabstat", token="ap")}
BACKBONES = {"convnext_tiny": 384, "efficientnet_b0": 384, "vit_small": 384, "bioclip2": 224}


def measure(backbone, size, model_kw, device):
    m = ESFormer(30, backbone, pretrained=backbone == "bioclip2", img_size=size, **model_kw).to(device).eval()
    total = sum(p.numel() for p in m.parameters())
    added = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("backbone."))
    x = torch.randn(1, 3, size, size, device=device)
    mask = torch.zeros(1, size, size, device=device)
    mask[:, size // 4:3 * size // 4, size // 3:2 * size // 3] = 1
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m(x, mask)
        flops = fc.get_total_flops()
        for _ in range(5):
            m(x, mask)
        ts = []
        for _ in range(30):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    del m
    return dict(params_M=total / 1e6, added_params_M=added / 1e6, gflops=flops / 1e9,
                latency_ms=1000 * sorted(ts)[len(ts) // 2])


def main(out, device, backbones):
    dev = torch.device(device)
    rows = []
    for bb in backbones:
        for name, kw in MODELS.items():
            r = measure(bb, BACKBONES[bb], kw, dev)
            rows.append(dict(backbone=bb, model=name, device=device, **r))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cost.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    a = ap.parse_args()
    main(a.out, a.device, a.backbones)
