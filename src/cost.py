"""Parameters, GFLOPs and latency of every backbone x model (reviewer request).

  python -m src.cost --out results/cost.csv
  python -m src.cost --masker              (leaf masker, CPU and GPU -> results/cost_masker.csv)
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


def profile(m, inputs, device):
    """GFLOPs of one forward pass and median batch-1 latency (ms) over 30 passes after 5 warm-ups."""
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            m(*inputs)
        for _ in range(5):
            m(*inputs)
        ts = []
        for _ in range(30):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    return fc.get_total_flops() / 1e9, 1000 * sorted(ts)[len(ts) // 2]


def measure(backbone, size, model_kw, device):
    m = ESFormer(30, backbone, pretrained=backbone == "bioclip2", img_size=size, **model_kw).to(device).eval()
    total = sum(p.numel() for p in m.parameters())
    added = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("backbone."))
    x = torch.randn(1, 3, size, size, device=device)
    mask = torch.zeros(1, size, size, device=device)
    mask[:, size // 4:3 * size // 4, size // 3:2 * size // 3] = 1
    gflops, ms = profile(m, (x, mask), device)
    del m
    return dict(params_M=total / 1e6, added_params_M=added / 1e6, gflops=gflops, latency_ms=ms)


def measure_masker(out="results/cost_masker.csv", size=448):
    """The leaf masker that crops (and, for mask-based models, masks) every field photograph."""
    from src.models.mask_head import LeafMasker
    rows = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        dev = torch.device(device)
        m = LeafMasker(pretrained=False).to(dev).eval()
        gflops, ms = profile(m, (torch.randn(1, 3, size, size, device=dev),), dev)
        rows.append(dict(device=device, params_M=sum(p.numel() for p in m.parameters()) / 1e6, gflops=gflops,
                         latency_ms=ms))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

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
    ap.add_argument("--masker", action="store_true", help="measure the leaf masker -> results/cost_masker.csv")
    a = ap.parse_args()
    measure_masker() if a.masker else main(a.out, a.device, a.backbones)
