"""How well does the learned venation field reproduce its ridge-filter supervision? (reviewer request)

  python -m src.vein_check results/kaggle/main_cnx_s0
For every saved checkpoint (ConvNeXt-T plain and SVA, seed 0, 5 folds) the vein head is run on the
fold's test images and compared with the pseudo-label inside the blade on the 96x96 grid: Pearson r,
and Dice / precision / recall after thresholding both maps at 0.5. This measures agreement with the
pseudo-label only; it is not a validation against manually traced veins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.splits import make_split
from src.evaluate import _prep, load_items
from src.paths import cache_root
from src.train import DEFAULTS, build_model


@torch.no_grad()
def check(run: Path, df: pd.DataFrame, device) -> dict:
    cfg = {**DEFAULTS, **json.loads((run / "metrics.json").read_text())["cfg"], "pretrained": False}
    model = build_model(cfg).to(device).eval()
    sd = torch.load(run / "model.pt", map_location=device)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    test_idx = make_split(df, cfg["fold"], cfg["protocol"])["test"]
    items = load_items(df, test_idx)
    G = cfg["grid"]
    rs, inter, sp, st = [], 0.0, 0.0, 0.0
    for i in range(0, len(items["img"]), 16):
        xs, ms = zip(*[_prep(a, b, cfg["img_size"]) for a, b in zip(items["img"][i:i + 16], items["mask"][i:i + 16])])
        x, m = torch.stack(xs).to(device), torch.stack(ms).to(device)
        _, aux = model(x, m, return_aux=True)
        pred = torch.sigmoid(aux["vein_logit"].float())
        if pred.shape[-1] != G:
            pred = F.interpolate(pred[:, None], size=(G, G), mode="bilinear", align_corners=False)[:, 0]
        blade = aux["blade"].bool()
        for j, v in enumerate(items["vein"][i:i + 16]):
            _, mv = _prep(items["img"][i + j], (v > 0.5).astype(np.uint8), cfg["img_size"])  # binarised target
            tv = torch.tensor(np.asarray(v), dtype=torch.float32)[None, None]
            tgt = F.interpolate(F.interpolate(tv, size=(cfg["img_size"],) * 2, mode="bilinear"), size=(G, G),
                                mode="area")[0, 0].to(device)
            b = blade[j]
            p, t = pred[j][b], tgt[b]
            if p.numel() > 10 and p.std() > 0 and t.std() > 0:
                rs.append(float(torch.corrcoef(torch.stack([p, t]))[0, 1]))
            pb, tb = p > 0.5, t > 0.5
            inter += float((pb & tb).sum())
            sp += float(pb.sum())
            st += float(tb.sum())
    return dict(variant=cfg["variant"], token=cfg["token"], fold=cfg["fold"], pearson_r=float(np.mean(rs)),
                dice=2 * inter / max(sp + st, 1), precision=inter / max(sp, 1), recall=inter / max(st, 1))


def main(root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cache_root() / "index.csv")
    rows = [check(p.parent, df, device) for p in sorted(root.rglob("model.pt"))]
    res = pd.DataFrame(rows)
    res.to_csv("results/vein_check.csv", index=False)
    print(res.round(3).to_string())
    print(res.groupby("variant")[["pearson_r", "dice", "precision", "recall"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle/main_cnx_s0"))
