"""Train + evaluate one (backbone, variant, fold, seed) run.

  python -m src.train --config configs/pilot.yaml --variant esa --fold 0 --seed 0

Outputs to results/<exp>/<run_name>/fold{f}_seed{s}/:
  metrics.json, preds_{test,cal}.npz, eval_*.json (from the evaluation suites), model.pt (opt.)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset
from src.data.gpu_aug import gpu_augment
from src.data.splits import leakage_report, make_split
from src.data.taxonomy import CLASSES
from src.metrics import classification_metrics
from src.models.esformer import ESFormer
from src.paths import cache_root, results_root

DEFAULTS = dict(
    exp="debug", backbone="convnext_tiny", variant="esa", protocol="grouped", fold=0, seed=0,
    img_size=384, grid=96, epochs=40, batch_size=32, lr_backbone=1e-4, lr_new=1e-3, weight_decay=0.05,
    warmup_epochs=2, label_smoothing=0.1, vein_weight=0.2, amp=True, workers=4, pretrained=True,
    d=192, K=32, T=16, R=2.0, kappa=4.0, nbins=256, heads=4, depth=2, weights="learned", attn="slab",
    use_blade=True, use_vein=True, save_model=False, tag="", evals=["rotation"],
    max_per_class=0, classes=0, grad_accum=1, gpu_aug=True, token=None, rot_aug=True,
)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_name(cfg: dict) -> str:
    n = f"{cfg['backbone']}_{cfg['variant']}"
    return n + (f"_{cfg['tag']}" if cfg["tag"] else "")


def build_model(cfg: dict):
    if cfg["variant"] == "dualbranch":
        from src.models.baselines import DualBranch
        return DualBranch(len(CLASSES), cfg["pretrained"])
    return ESFormer(len(CLASSES), cfg["backbone"], cfg["variant"], cfg["pretrained"], cfg["img_size"],
                    cfg["grid"], cfg["d"], cfg["K"], cfg["T"], cfg["R"], cfg["kappa"], cfg["nbins"],
                    cfg["heads"], cfg["depth"], cfg["weights"], cfg["attn"], cfg["use_blade"], cfg["use_vein"],
                    token=cfg["token"])


def vein_loss(vein_logit: torch.Tensor, vein_target: torch.Tensor, mask: torch.Tensor, grid: int):
    tgt = F.interpolate(vein_target[:, None], size=(grid, grid), mode="area")[:, 0]
    blade = (F.interpolate(mask[:, None], size=(grid, grid), mode="area")[:, 0] > 0.5).float()
    l = F.binary_cross_entropy_with_logits(vein_logit.float(), tgt, reduction="none")
    return (l * blade).sum() / blade.sum().clamp_min(1.0)


def forward(model, b, device):
    x, m = b["img"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
    if "skel" in b:
        return model(x, m, b["skel"].to(device, non_blocking=True))
    return model(x, m)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    L, Y, I = [], [], []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits, _ = forward(model, b, device)
        L.append(logits.float().cpu())
        Y.append(b["label"])
        I.append(b["index"])
    return torch.cat(L).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def subset(df: pd.DataFrame, idx: np.ndarray, cfg: dict) -> np.ndarray:
    d = df.loc[idx]
    if cfg["classes"]:
        d = d[d.label < cfg["classes"]]
    if cfg["max_per_class"]:
        d = d.groupby("label").head(cfg["max_per_class"])
    return d.index.to_numpy()


def train_one(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cache_root() / "index.csv")
    split = make_split(df, cfg["fold"], cfg["protocol"])
    leak = leakage_report(df, split)
    split = {k: subset(df, v, cfg) for k, v in split.items()}
    out = results_root() / cfg["exp"] / run_name(cfg) / f"fold{cfg['fold']}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    gpu_aug = cfg["gpu_aug"] and cfg["variant"] != "dualbranch" and torch.cuda.is_available()

    dl = lambda idx, train: DataLoader(  # noqa: E731
        LeafDataset(df, idx, cfg["img_size"], train, skeleton=cfg["variant"] == "dualbranch",
                    raw_uint8=train and gpu_aug), batch_size=cfg["batch_size"], shuffle=train,
        drop_last=train and len(idx) > cfg["batch_size"], num_workers=cfg["workers"] if train else 0, pin_memory=True,
        persistent_workers=train and cfg["workers"] > 0)
    tr_loader, cal_loader, te_loader = dl(split["train"], True), dl(split["cal"], False), dl(split["test"], False)

    model = build_model(cfg).to(device)
    bbmod = model.backbone if hasattr(model, "backbone") else nn.ModuleList([model.rgb, model.skel])
    bb = [p for p in bbmod.parameters() if p.requires_grad]
    bb_ids = {id(p) for p in bb}
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in bb_ids]
    opt = torch.optim.AdamW([dict(params=bb, lr=cfg["lr_backbone"]), dict(params=new, lr=cfg["lr_new"])],
                            weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(tr_loader) // cfg["grad_accum"])
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    log, t0, step = [], time.time(), 0
    for ep in range(cfg["epochs"]):
        model.train()
        tl, tv, n = 0.0, 0.0, 0
        for it, b in enumerate(tr_loader):
            if gpu_aug:
                x8, m8, v8 = (b[k].to(device, non_blocking=True) for k in ("img", "mask", "vein"))
                b["img"], b["mask"], b["vein"] = gpu_augment(x8, m8, v8, cfg["img_size"], rot=cfg["rot_aug"])
            m, v, y = (b[k].to(device, non_blocking=True) for k in ("mask", "vein", "label"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, vl = forward(model, b, device)
                lc = F.cross_entropy(logits.float(), y, label_smoothing=cfg["label_smoothing"])
                lv = vein_loss(vl, v, m, cfg["grid"]) if vl is not None else torch.zeros((), device=device)
                loss = (lc + cfg["vein_weight"] * lv) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (it + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            tl += lc.item() * len(y)
            tv += lv.item() * len(y)
            n += len(y)
        rec = dict(epoch=ep, train_ce=tl / n, train_vein=tv / n, time=time.time() - t0)
        if ep == cfg["epochs"] - 1 or ep % 5 == 4:
            lg, yy, _ = predict(model, cal_loader, device, use_amp)
            rec["cal_acc"] = float((lg.argmax(1) == yy).mean())
        log.append(rec)
        print(json.dumps(rec), flush=True)
    train_time = time.time() - t0

    res = dict(cfg=cfg, leakage=leak, n_train=len(split["train"]), n_cal=len(split["cal"]),
               n_test=len(split["test"]), train_time_s=train_time, log=log,
               params_total=sum(p.numel() for p in model.parameters()),
               params_fusion=sum(p.numel() for p in model.fusion.parameters()) if getattr(model, "fusion", None) else 0)
    for name, loader in (("cal", cal_loader), ("test", te_loader)):
        t = time.time()
        lg, yy, ii = predict(model, loader, device, use_amp)
        res[f"{name}_time_s"] = time.time() - t
        np.savez_compressed(out / f"preds_{name}.npz", logits=lg, label=yy, index=ii)
        res[name] = classification_metrics(lg, yy)
    print("TEST", json.dumps(res["test"]), flush=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))

    if cfg["evals"]:
        from src.evaluate import run_evals
        run_evals(model, df, split["test"], cfg, out, device)
    if cfg["save_model"]:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   out / "model.pt")
    return res


def parse() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml-parsed)")
    for k, v in DEFAULTS.items():
        if not isinstance(v, (list, bool)):
            ap.add_argument(f"--{k}", type=type(v), default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(yaml.safe_load(Path(a.config).read_text()) or {})
    for k in DEFAULTS:
        if getattr(a, k, None) is not None:
            cfg[k] = getattr(a, k)
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)  # launcher passes JSON; YAML 1.1 would read "5e-05" as a string
        except json.JSONDecodeError:
            cfg[k] = yaml.safe_load(v)
    return cfg


if __name__ == "__main__":
    train_one(parse())
