"""Evaluation suites run on a trained model and a test fold.

suites: rotation | corruptions | cueconflict | topodestroy
Each writes eval_<suite>.json (summary) and, where useful, eval_<suite>_samples.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.corruptions import CORRUPTIONS, SEVERITIES
from src.data.cueconflict import cue_conflict, make_pairs
from src.data.dataset import finalize, geometric, load_item, to_tensors
from src.data import topodestroy as td
from src.metrics import classification_metrics, softmax


def _prep(img: np.ndarray, mask01: np.ndarray, size: int, angle: float = 0.0):
    x, m, v = to_tensors(img, (mask01 > 0).astype(np.uint8) * 255, np.zeros(mask01.shape, np.uint8))
    if angle:
        x, m, v = geometric(x, m, v, angle=angle, scale=1.0)
    x, m, _ = finalize(x, m, v, size)
    return x, m


@torch.no_grad()
def predict_arrays(model, imgs, masks, size, device, bs=32, angle=0.0) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        xs, ms = zip(*[_prep(a, b, size, angle) for a, b in zip(imgs[i:i + bs], masks[i:i + bs])])
        x, m = torch.stack(xs).to(device), torch.stack(ms).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg, _ = model(x, m)
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


def suite_rotation(model, items, cfg, device, out: Path):
    imgs, masks, y = items["img"], items["mask"], items["label"]
    res, preds = {}, {}
    for ang in range(0, 360, 30):
        lg = predict_arrays(model, imgs, masks, cfg["img_size"], device, angle=float(ang))
        res[str(ang)] = classification_metrics(lg, y)
        preds[ang] = lg.argmax(1)
    P = np.stack(list(preds.values()), 1)
    summ = dict(per_angle=res, mean_acc=float(np.mean([r["acc"] for r in res.values()])),
                min_acc=float(np.min([r["acc"] for r in res.values()])),
                consistency=float((P == P[:, :1]).all(1).mean()))
    (out / "eval_rotation.json").write_text(json.dumps(summ, indent=1))
    return summ


def suite_corruptions(model, items, cfg, device, out: Path, seed: int):
    imgs, masks, y = items["img"], items["mask"], items["label"]
    clean = classification_metrics(predict_arrays(model, imgs, masks, cfg["img_size"], device), y)["acc"]
    res = dict(clean=clean)
    for name, fn in CORRUPTIONS.items():
        for s in SEVERITIES:
            rng = np.random.default_rng(seed * 1000 + s)
            ci, cm = zip(*[fn(a, b, s, rng) for a, b in zip(imgs, masks)])
            res[f"{name}/{s}"] = classification_metrics(
                predict_arrays(model, list(ci), list(cm), cfg["img_size"], device), y)["acc"]
    accs = [v for k, v in res.items() if "/" in k]
    res["mean_corrupted_acc"] = float(np.mean(accs))
    res["relative_robustness"] = float(np.mean(accs) / max(clean, 1e-9))
    (out / "eval_corruptions.json").write_text(json.dumps(res, indent=1))
    return res


def suite_cueconflict(model, items, cfg, device, out: Path, seed: int):
    imgs, masks, y = items["img"], items["mask"], items["label"]
    pairs = make_pairs(y, np.random.default_rng(seed))
    stim, sm, info = [], [], []
    for i, j in pairs:
        o, m, inf = cue_conflict(imgs[i], masks[i], imgs[j], masks[j])
        stim.append(o)
        sm.append(m)
        info.append(inf)
    pred = predict_arrays(model, stim, sm, cfg["img_size"], device).argmax(1)
    ya, yb = y[[i for i, _ in pairs]], y[[j for _, j in pairs]]
    df = pd.DataFrame(dict(a=[i for i, _ in pairs], b=[j for _, j in pairs], label_shape=ya, label_texture=yb,
                           pred=pred, **{k: [d[k] for d in info] for k in info[0]}))
    df.to_csv(out / "eval_cueconflict_samples.csv", index=False)

    def bias(d):
        s, t = (d.pred == d.label_shape).sum(), (d.pred == d.label_texture).sum()
        return dict(shape_frac=float(s / len(d)), texture_frac=float(t / len(d)),
                    shape_bias=float(s / max(s + t, 1)), n=int(len(d)))

    res = dict(all=bias(df), moderate_stretch=bias(df[df.stretch <= 2.0]))
    (out / "eval_cueconflict.json").write_text(json.dumps(res, indent=1))
    return res


def suite_topodestroy(model, items, cfg, device, out: Path, seed: int):
    imgs, masks, veins, y = items["img"], items["mask"], items["vein"], items["label"]
    base = softmax(predict_arrays(model, imgs, masks, cfg["img_size"], device))
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception:
        pass
    rows = []
    for edit, (kind, changes) in td.EDITS.items():
        rng = np.random.default_rng(seed + hash(edit) % 1000)
        fn = getattr(td, edit)
        ei, em = [], []
        for a, m, v in zip(imgs, masks, veins):
            o, mm = fn(a, m, rng) if kind == "blade" else fn(a, m, v, rng)
            ei.append(o)
            em.append(mm)
        p = softmax(predict_arrays(model, ei, em, cfg["img_size"], device))
        for k in range(len(imgs)):
            d = td.visual_distance(imgs[k], ei[k], masks[k])
            if lp is not None:
                with torch.no_grad():
                    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa
                    d["lpips"] = float(lp(t(imgs[k]), t(ei[k])).item())
            kl = float((base[k] * (np.log(base[k] + 1e-12) - np.log(p[k] + 1e-12))).sum())
            rows.append(dict(edit=edit, kind=kind, chi_changing=changes, idx=k, label=int(y[k]),
                             dchi_blade=td.chi(em[k]) - td.chi(masks[k]) if kind == "blade" else 0,
                             p_true_before=float(base[k, y[k]]), p_true_after=float(p[k, y[k]]),
                             flip=bool(base[k].argmax() != p[k].argmax()), kl=kl, **d))
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_topodestroy_samples.csv", index=False)
    res = {}
    for edit, g in df.groupby("edit"):
        res[edit] = dict(flip_rate=float(g.flip.mean()), mean_kl=float(g.kl.mean()),
                         mean_dp_true=float((g.p_true_before - g.p_true_after).mean()),
                         mean_rmse=float(g.pixel_rmse.mean()),
                         mean_lpips=float(g.lpips.mean()) if "lpips" in g else None,
                         mean_dchi=float(g.dchi_blade.mean()))
    (out / "eval_topodestroy.json").write_text(json.dumps(res, indent=1))
    return res


def suite_crossdomain(model, cfg, device, out: Path):
    from src.crossdomain import find_built, load_items as load_dimpsar
    built = find_built()
    if built is None:
        print("crossdomain: no DIMPSAR build found, skipping", flush=True)
        return None
    items = load_dimpsar(built)
    y = items["label"]
    res = dict(n=int(len(y)))
    for mode in ("masked", "raw"):
        lg = predict_arrays(model, items[mode], items["mask"], cfg["img_size"], device)
        np.savez_compressed(out / f"preds_dimpsar_{mode}.npz", logits=lg, label=y)
        restricted = np.full_like(lg, -1e9)
        overlap = np.unique(y)  # species present in the field set (DIMPSAR: the 20 shared species)
        restricted[:, overlap] = lg[:, overlap]
        m30, m20 = classification_metrics(lg, y), classification_metrics(restricted, y)
        res[mode] = dict(acc30=m30["acc"], bal_acc30=m30["bal_acc"], acc20=m20["acc"], bal_acc20=m20["bal_acc"],
                         macro_f1_20=m20["macro_f1"], ece30=m30["ece"])
    (out / "eval_crossdomain.json").write_text(json.dumps(res, indent=1))
    return res


def load_items(df: pd.DataFrame, idx) -> dict:
    imgs, masks, veins, labels = [], [], [], []
    for i in idx:
        r = df.loc[i]
        a, m, v = load_item(r.stem)
        imgs.append(a)
        masks.append((m > 127).astype(np.uint8))
        veins.append(v.astype(np.float32) / 255.0)
        labels.append(int(r.label))
    return dict(img=imgs, mask=masks, vein=veins, label=np.array(labels))


def run_evals(model, df, test_idx, cfg, out: Path, device):
    items = load_items(df, test_idx)
    seed = 1234 + cfg["fold"]
    for s in cfg["evals"]:
        if s == "rotation":
            r = suite_rotation(model, items, cfg, device, out)
        elif s == "corruptions":
            r = suite_corruptions(model, items, cfg, device, out, seed)
        elif s == "cueconflict":
            r = suite_cueconflict(model, items, cfg, device, out, seed)
        elif s == "topodestroy":
            r = suite_topodestroy(model, items, cfg, device, out, seed)
        elif s == "crossdomain":
            r = suite_crossdomain(model, cfg, device, out)
        else:
            raise ValueError(s)
        print(f"EVAL {s}", json.dumps(r)[:400], flush=True)
