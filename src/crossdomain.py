"""Cross-domain test set (P7): DIMPSAR field photographs (Pushpa & Shobha Rani, Data in Brief 2023,
CC BY 4.0; Kaggle mirror aryashah2k/indian-medicinal-leaves-dataset), 20 species overlapping ours.

Vernacular folder names were mapped to species and checked visually (note: Kannada 'Arali' is
Ficus religiosa (peepal), NOT Nerium oleander).

  python -m src.crossdomain build --masker results/maskhead/maskhead.pt --out results/dimpsar
builds <out>/{img,mask}/ crops (448px, leaf-centred via our own masker) + dimpsar_index.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from src.data.dataset import MEAN, STD
from src.data.masks import crop_square
from src.data.taxonomy import CLASSES

DIMPSAR_TO_OURS = {
    "Arali": "Ficus Religiosa (Peepal Tree)",
    "Betel": "Piper Betle (Betel)",
    "Curry": "Murraya Koenigii (Curry)",
    "Doddpathre": "Plectranthus Amboinicus (Mexican Mint)",
    "Drumstick": "Moringa Oleifera (Drumstick)",
    "Gasagase": "Muntingia Calabura (Jamaica Cherry-Gasagase)",
    "Guava": "Psidium Guajava (Guava)",
    "Hibiscus": "Hibiscus Rosa-sinensis",
    "Honge": "Pongamia Pinnata (Indian Beech)",
    "Jackfruit": "Artocarpus Heterophyllus (Jackfruit)",
    "Jasmine": "Jasminum (Jasmine)",
    "Lemon": "Citrus Limon (Lemon)",
    "Malabar_Spinach": "Basella Alba (Basale)",
    "Mango": "Mangifera Indica (Mango)",
    "Mint": "Mentha (Mint)",
    "Neem": "Azadirachta Indica (Neem)",
    "Nerale": "Syzygium Cumini (Jamun)",
    "Parijatha": "Nyctanthes Arbor-tristis (Parijata)",
    "Pomoegranate": "Punica Granatum (Pomegranate)",
    "Tulsi": "Ocimum Tenuiflorum (Tulsi)",
}
OVERLAP = sorted(CLASSES.index(v) for v in DIMPSAR_TO_OURS.values())


def find_root() -> Path:
    c = glob.glob("/kaggle/input/**/Medicinal Leaf dataset", recursive=True) or \
        glob.glob(os.environ.get("DIMPSAR_ROOT", "__none__") + "/**/Medicinal Leaf dataset", recursive=True)
    return Path(c[0])


@torch.no_grad()
def predict_mask(masker, rgb: np.ndarray, device, size: int = 448) -> np.ndarray:
    H, W = rgb.shape[:2]
    s = size / max(H, W)
    small = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    pad = np.full((size, size, 3), 255, np.uint8)
    pad[:small.shape[0], :small.shape[1]] = small
    x = ((torch.from_numpy(pad).permute(2, 0, 1).float() / 255 - MEAN) / STD)[None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.sigmoid(masker(x).float())[0].cpu().numpy()
    p = p[:small.shape[0], :small.shape[1]]
    m = (cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:  # keep the largest component (one leaf per photo)
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m


def build(masker_path: str | None, out: str, max_side: int = 1600, root: Path | None = None,
          mapping: dict | None = None, mask_fn=None):
    """Field test set. Defaults: DIMPSAR folders, our masker. A label-free `mask_fn(rgb)` replaces the
    masker for plain backgrounds (LeafSnap field photos); `mapping` is folder -> class index."""
    if mask_fn is None:
        from src.models.mask_head import LeafMasker
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        masker = LeafMasker(pretrained=False).to(dev).eval()
        masker.load_state_dict(torch.load(masker_path, map_location=dev))
        mask_fn = lambda rgb: predict_mask(masker, rgb, dev)  # noqa: E731
    root, out = root or find_root(), Path(out)
    mapping = mapping or {folder: CLASSES.index(cls) for folder, cls in DIMPSAR_TO_OURS.items()}
    for d in ("raw", "masked", "mask"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, label in mapping.items():
        files = sorted(f for f in (root / folder).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for k, f in enumerate(files):
            try:
                rgb = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            except Exception:
                continue
            s = min(1.0, max_side / max(rgb.shape[:2]))
            if s < 1:
                rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            m = mask_fn(rgb)
            frac = float(m.mean())
            ok = 0.005 < frac < 0.95
            stem = f"{label:02d}_{folder}_{k:04d}"
            if ok:
                raw_c, m_c, _ = crop_square(rgb, m)
                masked = raw_c.copy()
                masked[m_c == 0] = 255
                cv2.imwrite(str(out / "raw" / f"{stem}.jpg"), cv2.cvtColor(raw_c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out / "masked" / f"{stem}.jpg"), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(out / "mask" / f"{stem}.png"), m_c * 255)
            rows.append(dict(stem=stem, src=str(f.relative_to(root)), folder=folder, label=label,
                             mask_frac=frac, usable=ok))
    df = pd.DataFrame(rows)
    df.to_csv(out / "dimpsar_index.csv", index=False)
    summary = dict(n_files=len(df), n_usable=int(df.usable.sum()),
                   per_class=df.groupby("folder").usable.agg(["sum", "count"]).to_dict())
    (out / "dimpsar_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)[:500])


def find_built() -> Path | None:
    c = glob.glob("/kaggle/input/**/dimpsar_index.csv", recursive=True) or \
        glob.glob(os.environ.get("DIMPSAR_BUILT", "__none__") + "/dimpsar_index.csv")
    return Path(c[0]).parent if c else None


def load_items(built: Path):
    df = pd.read_csv(built / "dimpsar_index.csv")
    df = df[df.usable]
    items = dict(raw=[], masked=[], mask=[], label=df.label.to_numpy(), folder=df.folder.to_numpy())
    for s in df.stem:
        items["raw"].append(cv2.cvtColor(cv2.imread(str(built / "raw" / f"{s}.jpg")), cv2.COLOR_BGR2RGB))
        items["masked"].append(cv2.cvtColor(cv2.imread(str(built / "masked" / f"{s}.jpg")), cv2.COLOR_BGR2RGB))
        items["mask"].append((cv2.imread(str(built / "mask" / f"{s}.png"), 0) > 127).astype(np.uint8))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--masker", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.masker, a.out)
