"""DeepHerb-style reproduction (Roopashree & Anitha, IEEE Access 2021): frozen ImageNet Xception
features + RBF-SVM, under both protocols (random_dups as in prior work vs. our grouped protocol).

  python -m src.baseline_deepherb --out results/p1/deepherb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import LeafDataset
from src.data.splits import make_split
from src.metrics import classification_metrics
from src.paths import cache_root, results_root


@torch.no_grad()
def features(model, df, idx, dev, size=299):
    dl = torch.utils.data.DataLoader(LeafDataset(df, idx, size, False), batch_size=64, num_workers=4)
    F = []
    for b in dl:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            F.append(model(b["img"].to(dev)).float().cpu())
    return torch.cat(F).numpy()


def main():
    import timm
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_root() / "p1" / "deepherb"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("legacy_xception.tf_in1k", pretrained=True, num_classes=0).to(dev).eval()
    df = pd.read_csv(cache_root() / "index.csv")
    feats = features(model, df, df.index.to_numpy(), dev)  # one pass over all images (eval transform only)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for protocol in ("random_dups", "grouped"):
        for fold in range(5):
            sp = make_split(df, fold, protocol)
            tr = np.concatenate([sp["train"], sp["cal"]])
            sc = StandardScaler().fit(feats[tr])
            best = None
            for C in (1, 10, 100):  # select C on the calibration split
                clf = SVC(C=C, kernel="rbf", probability=False).fit(sc.transform(feats[sp["train"]]), df.label[sp["train"]])
                acc = (clf.predict(sc.transform(feats[sp["cal"]])) == df.label[sp["cal"]]).mean()
                if best is None or acc > best[0]:
                    best = (acc, C)
            clf = SVC(C=best[1], kernel="rbf").fit(sc.transform(feats[tr]), df.label[tr])
            dec = clf.decision_function(sc.transform(feats[sp["test"]]))
            m = classification_metrics(dec, df.label[sp["test"]].to_numpy())
            res[f"{protocol}/fold{fold}"] = dict(acc=m["acc"], bal_acc=m["bal_acc"], macro_f1=m["macro_f1"], C=best[1])
            print(protocol, fold, res[f"{protocol}/fold{fold}"], flush=True)
    for protocol in ("random_dups", "grouped"):
        accs = [v["acc"] for k, v in res.items() if k.startswith(protocol)]
        res[f"{protocol}/mean_acc"] = float(np.mean(accs))
        res[f"{protocol}/std_acc"] = float(np.std(accs))
    (out / "deepherb.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
