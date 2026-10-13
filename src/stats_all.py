"""All paired comparisons reported in the paper -> results/stats_all.json (+ printed summary).

  python -m src.stats_all results/kaggle
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.make_tables import collect
from src.stats import holm, paired, topo_regression

BASELINES = ["plain", "global", "maskpool", "strip", "slabonly"]


def main(root: Path):
    df = collect(root)
    out = dict(main={}, norot={}, topo={})
    for bb in ("convnext_tiny", "efficientnet_b0", "vit_small", "bioclip2"):
        out["main"][bb] = [paired(df, "sva", b, bb, exp="main") for b in BASELINES]
        out["topo"][bb] = topo_regression(df, bb, exp="main")
    out["norot"]["convnext_tiny"] = [paired(df, "sva", b, "convnext_tiny", exp="norot", rot_aug=False)
                                     for b in ["plain", "strip", "slabonly"]]
    # Holm correction over all SVA-vs-control field-transfer tests (corrected resampled t-test p-values)
    fam = [(arm, bb, i, k) for arm in ("main", "norot") for bb, comps in out[arm].items() for i, p in enumerate(comps)
           for k in ("crossdomain.masked.acc20", "crossdomain.raw.acc20") if k in p]
    adj = holm([out[a][bb][i][k]["p_corr_t"] for a, bb, i, k in fam])
    for (a, bb, i, k), q in zip(fam, adj):
        out[a][bb][i][k]["p_holm"] = q
    Path("results").mkdir(exist_ok=True)
    Path("results/stats_all.json").write_text(json.dumps(out, indent=1))
    keys = ["test.acc", "rotation.min_acc", "corruptions.mean_corrupted_acc", "cueconflict.all.shape_bias",
            "crossdomain.masked.acc20", "crossdomain.raw.acc20"]
    for arm in ("main", "norot"):
        for bb, comps in out[arm].items():
            for p in comps:
                cells = [f"{k.split('.')[-1] if 'cross' not in k else k.split('.')[1]} {100 * p[k]['mean_diff']:+.1f} "
                         f"[{100 * p[k]['ci95'][0]:+.1f},{100 * p[k]['ci95'][1]:+.1f}] ({p[k]['wins']}/{p[k]['n']}, "
                         f"pW={p[k]['p']:.2g}, pT={p[k]['p_corr_t']:.2g}"
                         + (f", pHolm={p[k]['p_holm']:.2g}" if "p_holm" in p[k] else "") + ")" for k in keys if k in p]
                print(f"{arm:5s} {bb:13s} sva-{p['b']:8s} " + "; ".join(cells))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle"))
