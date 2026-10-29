"""Round-6 kernel specs (fourth review), LeafSnap30 only.

  python configs/make_round6_jobs.py   -> configs/jobs/r6_*.yaml
r6_leafsnap_s2..s4  three more seeds of the five LeafSnap models (exp "leafsnap"), reported as an extension
                    of the pre-registered seeds 0-1 (the pre-registered table keeps seeds 0-1 only).
r6_leafsnap_checks  exp "leafsnap_checks": BioCLIP 2 plain and strip pooling (does the foundation model win
                    here too?) and ConvNeXt-T plain and strip at 40 epochs (does the 15-epoch choice matter?).
"""
from pathlib import Path

import yaml

from make_main_jobs import BACKBONE_CFG, VARIANTS, spec

OUT = Path(__file__).resolve().parent / "jobs"
MODELS = [VARIANTS["plain"], dict(variant="patchpool"), dict(variant="maskpool"), VARIANTS["strip"], VARIANTS["sva"]]


def write(name, s):
    s.update(data="medleaf-esa-leafsnap", kernel_sources=[])
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print(f"{name:20s} {len(s['jobs']):3d} jobs")


def main():
    folds = range(5)
    for s in (2, 3, 4):
        write(f"r6_leafsnap_s{s}", spec("leafsnap", [dict(backbone="convnext_tiny", fold=f, seed=s, epochs=15,
                                                          evals=["crossdomain"], **m) for f in folds for m in MODELS]))
    # BioCLIP 2: 20 epochs on the medicinal set -> 7 on LeafSnap's 2.9x larger training set (same step rule)
    bio = [dict(backbone="bioclip2", fold=f, seed=0, evals=["crossdomain"], **{**BACKBONE_CFG["bioclip2"], "epochs": 7},
                **VARIANTS[k]) for f in folds for k in ("plain", "strip")]
    ep40 = [dict(backbone="convnext_tiny", fold=f, seed=0, epochs=40, evals=["crossdomain"], tag="ep40", **VARIANTS[k])
            for f in folds for k in ("plain", "strip")]
    write("r6_leafsnap_checks", spec("leafsnap_checks", bio + ep40, pip=("lpips", "open_clip_torch")))


if __name__ == "__main__":
    main()
