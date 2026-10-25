"""Round-5 kernel specs (third review), medicinal-leaf part.

  python configs/make_round5_jobs.py   -> configs/jobs/r5_*.yaml
r5_lrsweep      plain and strip pooling (ConvNeXt-T, seed 0, 5 folds) at backbone lr 5e-5 and 2e-4; the
                main runs use 1e-4. Answers "the plain baseline was under-tuned": is the gap there at every lr?
r5_patchpool34  patch pooling seeds 3-4, so every ConvNeXt-T contrast uses 25 matched runs.
r5_leafsnap_s*  LeafSnap30 replication, 5 models x 5 folds per seed (pre-registered).
"""
from pathlib import Path

import yaml

from make_main_jobs import VARIANTS, spec

OUT = Path(__file__).resolve().parent / "jobs"


def write(name, s):
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print(f"{name:16s} {len(s['jobs']):3d} jobs")


def main():
    folds = range(5)
    write("r5_lrsweep", spec("lrsweep", [dict(backbone="convnext_tiny", fold=f, seed=0, lr_backbone=lr, variant=v,
                                              evals=["crossdomain"], tag=f"lr{lr:g}")
                                         for lr in (5e-5, 2e-4) for v in ("plain", "strip") for f in folds]))
    write("r5_patchpool34", spec("main", [dict(backbone="convnext_tiny", fold=f, seed=s, variant="patchpool",
                                                evals=["crossdomain", "corruptions"]) for s in (3, 4) for f in folds]))
    # LeafSnap30 replication (paper/preregistration_leafsnap.md): only the LeafSnap dataset is attached
    models = [VARIANTS["plain"], dict(variant="patchpool"), dict(variant="maskpool"), VARIANTS["strip"], VARIANTS["sva"]]
    for s in (0, 1):
        ls = spec("leafsnap", [dict(backbone="convnext_tiny", fold=f, seed=s, epochs=15, evals=["crossdomain"], **m)
                               for f in folds for m in models])
        ls.update(data="medleaf-esa-leafsnap", kernel_sources=[])
        write(f"r5_leafsnap_s{s}", ls)


if __name__ == "__main__":
    main()
