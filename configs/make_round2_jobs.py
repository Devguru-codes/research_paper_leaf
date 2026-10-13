"""Round-2 kernel specs (reviewer-driven), one notebook per spec, each <= ~9 GPU-h.

  python configs/make_round2_jobs.py   -> configs/jobs/r2_*.yaml
R1  ConvNeXt-T main, seeds 3-4                           (5 seeds total for the headline claim)
R2  no-rotation-augmentation arm, seeds 3-4             (5 seeds total)
R3  EfficientNet-B0 (second CNN family), seeds 0-2
R4  strong-augmentation (domain-generalisation) baseline: plain / strip / SVA, seeds 0-2
R5  key ablations completed to 5 folds x seeds 0-2       (field transfer only)
R6  ViT-S seeds 1-2                                      (optional, if quota remains)
"""
from pathlib import Path

import yaml

from make_main_jobs import BASE, VARIANTS, spec

OUT = Path(__file__).resolve().parent / "jobs"
FIELD_C = ["crossdomain", "corruptions"]


def write(name, s):
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print(f"{name:22s} {len(s['jobs']):3d} jobs")


def tagged(v, suffix):
    return {**v, "tag": (v.get("tag", "") + "_" + suffix).lstrip("_")}


def main():
    folds = range(5)
    # R1: one notebook per seed (25 runs, ~4.5 h)
    for s in (3, 4):
        write(f"r2_cnx_s{s}", spec("main", [dict(backbone="convnext_tiny", fold=f, seed=s, evals=FIELD_C, **v)
                                            for f in folds for v in VARIANTS.values()]))
    # R2: one notebook per seed (20 runs, ~3.2 h)
    for s in (3, 4):
        write(f"r2_norot_s{s}", spec("norot", [dict(backbone="convnext_tiny", fold=f, seed=s, rot_aug=False,
                                                    evals=["rotation", "crossdomain"], **tagged(v, "norot"))
                                               for f in folds for k, v in VARIANTS.items() if k != "global"]))
    # R3: EfficientNet-B0, one notebook per seed (25 runs)
    for s in (0, 1, 2):
        write(f"r2_effb0_s{s}", spec("main", [dict(backbone="efficientnet_b0", fold=f, seed=s, lr_backbone=2e-4,
                                                   evals=["rotation"] + FIELD_C, **v)
                                              for f in folds for v in VARIANTS.values()]))
    # R4: strong augmentation for plain / strip / SVA, one notebook per seed (15 runs)
    for s in (0, 1, 2):
        write(f"r2_strongaug_s{s}", spec("strongaug", [dict(backbone="convnext_tiny", fold=f, seed=s, strong_aug=True,
                                                            evals=["rotation"] + FIELD_C, **tagged(VARIANTS[k], "strongaug"))
                                                       for f in folds for k in ("plain", "strip", "sva")]))
    # R5: key ablations to 5 folds x seeds 0-2 (field only); seed-0 folds 0-2 already exist
    sva = VARIANTS["sva"]
    abl = {"unitw": dict(sva, weights="unit", tag="ap_unitw"),
           "globalattn": dict(sva, attn="global", tag="ap_globalattn"),
           "K16T8": dict(sva, K=16, T=8, tag="ap_K16T8"),
           "euler": dict(variant="esa", token="euler", tag="eu")}
    todo = [(f, 0) for f in (3, 4)] + [(f, s) for s in (1, 2) for f in folds]
    jobs = [dict(backbone="convnext_tiny", fold=f, seed=s, evals=["crossdomain"], **v) for f, s in todo for v in abl.values()]
    write("r2_abl_a", spec("ablations", jobs[:24]))
    write("r2_abl_b", spec("ablations", jobs[24:]))
    # R6 (optional): ViT-S seeds 1-2
    for s in (1, 2):
        write(f"r2_vit_s{s}", spec("main", [dict(backbone="vit_small", fold=f, seed=s, lr_backbone=5.0e-5,
                                                 evals=FIELD_C, **v) for f in folds for v in VARIANTS.values()]))


if __name__ == "__main__":
    main()
