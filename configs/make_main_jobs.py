"""Generate the main-experiment kernel specs (lean budget: full probe suite on seed 0 only).

  python configs/make_main_jobs.py   -> configs/jobs/main_*.yaml
Model names used throughout: plain | global (ap features, globally fused) | strip (strip pooling) |
slabonly (slab attention, no valuation content) | sva (sweep-valuation attention, proposed).
"""
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent / "jobs"
FULL = ["rotation", "corruptions", "cueconflict", "topodestroy", "crossdomain"]
VARIANTS = {
    "plain": dict(variant="plain"),
    "global": dict(variant="global", token="ap", tag="ap"),
    "strip": dict(variant="strip"),
    "slabonly": dict(variant="esa", token="none", tag="slabonly"),
    "sva": dict(variant="slabstat", token="ap", tag="ap"),
}
BACKBONE_CFG = {
    "convnext_tiny": dict(),
    "vit_small": dict(lr_backbone=5.0e-5),
    "bioclip2": dict(img_size=224, lr_backbone=3.0e-5, epochs=20),
}
BASE = dict(protocol="grouped", epochs=40, batch_size=32, workers=3)


def spec(exp, jobs, pip=("lpips",)):
    return dict(pip=list(pip), kernel_sources=["maskhead"], common=dict(exp=exp, **BASE), jobs=jobs)


def write(name, s):
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print(name, len(s["jobs"]), "jobs")


def main():
    # 1) ConvNeXt-T seed 0, full probes, save SVA/plain checkpoints for explainability
    jobs = []
    for f in range(5):
        for k, v in VARIANTS.items():
            jobs.append(dict(backbone="convnext_tiny", fold=f, seed=0, evals=FULL,
                             save_model=k in ("sva", "plain"), **v))
    write("main_cnx_s0", spec("main", jobs))
    # 2) ConvNeXt-T seeds 1-2, field-photo eval only
    jobs = [dict(backbone="convnext_tiny", fold=f, seed=s, evals=["crossdomain"], **v)
            for s in (1, 2) for f in range(5) for v in VARIANTS.values()]
    write("main_cnx_s12", spec("main", jobs))
    # 3/4) ViT-S and BioCLIP 2, seed 0, full probes
    for bb, name, pip in (("vit_small", "main_vit", ("lpips",)), ("bioclip2", "main_bioclip", ("lpips", "open_clip_torch"))):
        jobs = [dict(backbone=bb, fold=f, seed=0, evals=FULL, **BACKBONE_CFG[bb], **v)
                for f in range(5) for v in VARIANTS.values()]
        if bb == "bioclip2":  # ViT-L/14 runs + full probes may approach Kaggle's 12 h session cap: 2 notebooks
            write(name + "_a", spec("main", jobs[:13], pip))
            write(name + "_b", spec("main", jobs[13:], pip))
        else:
            write(name, spec("main", jobs, pip))
    # 5) ablations of the proposed model (ConvNeXt-T, seed 0, folds 0-2)
    sva = VARIANTS["sva"]
    abl = {
        "euler": dict(variant="esa", token="euler", tag="eu"),
        "hadwiger": dict(variant="esa", token="hadwiger", tag="had"),
        "bladeonly": dict(sva, use_vein=False, tag="ap_bladeonly"),
        "veinonly": dict(sva, use_blade=False, tag="ap_veinonly"),
        "unitw": dict(sva, weights="unit", tag="ap_unitw"),
        "permw": dict(sva, weights="permuted", tag="ap_permw"),
        "globalattn": dict(sva, attn="global", tag="ap_globalattn"),
        "K16T8": dict(sva, K=16, T=8, tag="ap_K16T8"),
        "kappa1": dict(sva, kappa=1.0, tag="ap_kappa1"),
        "kappa16": dict(sva, kappa=16.0, tag="ap_kappa16"),
    }
    jobs = [dict(backbone="convnext_tiny", fold=f, seed=0, evals=["rotation", "topodestroy", "crossdomain"], **v)
            for f in range(3) for v in abl.values()]
    write("main_ablations", spec("ablations", jobs))
    # 6) no-rotation-augmentation arm (ConvNeXt-T, seed 0, 5 folds)
    jobs = [dict(backbone="convnext_tiny", fold=f, seed=0, rot_aug=False, evals=["rotation", "crossdomain"],
                 **{**v, "tag": (v.get("tag", "") + "_norot").lstrip("_")})
            for f in range(5) for k, v in VARIANTS.items() if k != "global"]
    write("main_norot", spec("norot", jobs))
    # 6b) seeds 1 and 2 of the no-rotation arm (one notebook per seed, ~3.5 h each)
    for s in (1, 2):
        jobs = [dict(backbone="convnext_tiny", fold=f, seed=s, rot_aug=False, evals=["rotation", "crossdomain"],
                     **{**v, "tag": (v.get("tag", "") + "_norot").lstrip("_")})
                for f in range(5) for k, v in VARIANTS.items() if k != "global"]
        write(f"main_norot_s{s}", spec("norot", jobs))
    # 7) P1 leakage audit: prior-work reproductions + ours under random-with-duplicates vs grouped
    jobs = []
    for f in range(5):
        for proto in ("random_dups", "grouped"):
            jobs.append(dict(backbone="convnext_tiny", variant="dualbranch", protocol=proto, fold=f, seed=0, evals=[],
                             tag=f"p1_{proto}"))
        for k in ("plain", "sva"):
            jobs.append(dict(backbone="convnext_tiny", protocol="random_dups", fold=f, seed=0, evals=[],
                             **{**VARIANTS[k], "tag": (VARIANTS[k].get("tag", "") + "_p1rand").lstrip("_")}))
    write("main_p1", spec("p1", jobs))


if __name__ == "__main__":
    main()
