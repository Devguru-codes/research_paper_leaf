"""Round-4 kernel specs (second review): mechanism of the mask-pooling gain and generality.

  python configs/make_round4_jobs.py   -> configs/jobs/r4_*.yaml
r4_patchpool  ConvNeXt-T `patchpool`: the mask-pooling branch averaged over the whole image (no mask),
              seeds 0-2. Separates "mid-level features in an extra branch" from "knowing where the leaf is".
r4_mp_strong  mask pooling with strong augmentation, seeds 0-2 (pairs with r2_strongaug_s*)
r4_mp_norot   mask pooling without rotation augmentation, seeds 0-2 (pairs with main_norot*)
r4_dinov2     DINOv2 ViT-S/14 (general-purpose self-supervised backbone): plain, mask pooling, strip
              pooling, seed 0, 5 folds. Is "a foundation model closes the gap" specific to BioCLIP 2?
"""
from pathlib import Path

import yaml

from make_main_jobs import spec

OUT = Path(__file__).resolve().parent / "jobs"
FIELD_C = ["crossdomain", "corruptions"]


def write(name, s):
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print(f"{name:16s} {len(s['jobs']):3d} jobs")


def main():
    folds, seeds = range(5), range(3)
    write("r4_patchpool", spec("main", [dict(backbone="convnext_tiny", fold=f, seed=s, evals=FIELD_C,
                                             variant="patchpool") for s in seeds for f in folds]))
    write("r4_mp_strong", spec("strongaug", [dict(backbone="convnext_tiny", fold=f, seed=s, strong_aug=True,
                                                  evals=FIELD_C, variant="maskpool", tag="strongaug")
                                             for s in seeds for f in folds]))
    write("r4_mp_norot", spec("norot", [dict(backbone="convnext_tiny", fold=f, seed=s, rot_aug=False,
                                             evals=["rotation", "crossdomain"], variant="maskpool", tag="norot")
                                        for s in seeds for f in folds]))
    dino = [dict(variant="plain"), dict(variant="maskpool"), dict(variant="strip")]
    write("r4_dinov2", spec("main", [dict(backbone="dinov2_small", fold=f, seed=0, lr_backbone=3.0e-5,
                                          evals=FIELD_C, **v) for f in folds for v in dino]))


if __name__ == "__main__":
    main()
