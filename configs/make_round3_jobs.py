"""Round-3 kernel spec (reviewer-driven): mask-pooling control.

  python configs/make_round3_jobs.py   -> configs/jobs/r3_maskpool_cnx.yaml
The localised modules (strip, slab-only, SVA) pool patch tokens with the leaf mask, the plain and
global models do not. `maskpool` adds the mask-weighted mean of the same patch tokens and nothing
else, so it tests whether the field-transfer gain is explained by knowing where the leaf is.
ConvNeXt-T, 5 folds x seeds 0-4 (same runs as the main table), field + MedLeaf-C.
"""
from pathlib import Path

import yaml

from make_main_jobs import spec

OUT = Path(__file__).resolve().parent / "jobs"


def main():
    jobs = [dict(backbone="convnext_tiny", fold=f, seed=s, evals=["crossdomain", "corruptions"], variant="maskpool")
            for s in range(5) for f in range(5)]
    s = spec("main", jobs)
    (OUT / "r3_maskpool_cnx.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
    print("r3_maskpool_cnx", len(s["jobs"]), "jobs")


if __name__ == "__main__":
    main()
