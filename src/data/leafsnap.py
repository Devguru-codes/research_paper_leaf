"""LeafSnap30 (Zenodo 10.5281/zenodo.5061353, CC BY 4.0) as a second studio-to-field pair.

  python -m src.data.leafsnap extract   # zip -> data_leafsnap/lab/<species>/, data_leafsnap/field/<species>/
  MEDLEAF_RAW=data_leafsnap/lab MEDLEAF_CACHE=cache_leafsnap python -m src.data.audit
  python -m src.data.leafsnap field     # -> cache_leafsnap_field/ (crops + masks + dimpsar_index.csv)

Lab images are crops of pressed leaves; some still show the colour chart / ruler, which the mask's
leaf-colour score rejects. Each pressed leaf was photographed several
times (file names <plant>-<leaf>-<k>), so all photographs of one leaf form one group
(lab/specimen_groups.csv) and never cross folds. Grouping by plant is impossible: some species come
from a single plant. Field images are phone photographs of leaves on plain paper outdoors; both lab and
field masks come from the label-free background-colour mask (src.data.masks.bg_colour_mask), the
audit is run with MEDLEAF_MASK=bgcolour.
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

from src.paths import PROJECT

ZIP = PROJECT / "data_leafsnap" / "leafsnap30.zip"
OUT = PROJECT / "data_leafsnap"
PREFIX = "leafsnap-dataset-30subset/"


def extract():
    z = zipfile.ZipFile(ZIP)
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(PREFIX + "leafsnap-dataset-30subset-images.txt"), "utf-8"),
                               delimiter="\t"))
    groups = []
    for r in rows:
        src, species, path = r["source"].strip(), r["species"].strip().replace(" ", "_"), r["image_path"].strip()
        name = Path(path).name  # only the file name is used: no paths from the archive
        dst = OUT / src / species / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(PREFIX + path))
        if src == "lab":
            groups.append(dict(rel_path=f"{species}/{name}", group=species + "/" + "-".join(name.split("-")[:2])))
    with open(OUT / "lab" / "specimen_groups.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["rel_path", "group"])
        w.writeheader()
        w.writerows(groups)
    print({s: sum(r["source"] == s for r in rows) for s in ("lab", "field")}, "lab leaves:",
          len({g["group"] for g in groups}))


def field(out: Path = PROJECT / "cache_leafsnap_field"):
    from src.crossdomain import build
    from src.data.masks import bg_colour_mask
    classes = sorted(d.name for d in (OUT / "lab").iterdir() if d.is_dir())
    build(None, str(out), root=OUT / "field", mapping={c: i for i, c in enumerate(classes)}, mask_fn=bg_colour_mask)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    elif sys.argv[1] == "field":
        field()
