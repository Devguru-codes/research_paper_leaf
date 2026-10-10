# Data

The images are not redistributed here. To reproduce:

1. **Medicinal Leaf Dataset (segmented subset)**, Roopashree S. and Anitha J., Mendeley Data (2020),
   DOI [10.17632/nnytj2v3n5.1](https://doi.org/10.17632/nnytj2v3n5.1), CC BY 4.0.
   Download the *Segmented Medicinal Leaf Images* folder and either place it under `data/` as
   `data/Segmented Medicinal Leaf Images-20261003T153133Z-1-001/Segmented Medicinal Leaf Images/`
   (the default in `src/paths.py`) or set `MEDLEAF_RAW` to its location.
   Then run `python -m src.data.audit` to build `cache/` (crops, masks, vein pseudo-labels, index).
2. **DIMPSAR** field photographs, Pushpa B. R. and Shobha Rani N., *Data in Brief* 2023,
   DOI [10.1016/j.dib.2023.109388](https://doi.org/10.1016/j.dib.2023.109388), CC BY 4.0
   (Kaggle mirror: `aryashah2k/indian-medicinal-leaves-dataset`). Train our leaf masker with
   `python -m src.train_maskhead`, then build the leaf-centred field set with
   `python -m src.crossdomain build --masker <maskhead.pt> --out <dir>` and set `DIMPSAR_BUILT=<dir>`.

## `audit/`

- `index.csv`: one row per image with its class, the exact-duplicate group, the leaf (re-shot) group
  used for grouped cross-validation, and whether it was kept after de-duplication (1,825 of 1,835).
- `audit.json`: summary of the audit (duplicate pairs, re-shot links, file-name collisions).
