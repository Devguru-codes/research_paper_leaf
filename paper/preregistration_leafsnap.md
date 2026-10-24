# Pre-registered predictions: LeafSnap30 replication

Written on 2026-10-25, before any LeafSnap30 model was trained or evaluated. The commit that adds this
file to the public repository is its timestamp.

## Design (fixed now)

- Data: LeafSnap30 (Zenodo 10.5281/zenodo.5061353, CC BY 4.0): 30 tree species, 5,320 lab photographs of
  pressed leaves (training, 5-fold cross-validation grouped by physical leaf) and 2,075 field phone
  photographs (test only). Leaf masks for both come from a label-free background-colour mask.
- Models: ConvNeXt-T plain, patch pooling, mask pooling, strip pooling, SVA; same code and recipe as
  the medicinal study, except 15 epochs (LeafSnap's training set is 2.9 times larger, so the number of
  optimisation steps is similar).
- Runs: 5 folds x 2 seeds per model. Outcome: 30-way accuracy on raw field photographs (primary) and on
  background-whitened field photographs (secondary).
- Analysis: the same contrasts, corrected resampled t-test, Holm correction within the family and TOST
  (+-3 points) as in the medicinal study (src/study_contrasts.py), applied unchanged.

## Predictions

The medicinal study attributed the field gain to (a) leaf location, i.e. discounting a cluttered
background, and (b) localised pooling. LeafSnap field photographs show leaves on plain paper, not on
clutter. We therefore predict:

1. **Total gain:** strip pooling and SVA beat plain on raw field photographs (positive difference,
   significant after Holm).
2. **Leaf location is smaller than on DIMPSAR:** mask pooling minus patch pooling on raw field
   photographs is smaller than the 4.4 points measured on DIMPSAR, because there is little
   background clutter to discount.
3. **Localisation still helps:** strip pooling minus mask pooling is positive on raw field photographs.
4. **Measured geometry adds little:** SVA minus strip pooling lies within +-3 points.

Any outcome will be reported, including predictions that fail.
