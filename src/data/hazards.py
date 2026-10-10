"""Hazard tiers used only by the (secondary) hazard-aware conformal deployment layer.

This is an explicit, documented ASSUMPTION table (paper Table S2), conservative by design:
  2 = high      ingestion of leaves is documented to be life-threatening
  1 = caution   toxic constituents documented in the plant (leaves or other organs);
                mammalian leaf toxicity evidence mixed/limited
  0 = baseline  no tier assigned here (does NOT imply safety)

Sources
- Nerium oleander: cardiac glycosides (oleandrin) in all parts; fatal human poisonings from leaves.
  Farkhondeh et al., Hum Exp Toxicol 2020, doi:10.1177/0960327120901571;
  human-death review, PMC11946565; Okuda et al., J Forensic Sci 2024, doi:10.1111/1556-4029.15393.
- Tabernaemontana divaricata: toxic indole alkaloids (e.g. voacangine, ibogamine-type) in leaves/branches
  (PMC5971036); acute rat leaf-extract toxicity low at 2000 mg/kg (PMC4587605) -> caution, not high.
- Pongamia pinnata: seeds poisonous (karanjin, pongamol; historic fish poison), leaves not established
  as toxic (World Agroforestry Pongamia factsheet; Purdue NewCROP) -> caution.
"""
from src.data.taxonomy import CLASSES

HAZARD_TIER = {c: 0 for c in CLASSES}
HAZARD_TIER["Nerium Oleander (Oleander)"] = 2
HAZARD_TIER["Tabernaemontana Divaricata (Crape Jasmine)"] = 1
HAZARD_TIER["Pongamia Pinnata (Indian Beech)"] = 1

TIERS = [HAZARD_TIER[c] for c in CLASSES]
