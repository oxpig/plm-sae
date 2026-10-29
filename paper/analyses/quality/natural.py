"""Per-sequence metrics of the natural references: the first 1,000 paired OAS training sequences, paired (p-IgGen) and
heavy chains only (IgLM, ProGen2-OAS) -> <work_dir>/quality/natural_{paired,heavy}.npz.
Usage: python -m paper.analyses.quality.natural"""
import os

import numpy as np

from paper.analyses.quality.common import B, first1000, per_sequence


def main():
    os.makedirs(B, exist_ok=True)
    T = first1000("train")
    vh_al, vl_al = T["aligned_VHorVHH"].tolist(), T["aligned_VL"].tolist()
    vh, vl = T["sequence_alignment_aa_heavy"].tolist(), T["sequence_alignment_aa_light"].tolist()
    for mode, args in (("paired", (vh_al, vl_al, vh, vl)), ("heavy", (vh_al, None, vh, None))):
        d = per_sequence(*args)
        np.savez(B / f"natural_{mode}.npz", **d)
        print(mode, {k: round(float(np.nanmean(v)), 4) for k, v in d.items()})


if __name__ == "__main__":
    main()
