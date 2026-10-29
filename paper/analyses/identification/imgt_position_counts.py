"""IMGT activations of the three highest-F1 IGHJ4 latents of the IgLM and ProGen2-OAS TopK SAEs (Supplementary Figure 8,
the "five of six" of Section 4.3): over the IGHJ4 validation heavy chains, the number of residues with a non-zero
activation and the activation at each IMGT position (paper.analyses.identification.imgt). Residue k sits at token 2 + k
in IgLM ('[HEAVY][HUMAN]' + VH) and 1 + k in ProGen2-OAS ('1' + VH + '2').
Reads <work_dir>/activations/{iglm,progen2oas}_topk_r8/val_ighj4_tok/ (paper/scripts/figures/encode_ighj4_tokens.sh).
Writes <work_dir>/identification/imgt/imgt_position_counts.json (read by paper.figures.fig_imgt).
Usage: python -m paper.analyses.identification.imgt_position_counts   (40 processes)"""
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.analyses.identification.imgt import CANONICAL_NUMBERING, build_alignment_map, subset_csv
from paper.analyses.identification.localisation import row_of

NPOS = len(CANONICAL_NUMBERING)
SETS = [("IgLM", "iglm_topk_r8", 2, [3036, 1565, 2705]), ("ProGen2-OAS", "progen2oas_topk_r8", 1, [8218, 2303, 9879])]


def work(chunk):
    cnt = mass = None
    for path, lats, pos, start, L in chunk:
        cols = np.asarray(np.load(path, mmap_mode="r")[start:start + L][:, lats], dtype=np.float64)
        if cnt is None:
            cnt, mass = np.zeros((len(lats), NPOS)), np.zeros((len(lats), NPOS))
        np.add.at(cnt.T, pos, (cols > 0).astype(float))
        np.add.at(mass.T, pos, np.where(cols > 0, cols, 0))
    return cnt, mass


def main():
    out = P.work("identification", "imgt")
    os.makedirs(out, exist_ok=True)
    val = pd.read_csv(subset_csv("IGHJ4")).set_index("orig_row", drop=False)
    align = build_alignment_map(val)[0]
    res = {}
    for model, sae, offset, lats in SETS:
        files = {row_of(f): f for f in glob.glob(str(P.activations(sae, "val_ighj4_tok", "*.npy")))}
        tasks = []
        for row in val.index:
            if row not in align or row not in files:
                continue
            pos = np.array([i for i, aa in enumerate(align[row]) if aa != "-"])
            L = len(str(val.loc[row, "VHorVHH"]))
            if len(pos) == L and offset + L <= 200:
                tasks.append((files[row], np.array(lats), pos, offset, L))
        with Pool(40) as p:
            parts = [x for x in p.map(work, [tasks[i::400] for i in range(400)]) if x[0] is not None]
        cnt, mass = sum(c for c, _ in parts), sum(m for _, m in parts)
        for k, lat in enumerate(lats):
            res[f"{model}|{lat}"] = dict(n_sequences=len(tasks), count=cnt[k].tolist(), mass=mass[k].tolist())
        print(model, len(tasks), "sequences", flush=True)
    json.dump(dict(positions=CANONICAL_NUMBERING, **res), open(out / "imgt_position_counts.json", "w"))


if __name__ == "__main__":
    main()
