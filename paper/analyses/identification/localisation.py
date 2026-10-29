"""IMGT localisation of the highest-F1 p-IgGen latent of each J gene and matched SAE (Supplementary Table 14; the
activation profiles of Supplementary Figure 9). Over validation heavy chains of the latent's gene, the activation at
residue k (token 1 + k of '1' + VH + '2') is added to that residue's IMGT position (paper.analyses.identification.imgt).
Per latent: share of the activation at IMGT 115-128 (J region) and number of positions holding 90% of it.
Reads the per-token validation latents <work_dir>/activations/<sae>/val/ (scripts/extraction/encode.sh) and the F1
ranking identification/pairs/pairs_<sae>_top5.txt.
Writes <work_dir>/identification/imgt/localisation.csv and mass_<SAE>.npz (activation per IMGT position, keys
<gene>_<latent>, read by paper.figures.fig_imgt). Usage: python -m paper.analyses.identification.localisation (40 processes)"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.analyses.identification.imgt import CANONICAL_NUMBERING, build_alignment_map, in_jregion
from paper.benchmark._inputs import ranking

SAES = {"TopK": "piggen_topk_r8", "BatchTopK": "piggen_batchtopk_r8", "Ordered": "piggen_ordered_r8"}
JMASK = np.array([in_jregion(l) for l in CANONICAL_NUMBERING])
NPOS = len(CANONICAL_NUMBERING)


def row_of(f):
    return int(os.path.basename(f).split(".")[0].split("_")[0])


def work(chunk):
    """chunk: (latent file, latent, IMGT position of each residue, length, gene) -> {gene: activation per position}."""
    acc = {}
    for path, lat, pos, L, gene in chunk:
        col = np.asarray(np.load(path, mmap_mode="r")[1:1 + L, lat], dtype=np.float64)
        col[col < 0] = 0
        np.add.at(acc.setdefault(gene, np.zeros(NPOS)), pos, col)
    return acc


def main():
    out = P.work("identification", "imgt")
    os.makedirs(out, exist_ok=True)
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"))
    align = build_alignment_map(val)[0]
    gene_of = val.j_gene_anarci_heavy.astype(str).str.split("*").str[0]
    rows = []
    for arch, sae in SAES.items():
        top = {}
        for line in open(ranking(f"pairs_{sae}_top5.txt")):
            if ":" in line:
                g, l = line.strip().split(":")
                top.setdefault(g, int(l))
        files = {row_of(f): f for f in glob.glob(str(P.activations(sae, "val", "*.npy")))}
        tasks = []
        for row in val.index:
            g = gene_of[row]
            if g not in top or row not in align or row not in files:
                continue
            pos = np.array([i for i, aa in enumerate(align[row]) if aa != "-"])
            L = len(str(val.loc[row, "VHorVHH"]))
            if len(pos) == L:
                tasks.append((files[row], top[g], pos, L, g))
        mass = {}
        with Pool(40) as p:
            for acc in p.imap_unordered(work, [tasks[i::400] for i in range(400)]):
                for g, m in acc.items():
                    mass[g] = mass.get(g, 0) + m
        nseq = pd.Series([t[4] for t in tasks]).value_counts()
        for g, m in sorted(mass.items()):
            order, total = np.argsort(-m), m.sum()
            rows.append(dict(arch=arch, gene=g, latent=top[g], n_sequences=int(nseq[g]),
                             frac_J_115_128=float(m[JMASK].sum() / total),
                             n_pos_90=int(np.searchsorted(np.cumsum(m[order]) / total, 0.90) + 1)))
        np.savez(out / f"mass_{sae}.npz", **{f"{g}_{top[g]}": m for g, m in mass.items()})
        print(arch, "done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "localisation.csv", index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
