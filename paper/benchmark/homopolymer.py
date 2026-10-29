"""Homopolymer runs in the CDRH3 of steered p-IgGen libraries (Supplementary Section A.4, "Effect on the CDRH3").

Per library: the share of heavy chains whose CDRH3 region contains a run of at least four identical residues. Heavy
chain: the sequence up to the end of FR4 (first W[GA]xG at index >= 90, plus 11 residues; else the first 'VSS' after
95; else the first 130 residues). CDRH3 region: after the C of the last [YF][YFH]C between positions 80 and 110 (else
from 85) up to the W[GA]xG motif plus 4 residues.
  measure   output-selected and baseline latents of p-IgGen in final_latents.csv: the share at
            alpha 0 (homo0) and at alpha* (homo1, the ladder alpha nearest paper_astar_med), three seeds pooled
            -> <work_dir>/benchmark/homopolymer/homopolymer_latents.csv
  summary   median rise (percentage points) over the output-selected and baseline latents of the paper's sets
            (paper.benchmark.rsets, ranks 1-3)
Usage: python -m paper.benchmark.homopolymer measure|summary   (measure: NPROC, default 36)"""
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import analyze_final as A
from paper.benchmark import rsets
from paper.benchmark._inputs import inp

WG, CYS, HOMO = re.compile(r"W[GA].G"), re.compile(r"[YF][YFH]C"), re.compile(r"(.)\1{3,}")


def cdr3_region(line):
    """CDRH3 region of a p-IgGen paired sequence (see the module docstring)."""
    m = [x for x in WG.finditer(line) if x.start() >= 90]
    if m:
        w = m[0].start()
        hend = min(len(line), w + 11)
    else:
        w = None
        k = line.find("VSS", 95)
        hend = k + 3 if k > 0 else min(len(line), 130)
    heavy, c = line[:hend], None
    for x in CYS.finditer(heavy[:w] if w else heavy):
        if 80 <= x.start() + 2 <= 110:
            c = x.start() + 2
    return heavy[(c + 1 if c is not None else 85):(w + 4 if w is not None else len(heavy))]


def lib(args):
    """(sequences, sequences with a homopolymer run in the CDRH3 region) of one library."""
    d, lat, a, s = args
    p = f"{d}/steered_l3_f{lat}_a{a:.2f}_seed{s}_seq1000.txt"
    n = k = 0
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                n += 1
                k += bool(HOMO.search(cdr3_region(line.strip())))
    return n, k


def measure():
    F = P.work("benchmark")
    R = pd.read_csv(F / "final_latents.csv")
    R = R[(R.model == "piggen") & R.set.isin(["output", "baseline"])]
    E = A.experiments()
    meta, jobs = [], []
    for r in R.itertuples():
        e = E[(E.task == r.task) & (E.latent == r.latent) & (E.target == r.target)].iloc[0]
        al = np.array(e.alphas)
        a1 = float(al[np.argmin(np.abs(al - (r.paper_astar_med if r.paper_astar_med == r.paper_astar_med else 0.0)))])
        meta.append((r, e.dir, a1))
        jobs += [(e.dir, int(r.latent), a, s) for a in (0.0, a1) for s in (0, 1, 2)]
    jobs = list(dict.fromkeys(jobs))
    with Pool(int(os.environ.get("NPROC", 36))) as p:
        res = dict(zip(jobs, p.map(lib, jobs, chunksize=8)))
    rows = []
    for r, d, a1 in meta:
        share = []
        for a in (0.0, a1):
            n, k = map(sum, zip(*[res[(d, int(r.latent), a, s)] for s in (0, 1, 2)]))
            share.append(k / max(n, 1))
        rows.append(dict(model=r.model, set=r.set, arch=r.arch, latent=r.latent, target=r.target, astar=a1,
                         homo0=share[0], homo1=share[1]))
    os.makedirs(F / "homopolymer", exist_ok=True)
    pd.DataFrame(rows).to_csv(F / "homopolymer" / "homopolymer_latents.csv", index=False)


def summary():
    D = pd.read_csv(inp("benchmark", "homopolymer", "homopolymer_latents.csv"))
    E = rsets.existing_sets()
    E = E[E.model == "piggen"]
    for s in ("output", "baseline"):
        x = D[D.set == s].merge(E[E.set == s][["arch", "target", "latent"]], on=["arch", "target", "latent"])
        print(f"p-IgGen {s}: {len(x)} latents, median rise in CDRH3 homopolymer share "
              f"{(100 * (x.homo1 - x.homo0)).median():.1f} percentage points")


if __name__ == "__main__":
    {"measure": measure, "summary": summary}[sys.argv[1]]()
