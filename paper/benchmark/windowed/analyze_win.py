"""Scores of the windowed steering at natural scale (Supplementary Section A.4): the benchmark's readout and criteria
(paper.benchmark.analyze_final: target share among numbered human heavy chains with a germline J segment, alpha* =
largest alpha with coverage >= 80%) for every latent of every finished and numbered task in win/tasks/done. Alpha 0
is the task's unsteered library, shared by the latents of the task.
Writes <work_dir>/benchmark/win/win_latents.csv (bench, bench_mag: correlation and magnitude criteria; astar, astar_mult).
Usage: python -m paper.benchmark.windowed.analyze_win   (NPROC, default 16)"""
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import analyze_final as A

W = P.work("benchmark", "win")


def lib(args):
    """n_Hj (numbered human heavy chains with a germline J segment) and tgt (those called as the target)."""
    d, lat, g, a, s = args
    r = dict(dir=d, latent=lat, target=g, alpha=a, seed=s, n_Hj=0, tgt=0)
    fm = f"{d}/VH_alignment_layer_3_gene_{g}_features_[{lat}]_alpha_{a}_seed_{s}_sequences_1000_formatted.csv"
    if os.path.exists(fm) and os.path.getsize(fm) > 0:
        try:
            df = pd.read_csv(fm, usecols=["chain_type", "species", "j_gene_species", "j_gene", "j_gene_score"])
        except Exception:
            return r
        H = df[(df.chain_type == "H") & (df.species == "human") & (df.j_gene_species == "human")]
        jg, sc = H.j_gene.astype(str).str.split("*").str[0], H.j_gene_score
        r["n_Hj"], r["tgt"] = int((sc >= A.JMIN).sum()), int(((jg == g) & (sc >= A.JMIN)).sum())
    return r


def tasks():
    """win_configs.csv rows of the finished and numbered tasks, with their library directory."""
    C = pd.read_csv(W / "win_configs.csv")
    done = {os.path.basename(p) for p in glob.glob(f"{W}/tasks/done/*") if "." not in os.path.basename(p)}
    done = {t for t in done if os.path.exists(f"{W}/anarci/{t}/assemble.log")
            and "HI_DONE" in open(f"{W}/anarci/{t}/assemble.log").read()}
    C = C[C.task.isin(done)].copy()
    out = {t: re.search(r"OUT=(.*)", open(f"{W}/tasks/done/{t}").read()).group(1) for t in done}
    C["dir"] = [f"{out[t]}/{s}/{g}" for t, s, g in zip(C.task, C.set, C.gene)]
    return C


def main():
    C = tasks()
    jobs = list(dict.fromkeys((d, int(l), g, f"{a:.2f}", s) for d, l, g, a in zip(C.dir, C.latent, C.gene, C.alpha)
                              for s in (0, 1, 2)))
    with Pool(int(os.environ.get("NPROC", 16))) as p:
        L = pd.DataFrame(p.map(lib, jobs, chunksize=16))
    L["alpha"] = L.alpha.astype(float)
    Li = L.set_index(["dir", "latent", "target", "alpha", "seed"]).sort_index()
    base = {(r.model, r.arch, r.gene): [Li.loc[(r.dir, r.latent, r.gene, 0.0, s)] for s in (0, 1, 2)]
            for r in C[C.alpha == 0].itertuples()}
    rows = []
    for (m, arch, st, g, lat), c in C[C.alpha > 0].groupby(["model", "arch", "set", "gene", "latent"]):
        b = base.get((m, arch, g))
        if b is None:
            continue
        c = c.sort_values("alpha")
        AL, MU = np.r_[0.0, c.alpha.values], np.r_[0.0, c.mult.values]
        X = [[b[s]] + [Li.loc[(d, lat, g, a, s)] for d, a in zip(c.dir, c.alpha)] for s in (0, 1, 2)]
        nHj = np.array([[x.n_Hj for x in xs] for xs in X], float)
        tg = np.array([[x.tgt for x in xs] for xs in X], float)
        with np.errstate(all="ignore"):
            pb = A.paper(AL, tg / nHj, nHj / 1000)
        k = int(np.argmin(np.abs(AL - pb["paper_astar_med"]))) if pb["paper_astar_med"] == pb["paper_astar_med"] else 0
        rows.append(dict(model=m, arch=arch, set=st, target=g, latent=lat, start=c.start.iloc[0], end=c.end.iloc[0],
                         bench=pb["paper_corr"], bench_mag=pb["paper_mag"], astar=AL[k], astar_mult=MU[k]))
    D = pd.DataFrame(rows)
    D.to_csv(W / "win_latents.csv", index=False)
    print(D.groupby(["model", "set"]).bench.apply(lambda x: f"{(x == 'pass').sum()}/{len(x)}").to_string())


if __name__ == "__main__":
    main()
