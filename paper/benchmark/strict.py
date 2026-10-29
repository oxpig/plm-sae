"""Stricter readout (Supplementary Section A.4, "Stricter readout"): the target share is the number of human heavy
chains called as the target with J germline identity >= 0.8 over all sequences of the library, so every unnumbered,
non-human, low-identity or other-gene sequence counts against it. alpha* comes from the benchmark's 80% coverage gate,
and the paper's criteria (paper.benchmark.analyze_final.paper) are applied as in the benchmark. Check column chk_corr
(identity >= 0.5 over the benchmark's denominator) must equal the benchmark verdict.

  gated     p-IgGen, gated (highest-F1, baseline, output-selected) -> <work_dir>/benchmark/strict/strict_latents.csv
  existing  IgLM ungated and ProGen2-OAS (temperature 1.0)          -> <work_dir>/strict/strict_existing.csv
  summary   three latents per SAE-gene pair, k of 54 and stratified exact tests against the baselines
            -> <work_dir>/strict/summary.md (reads the two tables from <work_dir>, else <results_dir>)
Usage: python -m paper.benchmark.strict gated|existing|summary   (gated and existing need the libraries; NPROC)"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import analyze_final as A
from paper.benchmark import rsets as R
from paper.benchmark._inputs import inp

TS = (0.5, 0.8)


def lib(args):
    """Library counts: n_gen, n_Hj (as the benchmark) and t5, t8 (target calls with J identity >= 0.5, 0.8)."""
    d, lat, g, a, s = args
    r = dict(dir=d, latent=lat, target=g, alpha=a, seed=s, n_gen=np.nan, n_Hj=0, **{f"t{int(T * 10)}": 0 for T in TS})
    txt = f"{d}/steered_l3_f{lat}_a{a:.2f}_seed{s}_seq1000.txt"
    if os.path.exists(txt):
        r["n_gen"] = sum(1 for l in open(txt) if l.strip())
    fm = f"{d}/VH_alignment_layer_3_gene_{g}_features_[{lat}]_alpha_{a:.2f}_seed_{s}_sequences_1000_formatted.csv"
    if os.path.exists(fm) and os.path.getsize(fm) > 0:
        try:
            df = pd.read_csv(fm, usecols=["chain_type", "species", "j_gene_species", "j_gene", "j_gene_score"])
        except Exception:
            return r
        H = df[(df.chain_type == "H") & (df.species == "human") & (df.j_gene_species == "human")]
        tg = H.j_gene.astype(str).str.split("*").str[0] == g
        r["n_Hj"] = int((H.j_gene_score >= A.JMIN).sum())
        for T in TS:
            r[f"t{int(T * 10)}"] = int((tg & (H.j_gene_score >= T)).sum())
    return r


def score(items):
    """items: (meta dict, library dir, alphas). One row per latent: chk_corr, astar, s8_corr, s8_mag, s8_base, s8_ach."""
    jobs = list(dict.fromkeys((d, int(m["latent"]), m["target"], float(a), s) for m, d, al in items for a in al
                              for s in (0, 1, 2)))
    with Pool(int(os.environ.get("NPROC", 16))) as p:
        L = pd.DataFrame(p.map(lib, jobs, chunksize=32)).set_index(["dir", "latent", "target"]).sort_index()
    out = []
    for m, d, al in items:
        AL = np.array(sorted(al))
        g = L.loc[[(d, int(m["latent"]), m["target"])]].reset_index()
        g["ai"] = g.alpha.map(lambda a: int(np.argmin(np.abs(AL - a))))
        X = {c: np.full((3, len(AL)), np.nan) for c in ["n_gen", "n_Hj", "t5", "t8"]}
        for x in g.itertuples():
            for c in X:
                X[c][int(x.seed), x.ai] = getattr(x, c)
        covs = np.nan_to_num(X["n_Hj"]) / 1000
        with np.errstate(all="ignore"):
            P_bench, P8 = np.where(X["n_Hj"] > 0, X["t5"] / X["n_Hj"], np.nan), X["t8"] / X["n_gen"]
        c0, q = A.paper(AL, P_bench, covs), A.paper(AL, P8, covs)
        out.append({**m, "chk_corr": c0["paper_corr"], "astar": c0["paper_astar_med"], "s8_corr": q["paper_corr"],
                    "s8_mag": q["paper_mag"], "s8_base": q["paper_base"], "s8_ach": q["paper_ach"]})
    O = pd.DataFrame(out)
    print("check: identity >= 0.5 reproduces the benchmark verdict for", int((O.chk_corr == O.bench_corr).sum()), "of", len(O))
    return O


def gated():
    F = P.work("benchmark")
    R_ = pd.read_csv(F / "final_latents.csv")
    R_ = R_[(R_.model == "piggen") & R_.set.isin(["f1", "baseline", "output"])]
    E = {(e.task, e.latent, e.target): e for e in A.experiments().itertuples()}
    items = []
    for r in R_.itertuples():
        e = E[(r.task, r.latent, r.target)]
        items.append((dict(model=r.model, set=r.set, arch=r.arch, target=r.target, latent=r.latent, task=r.task,
                           bench_corr=r.paper_corr), e.dir, e.alphas))
    os.makedirs(F / "strict", exist_ok=True)
    score(items).to_csv(F / "strict" / "strict_latents.csv", index=False)


def existing():
    from paper.benchmark.progen2_t10 import score as S10
    FE = {(e.task, int(e.latent), e.target): e for e in A.experiments().itertuples()}
    TE = {e.task: e for e in S10.experiments().itertuples()}
    items = []
    for r in R.existing_sets().query("model in ['IgLM', 'progen2_oas']").itertuples():
        m = dict(model=r.model, arch=r.arch, target=r.target, latent=int(r.latent), rank=int(r.rank), set=r.set,
                 task=r.task, bench_corr=r.paper_corr)
        e = FE.get((r.task, int(r.latent), r.target)) if r.model == "IgLM" else TE.get(r.task)
        items.append((m, e.dir, e.alphas))
    os.makedirs(P.work("strict"), exist_ok=True)
    score(items).to_csv(P.work("strict", "strict_existing.csv"), index=False)


def summary():
    POS, RD, OR = R.rank_tables()
    G = pd.read_csv(inp("benchmark", "strict", "strict_latents.csv"))
    G = G[G.model == "piggen"].drop(columns=["rank"], errors="ignore")
    G = pd.concat([G[G.set == s].merge(rk, on=R.KEY) for s, rk in (("f1", POS), ("baseline", RD), ("output", OR))])
    T = pd.concat([G, pd.read_csv(inp("strict", "strict_existing.csv"))], ignore_index=True)
    T = T[T["rank"].between(1, R.K)].drop_duplicates(R.KEY + ["set"])
    sets = ["f1", "baseline", "output"]
    lines = ["# Stricter readout (J identity >= 0.8 over all sequences), three latents per SAE-gene pair, k of 54", "",
             "p-IgGen gated, IgLM ungated, ProGen2-OAS at temperature 1.0. p: stratified exact test against the baselines.",
             "", "| Model | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for m in R.M:
        x = T[T.model == m]
        sm = R.summarise(x, sets, "s8_corr", ["arch", "target"], "baseline")
        cells = [f"{sm[s]['k']}" + (f" (p = {sm[s]['p_vs_baseline']:.2g})" if "p_vs_baseline" in sm[s] else "") for s in sets]
        lines.append(f"| {m} | " + " | ".join(cells) + " |" + ("" if sm["complete"] else f" INCOMPLETE {sm.get('missing')}"))
    os.makedirs(P.work("strict"), exist_ok=True)
    open(P.work("strict", "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"gated": gated, "existing": existing, "summary": summary}[sys.argv[1]]()
