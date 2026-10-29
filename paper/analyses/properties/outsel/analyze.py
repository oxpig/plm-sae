"""Readouts and criteria of the property runs (Supplementary Sections A.10 and A.12).
Per library: numbered human heavy chains with J identity >= 0.5 (paper.benchmark.analyze_final.lib_stats n_Hj; coverage
= n_Hj / 1,000 per seed), heavy chains with a parsed CDRH3 and their Y-rich, positive and long counts
(paper.analyses.properties.props), and for developability runs the median FlashTAP metric over the scored sequences
(tap/<run>/a<alpha>_s<seed>.csv).
Per manifest row (target x SAE x set), the benchmark's criteria (alpha* per seed = largest alpha with coverage >= 80%;
correlation: Spearman rho >= 0.7, p < 0.05 over alpha <= alpha* in >= 2 of 3 seeds, in the target direction; magnitude,
CDRH3 properties only: >= 25% rise at alpha* in >= 2 seeds):
  CDRH3 properties   readout = the property's share of heavy chains with a parsed CDRH3
  developability     readout = median metric, steered down for PPC, PNC and PSH and up for SFvCSP
Output-selected against baseline latents: stratified exact test (paper.benchmark.rsets.strat_exact, strata = SAE x
target; the Mantel-Haenszel p of the paper).
Writes libraries.csv, results_graded.csv, results_dev.csv and SUMMARY.md in <work_dir>/outsel_properties.
Usage: python -m paper.analyses.properties.outsel.analyze   (NPROC)"""
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from plm_sae import paths as P
from plm_sae.steering.criterion import judge_pair
from paper.analyses.properties.props import props
from paper.benchmark import analyze_final as af
from paper.benchmark.rsets import strat_exact

S = P.work("outsel_properties")
LAB, DEV = ["Yrich", "positive", "long"], ["PPC", "PNC", "PSH", "SFvCSP"]
SIGN = {"PPC": -1, "PNC": -1, "PSH": -1, "SFvCSP": +1}


def kv(p):
    return dict(re.findall(r'^(\w+)="?(.*?)"?$', Path(p).read_text(), re.M))


def lib(u):
    d, lat, g, a, s, run = u
    r = af.lib_stats((d, int(lat), g, float(a), int(s)))
    o = dict(run=run, alpha=float(a), seed=int(s), n_gen=r["n_gen"], n_Hj=r["n_Hj"], n_cdr3=0, **{l: 0 for l in LAB})
    f = f"{d}/steered_l3_f{lat}_a{float(a):.2f}_seed{s}_seq1000.txt"
    if os.path.exists(f):
        for line in open(f):
            p = props(line.strip()) if line.strip() else None
            if p:
                o["n_cdr3"] += 1
                o["Yrich"] += p["Y"] >= 0.2
                o["positive"] += p["charge"] >= 1
                o["long"] += p["len"] >= 18
    t = S / "tap" / run / f"a{float(a):.2f}_s{s}.csv"
    if t.exists():
        T = pd.read_csv(t)
        o["n_tap"] = len(T)
        for m in DEV:
            o[f"med_{m}"] = float(T[m].median()) if len(T) else np.nan
    return o


def crit(AL, Pr, covs, sign):
    """paper.benchmark.analyze_final.paper with a hypothesised sign (+1 reproduces it)."""
    sd, nmag, ast, ach = [], 0, [], []
    for s in range(3):
        ok = np.where(covs[s] >= 0.8)[0]
        if not len(ok):
            sd.append(dict(spearman_rho=np.nan, spearman_p=np.nan))
            ast.append(np.nan)
            ach.append(np.nan)
            continue
        k = ok.max()
        y, m = Pr[s, :k + 1], ~np.isnan(Pr[s, :k + 1])
        ast.append(AL[k])
        ach.append(Pr[s, k])
        if m.sum() >= 3 and np.unique(y[m]).size >= 2:
            q = stats.spearmanr(AL[:k + 1][m], y[m])
            sd.append(dict(spearman_rho=q.statistic, spearman_p=q.pvalue))
        else:
            sd.append(dict(spearman_rho=np.nan, spearman_p=np.nan))
        if sign > 0 and Pr[s, 0] > 0 and not np.isnan(Pr[s, k]) and Pr[s, k] / Pr[s, 0] - 1 >= .25:
            nmag += 1
    v, npass, _ = judge_pair(sd, hypothesised_sign=sign)
    rh = [x["spearman_rho"] for x in sd]
    return dict(corr=v, corr_nseeds=npass, mag=("pass" if v == "pass" and nmag >= 2 else "fail") if sign > 0 else "",
                astar_med=np.nanmedian(ast) if not np.all(np.isnan(ast)) else np.nan,
                rho_med=np.nanmedian(rh) if not np.all(np.isnan(rh)) else np.nan,
                base=np.nanmean(Pr[:, 0]), at_astar=np.nanmean(ach))


def main():
    M = pd.read_csv(S / "manifest.csv")
    done = {t for t in os.listdir(S / "tasks/done") if "." not in t}
    units, alphas = [], {}
    for run in M.run.unique():
        for t in (t for t in done if t == run or t.endswith("_" + run)):
            for l in open(kv(S / "tasks/done" / t)["UNITS"]):
                if l.strip():
                    units.append(tuple(l.split()) + (run,))
                    alphas.setdefault(run, set()).add(float(l.split()[3]))
    with Pool(int(os.environ.get("NPROC", 16))) as p:
        L = pd.DataFrame(p.map(lib, sorted(set(units)), chunksize=8))
    L.to_csv(S / "libraries.csv", index=False)
    Li, G, D = L.set_index(["run", "alpha", "seed"]), [], []
    for r in M.itertuples():
        if r.run not in alphas:
            continue
        AL = np.array(sorted(alphas[r.run]))
        X = [[Li.loc[(r.run, a, s)] for a in AL] for s in (0, 1, 2)]
        covs = np.array([[x.n_Hj for x in xs] for xs in X], float) / 1000
        base = dict(target=r.target, arch=r.arch, set=r.set, latent=r.latent, start=r.start, run=r.run, n_alpha=len(AL),
                    alpha_max=AL[-1])
        if r.kind == "graded":
            num = np.array([[getattr(x, r.target) for x in xs] for xs in X], float)
            den = np.array([[x.n_cdr3 for x in xs] for xs in X], float)
            with np.errstate(all="ignore"):
                c = crit(AL, np.where(den > 0, num / den, np.nan), covs, +1)
            G.append({**base, **c, "gain_pp": 100 * (c["at_astar"] - c["base"])})
        elif f"med_{r.target}" in L.columns:
            Pr = np.array([[getattr(x, f"med_{r.target}") for x in xs] for xs in X], float)
            D.append({**base, **crit(AL, Pr, covs, SIGN[r.target])})
    RG, RD = pd.DataFrame(G), pd.DataFrame(D)
    RG.to_csv(S / "results_graded.csv", index=False)
    RD.to_csv(S / "results_dev.csv", index=False)
    out = []
    for name, R, cols in (("CDRH3 properties", RG, ["corr", "mag"]), ("Developability", RD, ["corr"])):
        if not len(R):
            continue
        R = R.assign(grp=R.set.str.replace(r"\d$", "", regex=True).map({"id_r": "f1", "out_r": "output", "base": "baseline"}),
                     stratum=R.arch + "|" + R.target)
        out.append(f"## {name}: latents passing, per SAE (of 3 per target)")
        for (a, g), x in R.groupby(["arch", "grp"]):
            out.append(f"- {a} {g}: " + ", ".join(f"{c} {int((x[c] == 'pass').sum())}/{len(x)}" for c in cols))
        strata = sorted(set(R.stratum))
        for c in cols:
            v = {g: R[R.grp == g].groupby("stratum")[c].apply(lambda s: list((s == "pass").astype(int))).reindex(strata)
                 for g in ("output", "baseline")}
            out.append(f"- {c}: output {sum(map(sum, v['output']))}/{sum(map(len, v['output']))} vs baseline "
                       f"{sum(map(sum, v['baseline']))}/{sum(map(len, v['baseline']))}, stratified exact (Mantel-Haenszel) "
                       f"p = {strat_exact(v['output'], v['baseline']):.3g}")
        out.append("")
    (S / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
