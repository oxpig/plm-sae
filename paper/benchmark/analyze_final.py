"""Scoring of the steering benchmark (Methods; Supplementary Section A.2): every steered latent of every finished task.

Experiments = (library directory, latent, target) from the units files of the tasks in tasks/{done,failed,claimed,
todo}; the latent set (metadata()) from the directory layout and positions/*.csv.
Per library (lib_stats): numbered human heavy chains (chain, V and J species human) with their J calls, those with a
germline J segment (J identity >= JMIN = 0.5; ANARCI calls IGHJ2 for chains without one), numbered light chains, and the
residue composition of positions 1-95.
Readout: the target J's share of the numbered human heavy chains with a germline J segment; for kappa / lambda, the
target's share of the human light chains. Coverage of a library: those chains / 1,000 (light chains: relative to
alpha 0).
Per latent:
  valid_amax       largest alpha of the contiguous range from 0 whose pooled coverage (three seeds) is >= 80%
  unsaturated      the coverage is still >= 80% at the last alpha of the (extended) ladder
  vtv_valid_amax   total variation between the residue composition (positions 1-95) at alpha 0 and at valid_amax
  paper_*          the correlation and magnitude criteria (paper(), below)
Writes final_latents.csv and final_libraries.csv.gz (per-library counts) in <work_dir>/benchmark.
Usage: python -m paper.benchmark.analyze_final   (environment NPROC, default 24)"""
import collections
import os
import re
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from plm_sae import paths as P
from plm_sae.steering.criterion import judge_pair

warnings.filterwarnings("ignore")
F = P.work("benchmark")
GENES = [f"IGHJ{i}" for i in range(1, 7)]
AA = "ACDEFGHIKLMNPQRSTVWY"
JMIN = 0.5
_RUNS = re.compile(r"^(?:.*/)?(runs/(?:piggen|IgLM|progen2_oas)/.*)$")


def rel_run_dir(d):
    """'.../runs/<model>/...' -> 'runs/<model>/...': library directories as stored in the result tables, relative to
    their stage, so that results can be read from any location."""
    m = _RUNS.match(str(d))
    return m.group(1) if m else str(d)


def task_files(root=F):
    out = []
    for q in ("done", "failed", "claimed", "todo"):
        if not (root / "tasks" / q).is_dir():
            continue
        for t in sorted(os.listdir(root / "tasks" / q)):
            if "." in t:
                continue
            kv = dict(re.findall(r'^(\w+)="?(.*?)"?$', (root / "tasks" / q / t).read_text(), re.M))
            starts = {}
            for line in open(kv["CFG"].split(" ")[0]):
                if line.strip():
                    lat, _, st, tag = line.split()[:4]
                    starts[(int(lat), tag.split("/")[1])] = int(st)
            out.append(dict(task=t, status=q, model=kv["MODEL"], units=kv["UNITS"], out=kv["OUT"], starts=starts))
    return out


def experiments(root=F):
    rows = {}
    for t in task_files(root):
        for line in open(t["units"]):
            if not line.strip():
                continue
            d, lat, g, a, s = line.split()
            e = rows.setdefault((d, int(lat), g), dict(dir=d, latent=int(lat), target=g, task=t["task"], status=t["status"],
                                                       model=t["model"], start=t["starts"].get((int(lat), g)), alphas=set()))
            if t["status"] == "done":
                e["alphas"].add(float(a))   # alphas of unfinished (extension) tasks are not scored yet
            elif not e["alphas"]:
                e["status"] = t["status"]
    E = pd.DataFrame(rows.values())
    E["alphas"] = E.alphas.map(sorted)
    E.loc[E.alphas.map(len) > 0, "status"] = "done"
    E["rule"] = E.dir.map(lambda d: Path(d).parent.name)
    rel = E.dir.map(lambda d: Path(rel_run_dir(d)).parts[1:])   # model / SAE / [sub] / rule / gene
    E["arch"] = rel.map(lambda p: p[1].replace("BatchTopK", "Batch TopK"))
    E["sub"] = rel.map(lambda p: p[2] if len(p) == 5 else "")
    return E


def metadata(E):
    """Set label, F1 rank, co-activation group and matched latent of each experiment. Sets: f1, baseline, output (each
    gated or "<set> ungated"), joint, light chain, developability, section41."""
    rank, ctrl, joint = {}, {}, {}
    for m in ("piggen", "IgLM", "progen2_oas"):
        if (F / f"positions/{m}_positions.csv").exists():
            for r in pd.read_csv(F / f"positions/{m}_positions.csv").itertuples():
                rank[(m, r.arch, r.gene, int(r.latent))] = int(r.rank)
        for f in (F / f"positions/{m}_controls.csv", F / f"positions/{m}_controls2.csv"):
            if f.exists():
                for r in pd.read_csv(f).itertuples():
                    ctrl[(m, r.arch, r.gene, int(r.latent))] = int(r.matched_to)
    if (F / "manifest_joint.csv").exists():
        for r in pd.read_csv(F / "manifest_joint.csv").itertuples():
            joint[(r.arch, r.phi, r.gene, int(r.latent))] = r.members
    named = {"lct": "light chain", "dev": "developability", "section41": "section41"}
    out = []
    for e in E.itertuples():
        k = (e.model, e.arch, e.target, e.latent)
        un = "" if e.rule == "gated" else " ungated"
        d = dict(rank=np.nan, matched_to=np.nan, members="", phi="")
        if e.sub == "baseline":
            d.update(set="baseline" + un, matched_to=ctrl.get(k, np.nan))
        elif e.sub == "output":
            d.update(set="output" + un)
        elif e.sub.startswith("joint_"):
            phi = e.sub.split("_")[1]
            d.update(set="joint", phi=phi, members=joint.get((e.arch, phi, e.target, e.latent), ""))
        elif e.sub in named:
            d.update(set=named[e.sub])
        else:
            d.update(set="f1" + un, rank=rank.get(k, np.nan))
        out.append(d)
    return pd.concat([E.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def lib_stats(args):
    """Counts of one library: n_gen, numbered human heavy chains (n_H) and their J calls (IGHJk), those with a germline
    J segment (n_Hj, IGHJk_g), light chains (n_light, n_K, n_L), residue counts of positions 1-95, formatted."""
    d, lat, g, a, s = args
    r = dict(dir=d, latent=lat, target=g, alpha=a, seed=s, n_gen=0, n_H=0, n_Hj=0, n_light=0, n_K=0, n_L=0,
             **{j: 0 for j in GENES}, **{j + "_g": 0 for j in GENES}, **{c: 0 for c in AA})
    txt = f"{d}/steered_l3_f{lat}_a{a:.2f}_seed{s}_seq1000.txt"
    if os.path.exists(txt):
        c, n = collections.Counter(), 0
        for l in open(txt):
            l = l.strip()
            if l:
                n += 1
                c.update(l[:95])
        r["n_gen"] = n
        r.update({x: c.get(x, 0) for x in AA})
    fm = f"{d}/VH_alignment_layer_3_gene_{g}_features_[{lat}]_alpha_{a:.2f}_seed_{s}_sequences_1000_formatted.csv"
    if os.path.exists(fm) and os.path.getsize(fm) > 0:
        try:
            df = pd.read_csv(fm, usecols=["chain_type", "species", "j_gene_species", "j_gene", "j_gene_score"])
        except Exception:
            return r
        H = df[(df.chain_type == "H") & (df.species == "human") & (df.j_gene_species == "human")]
        jg = H.j_gene.astype(str).str.split("*").str[0].value_counts()
        r.update(n_H=len(H), **{j: int(jg.get(j, 0)) for j in GENES})
        Hg = H[H.j_gene_score >= JMIN]
        jgg = Hg.j_gene.astype(str).str.split("*").str[0].value_counts()
        r.update(n_Hj=len(Hg), **{j + "_g": int(jgg.get(j, 0)) for j in GENES})
        L = df[df.chain_type.isin(["K", "L"])]
        Lh = L[L.species == "human"]
        r.update(n_light=len(L), n_K=int((Lh.chain_type == "K").sum()), n_L=int((Lh.chain_type == "L").sum()))
        r["formatted"] = True
    return r


def libraries(E, cache=F / "final_libraries.csv.gz"):
    jobs = [(e.dir, e.latent, e.target, a, s) for e in E.itertuples() for a in e.alphas for s in (0, 1, 2)]
    with Pool(int(os.environ.get("NPROC", 24))) as p:
        L = pd.DataFrame(p.map(lib_stats, jobs, chunksize=64))
    L["formatted"] = L.get("formatted", False).fillna(False).astype(bool)
    if cache:
        L.to_csv(cache, index=False)
    return L


def tv(c0, c1):
    """Total variation between two count vectors."""
    a, b = np.asarray(c0, float), np.asarray(c1, float)
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum() if a.sum() and b.sum() else np.nan


def paper(AL, P, covs):
    """The criteria of the paper. Per seed, alpha* = the largest alpha with coverage >= 80%. Correlation: Spearman's
    rho of the readout over alpha <= alpha* (plm_sae.steering.criterion.judge_pair: rho >= 0.7, p < 0.05, in at least
    2 of 3 seeds). Magnitude: correlation passes and at least 2 seeds gain >= 25% at alpha* over alpha 0.
    AL: alphas; P: [3, len(AL)] readout per seed; covs: [3, len(AL)] coverage per seed."""
    sd, nmag, ast, ach = [], 0, [], []
    for s in range(3):
        ok = np.where(covs[s] >= 0.8)[0]
        if not len(ok):
            sd.append(dict(spearman_rho=np.nan, spearman_p=np.nan))
            ast.append(np.nan)
            ach.append(np.nan)
            continue
        k = ok.max()
        y = P[s, :k + 1]
        m = ~np.isnan(y)
        ast.append(AL[k])
        ach.append(P[s, k])
        if m.sum() >= 3 and np.unique(y[m]).size >= 2:
            q = stats.spearmanr(AL[:k + 1][m], y[m])
            sd.append(dict(spearman_rho=q.statistic, spearman_p=q.pvalue))
        else:
            sd.append(dict(spearman_rho=np.nan, spearman_p=np.nan))
        if P[s, 0] > 0 and not np.isnan(P[s, k]) and P[s, k] / P[s, 0] - 1 >= .25:
            nmag += 1
    v, npass, _ = judge_pair(sd, hypothesised_sign=+1)
    with np.errstate(all="ignore"):
        bm, bs, am, asd = np.nanmean(P[:, 0]), np.nanstd(P[:, 0], ddof=1), np.nanmean(ach), np.nanstd(ach, ddof=1)
    return dict(paper_corr=v, paper_mag="pass" if v == "pass" and nmag >= 2 else "fail", paper_nmag=nmag,
                paper_npass=npass, paper_base=bm, paper_base_sd=bs, paper_ach=am, paper_ach_sd=asd,
                paper_astar_med=np.nanmedian(ast) if not np.all(np.isnan(ast)) else np.nan,
                paper_rho_med=np.nanmedian([x["spearman_rho"] for x in sd])
                if not all(np.isnan(x["spearman_rho"]) for x in sd) else np.nan)


def score(E, L):
    """One row per experiment of E (with metadata) from the library table L."""
    L = L.set_index(["dir", "latent", "target"]).sort_index()
    rows = []
    for e in E.itertuples():
        AL, key = np.array(e.alphas), (e.dir, e.latent, e.target)
        if key not in L.index:
            continue
        g = L.loc[[key]].reset_index()
        g["ai"] = g.alpha.map(lambda a: int(np.argmin(np.abs(AL - a))))
        Hn, J = np.zeros((3, len(AL))), np.zeros((3, len(AL), 6))
        Ln, K, Lm = np.zeros((3, len(AL))), np.zeros((3, len(AL))), np.zeros((3, len(AL)))
        comp, fmt = np.zeros((len(AL), 20)), np.zeros((3, len(AL)), bool)
        for x in g.itertuples():
            s, j = int(x.seed), x.ai
            Hn[s, j] = x.n_Hj
            J[s, j] = [getattr(x, q + "_g") for q in GENES]
            Ln[s, j], K[s, j], Lm[s, j] = x.n_light, x.n_K, x.n_L
            comp[j] += [getattr(x, c) for c in AA]
            fmt[s, j] = x.formatted
        with np.errstate(all="ignore"):
            if e.target in GENES:
                P = np.where(Hn > 0, J[:, :, GENES.index(e.target)] / Hn, np.nan)
                cov, covs = Hn.sum(0) / 3000, Hn / 1000
            elif e.target in ("kappa", "lambda"):
                num, den = (K if e.target == "kappa" else Lm), K + Lm
                P = np.where(den > 0, num / den, np.nan)
                cov, covs = Ln.sum(0) / max(Ln[:, 0].sum(), 1), Ln / np.maximum(Ln[:, :1], 1)
            else:   # developability: scored with FlashTAP (paper.analyses.properties.dev_steering)
                P, cov, covs = np.full((3, len(AL)), np.nan), Hn.sum(0) / 3000, None
        bad = np.where(~(cov >= 0.8))[0]
        k = bad[0] if len(bad) else len(AL)
        r = dict(model=e.model, set=e.set, arch=e.arch, sub=e.sub, rule=e.rule, start=e.start, target=e.target,
                 latent=e.latent, rank=e.rank, matched_to=e.matched_to, phi=e.phi, members=e.members,
                 task=e.task, status=e.status, n_alpha=len(AL), alpha_max=AL[-1], libs_formatted=int(fmt.sum()),
                 libs_expected=3 * len(AL), valid_amax=AL[k - 1] if k else np.nan,
                 vtv_valid_amax=tv(comp[0], comp[k - 1]) if k else np.nan, unsaturated=bool(k == len(AL)))
        if covs is not None:
            r.update(paper(AL, P, covs))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    E = metadata(experiments())
    done = E[E.status == "done"]
    R = score(done, libraries(done))
    R.to_csv(F / "final_latents.csv", index=False)
    pend = E[E.status != "done"].groupby(["status", "set", "model"]).size()
    print(R.groupby(["model", "set"]).paper_corr.apply(lambda x: f"{(x == 'pass').sum()}/{len(x)}").to_string())
    if len(pend):
        print("not finished:\n" + pend.to_string())


if __name__ == "__main__":
    main()
