"""Scores of the ProGen2-OAS benchmark (temperature 1.0) with the benchmark's scorer (paper.benchmark.analyze_final:
germline-J readout, coverage per seed = numbered human heavy chains with a germline J segment / 1,000, the paper's
criteria). One latent per base task t10{a,b,c}_*; extension (t10x*) and refinement (t10r_*) tasks add alphas to the
same library directory. Only done tasks whose numbering has finished contribute alphas.
Writes <work_dir>/progen2_t10/score/{latents.csv, libraries.csv.gz, summary.md}; summary.md has the counts per set and
SAE (the tests of Supplementary Table 7 are paper.tables.benchmark_tests).
Usage: python -m paper.benchmark.progen2_t10.score [--partial]   (without --partial it refuses to run while tasks are
queued, claimed or unnumbered)"""
import os, re, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from plm_sae import paths as P
from paper.benchmark import analyze_final as af
B = P.work("progen2_t10")
PARTIAL = "--partial" in sys.argv


def kv(p): return dict(re.findall(r'^(\w+)="?(.*?)"?$', Path(p).read_text(), re.M))
def numbered(t): l = B / "anarci" / t / "assemble.log"; return l.exists() and "HI_DONE" in l.read_text()


def experiments():
    S = pd.read_csv(B / "selected.csv"); rows = {}
    for r in S.itertuples():
        rows[r.task] = dict(task=r.task, set=r.set, arch=r.arch, target=r.target, rank=r.rank, latent=r.latent, start=r.start, alphas=set(), dir=None, pending=0)
    for q in ("done", "claimed", "todo", "failed"):
        for t in sorted(os.listdir(B / "tasks" / q)):
            if "." in t: continue
            parent = t.split("_", 1)[1] if t.startswith(("t10x", "t10r")) else t
            e = rows[parent]; k = kv(B / "tasks" / q / t)
            if q != "done" or not numbered(t): e["pending"] += 1; continue
            for line in open(k["UNITS"]):
                if line.strip():
                    d, lat, g, a, s = line.split(); assert int(lat) == e["latent"] and g == e["target"]; e["dir"] = d; e["alphas"].add(float(a))
    E = pd.DataFrame(rows.values()); E["alphas"] = E.alphas.map(sorted)
    return E


def main():
    E = experiments(); npend = int((E.pending > 0).sum())
    if npend and not PARTIAL: sys.exit(f"{npend} latents still have queued/claimed/unnumbered tasks; use --partial")
    E = E[E.alphas.map(len) > 0].copy()
    E = E.assign(model="progen2_oas", sub=E.set, rule="t10", status="done", lowidx=False, matched_to=np.nan, phi="", members="")
    jobs = [(e.dir, e.latent, e.target, a, s) for e in E.itertuples() for a in e.alphas for s in (0, 1, 2)]
    with Pool(int(os.environ.get("NPROC", 16))) as p: L = pd.DataFrame(p.map(af.lib_stats, jobs, chunksize=16))
    L["formatted"] = L.get("formatted", False).fillna(False).astype(bool); L.to_csv(B / "score/libraries.csv.gz", index=False)
    R = af.score(E, L)
    R = R.merge(E[["latent", "target", "arch", "pending"]], on=["latent", "target", "arch"], how="left")
    R.to_csv(B / "score/latents.csv", index=False)
    out = [f"# ProGen2-OAS T = 1.0 + human FR1 prompt ({'PARTIAL' if PARTIAL else 'final'}; {len(R)} latents scored, {npend} with pending tasks)\n"]
    tab = R.groupby(["set", "arch"]).apply(lambda g: pd.Series(dict(n=len(g), corr=int((g.paper_corr == "pass").sum()), both=int((g.paper_mag == "pass").sum()),
                                                                    unsat=int(g.unsaturated.sum()), astar_med=g.paper_astar_med.median(),
                                                                    rho_med=round(g.paper_rho_med.median(), 2)))).reset_index()
    out.append(tab.to_string(index=False))
    txt = "\n".join(out); (B / "score/summary.md").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
