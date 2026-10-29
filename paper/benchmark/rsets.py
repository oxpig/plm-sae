"""Comparison sets of the steering benchmark and the stratified exact test (Table 2, Supplementary Tables 7 and 8).

Each model is compared under the protocol the paper reports:
  p-IgGen      gated       (benchmark/final_latents.csv, sets f1, baseline, output)
  IgLM         ungated     (same file, sets f1 ungated, baseline ungated, output ungated)
  ProGen2-OAS  temperature 1.0 with the human FR1 prompt (progen2_t10/score/latents.csv)
Sets, three latents per SAE and gene: f1 = highest-F1 ranks 1-3 (positions/<model>_positions.csv), baseline = draws 1-3
(positions/<model>_controls*.csv), output = output-selected ranks 1-3 (plm_sae.steering.pick_output on
probe/scores_<model>.csv). Files are read from <work_dir> where present, else <results_dir>."""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from plm_sae.steering.pick_output import picks
from paper.benchmark._inputs import inp

M = ["piggen", "IgLM", "progen2_oas"]
K = 3
KEY = ["model", "arch", "target", "latent"]
ARCH = ["TopK", "Batch TopK", "Ordered"]
PROTOCOL = {"piggen": "gated", "IgLM": "ungated", "progen2_oas": "temperature 1.0, FR1 prompt"}


def ok(s):
    return s.astype(str).isin(["pass", "True"])


def rank_tables():
    """(f1 ranks, baseline draws, output ranks) as DataFrames with KEY + rank."""
    D = []
    for m in M:
        c1 = pd.read_csv(inp("benchmark", "positions", f"{m}_controls.csv")).assign(rank=1)
        c2 = pd.read_csv(inp("benchmark", "positions", f"{m}_controls2.csv"))
        c2["rank"] = c2.groupby(["arch", "gene"]).cumcount() + 2
        D += [dict(model=m, arch=r.arch, target=r.gene, latent=int(r.latent), rank=int(r.rank))
              for c in (c1, c2) for r in c.itertuples()]
    pos = pd.concat([pd.read_csv(inp("benchmark", "positions", f"{m}_positions.csv")).assign(model=m) for m in M])
    pos = pos[pos["rank"] >= 1][["model", "arch", "gene", "latent", "rank"]].rename(columns={"gene": "target"})
    out = pd.concat([picks(pd.read_csv(inp("benchmark", "probe", f"scores_{m}.csv"))) for m in M])
    out = out[["model", "arch", "gene", "latent", "rank"]].rename(columns={"gene": "target"})
    return pos, pd.DataFrame(D), out


def existing_sets(FL=None, T10L=None):
    """Rows (model, arch, target, latent, rank, set, task, paper_corr, paper_mag) of the three sets per model under its
    reported protocol, ranks 1-3; set in {f1, baseline, output}."""
    POS, RD, OR = rank_tables()
    FL = pd.read_csv(inp("benchmark", "final_latents.csv")) if FL is None else FL
    cols = KEY + ["rank", "set", "task", "paper_corr", "paper_mag"]
    parts = []
    for m, un in (("piggen", ""), ("IgLM", " ungated")):
        x = FL[FL.model == m].drop(columns=["rank"])
        for s, R in (("f1", POS), ("baseline", RD), ("output", OR)):
            parts.append(x[x.set == s + un].merge(R, on=KEY).assign(set=s))
    t = pd.read_csv(inp("progen2_t10", "score", "latents.csv")) if T10L is None else T10L
    parts.append(t.assign(model="progen2_oas"))
    out = pd.concat(parts, ignore_index=True)
    return out[out["rank"].between(1, K)].drop_duplicates(KEY + ["set"]).reset_index(drop=True)[cols]


def strat_exact(a, b):
    """a, b: Series pair -> list of 0/1 (same pairs). Two-sided exact conditional test stratified by pair (sum of the
    probabilities <= that of the observed count)."""
    pmf, xobs = np.array([1.0]), 0
    for p in a.index:
        x, y = np.asarray(a[p]), np.asarray(b[p])
        n, Kp, d = len(x) + len(y), int(x.sum() + y.sum()), len(x)
        xobs += int(x.sum())
        pmf = np.convolve(pmf, hypergeom.pmf(np.arange(d + 1), n, Kp, d))
    return float(min(1.0, pmf[pmf <= pmf[xobs] * (1 + 1e-7)].sum()))


def summarise(T, sets, col, pair_cols, ref):
    """k of 3n per set and the stratified exact test of every set against ref."""
    T = T[T.set.isin(sets) & (T["rank"] <= K)].copy()
    T["pair"] = T[pair_cols].astype(str).agg("|".join, axis=1)
    T["pass"] = ok(T[col]).astype(int)
    pairs = sorted(set.union(*[set(T[T.set == s].pair) for s in sets]))
    out, G = {"n_pairs": len(pairs), "n": K * len(pairs), "complete": True}, {}
    for s in sets:
        x = T[T.set == s].drop_duplicates(["pair", "rank"])
        full = x.groupby("pair")["rank"].nunique().reindex(pairs).fillna(0)
        if (full < K).any():
            out["complete"] = False
            out.setdefault("missing", {})[s] = int((K - full[full < K]).sum())
        G[s] = x.groupby("pair")["pass"].apply(list).reindex(pairs).apply(lambda v: v if isinstance(v, list) else [])
        out[s] = {"k": int(x["pass"].sum())}
    for s in sets:
        if s != ref and out["complete"]:
            out[s]["p_vs_" + ref] = strat_exact(G[s], G[ref])
    return out
