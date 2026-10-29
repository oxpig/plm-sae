"""Generation quality of steered libraries by set (Supplementary Section A.4 "Generation quality", Supplementary Table 11,
Supplementary Figure 4), from the per-library results of the steered generation analysis: <results_dir>/quality/
{per_latent.csv, per_library.csv, ablang/scores/natural_{paired,heavy}.npy}. Libraries: seed 0 at alpha* of every latent
passing the correlation criterion (p-IgGen gated, paired VH+VL against natural pairs; IgLM without the position gate and
ProGen2-OAS at temperature 1.0, VH against natural VH).
  check   rebuilds every cell of Supplementary Table 11 per SAE and set and compares it with the rows of
          \\label{tab:t3_batch} of a LaTeX file (--tex); exits non-zero on any mismatch
  pool    <out>/per_library_long.csv (one row per library and metric, the data of Supplementary Figure 4) and
          <out>/quality_by_set.csv (per model and set, pooled over the three SAEs: median library mean and median |r|)
Usage: python -m paper.analyses.quality.by_set pool [--out DIR]          (default <work_dir>/quality/by_set)
       python -m paper.analyses.quality.by_set check --tex manuscript.tex"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from plm_sae import paths as P
from paper.benchmark._inputs import inp

MET = ["VH_hamming_to_test", "VL_hamming_to_test", "diversity", "ESM2_loglikelihood", "IgLM_loglikelihood"]
DG = {"VH_hamming_to_test": 2, "VL_hamming_to_test": 2, "diversity": 3, "ESM2_loglikelihood": 3, "IgLM_loglikelihood": 3, "ablang": 3}
LABEL = {"f1": "Highest F1", "baseline": "Baseline", "output": "Output-selected"}
SETS = {m: LABEL for m in ("piggen", "IgLM", "progen2_oas")}
MN = {"piggen": "p-IgGen", "IgLM": "IgLM", "progen2_oas": "ProGen2-OAS"}


def load():
    Q = inp("quality", "per_latent.csv").parent
    PL = pd.read_csv(Q / "per_latent.csv")
    PL = PL[(PL.ref == "natural") & PL.metric.isin(MET)].copy()
    D = pd.read_csv(Q / "per_library.csv")
    nat = {k: np.load(Q / "ablang" / "scores" / f"natural_{k}.npy") for k in ("paired", "heavy")}
    return PL, D, nat


def per_sae_rows(PL, D, nat):
    """Supplementary Table 11 per SAE and set (median over latents of library mean and of |r|; p < 0.05 share)."""
    T = []
    for m, sets in SETS.items():
        pn = PL[PL.model == m]; natv = pn.groupby("metric").mean_ref.first()
        T.append(dict(model=m, set="Natural", arch="", n=np.nan, **{f"{k}_mean": natv.get(k, np.nan) for k in MET},
                      ablang_mean=np.nanmean(nat["paired" if m == "piggen" else "heavy"])))
        for s in ["unsteered"] + list(sets):
            for a in (["none"] if s == "unsteered" else ["TopK", "Batch TopK", "Ordered"]):
                d = D[(D.model == m) & (D.set == s) & ((D.arch == a) | (s == "unsteered"))].dropna(subset=["mean"])
                p = PL[(PL.model == m) & (PL.set == s) & ((PL.arch == a) | (s == "unsteered"))]
                if not len(d): continue
                ex = {}
                for k in MET:
                    z = p[p.metric == k]
                    if len(z): ex[f"{k}_mean"] = z.mean_steered.median(); ex[f"{k}_r"] = z.rank_biserial.abs().median()
                k_ = int((p.p < 0.05).sum()) + int((d.p < 0.05).sum()); n_ = len(p) + len(d)
                T.append(dict(model=m, set="Unsteered" if s == "unsteered" else sets[s], arch="" if s == "unsteered" else a,
                              n=len(d), **ex, ablang_mean=d["mean"].median(), ablang_r=d.r.abs().median(), share=f"{k_}/{n_}"))
    return pd.DataFrame(T)


def long_table(PL, D):
    a = PL[["model", "arch", "set", "latent", "target", "alpha", "lib", "metric", "mean_steered", "p", "rank_biserial", "n_steered"]].copy()
    a["tag"] = a.lib.str.extract(r"steered_l3_(f\d+_a[\d.]+)_seed0")[0]
    a = a.rename(columns={"mean_steered": "mean", "rank_biserial": "r", "n_steered": "n_seq"})
    b = D.dropna(subset=["mean"])[["model", "arch", "set", "latent", "alpha", "tag", "mean", "p", "r", "n"]].rename(columns={"n": "n_seq"}).assign(metric="ablang")
    X = pd.concat([a, b], ignore_index=True)
    X = X[[(s_ == "unsteered") or (s_ in SETS[m]) for m, s_ in zip(X.model, X.set)]].copy()
    X["set_label"] = [("Unsteered" if s == "unsteered" else SETS[m][s]) for m, s in zip(X.model, X.set)]
    return X


def pooled(X, PL, nat):
    """Per model and set, pooled over the three SAEs; 'All passing' = the three sets together."""
    rows = []
    for m in SETS:
        x = X[X.model == m]; natv = PL[PL.model == m].groupby("metric").mean_ref.first()
        r0 = dict(model=MN[m], set="Natural", n=np.nan)
        for k in MET: r0[f"{k}_mean"] = natv[k] if (k in natv and not (m != "piggen" and k == "VL_hamming_to_test")) else np.nan
        r0["ablang_mean"] = np.nanmean(nat["paired" if m == "piggen" else "heavy"]); rows.append(r0)
        for s in ["Unsteered", "Highest F1", "Baseline", "Output-selected", "All passing"]:
            y = x[x.set_label == s] if s != "All passing" else x[x.set_label != "Unsteered"]
            r = dict(model=MN[m], set=s, n=int((y.metric == "ablang").sum())); k_sig = n_cmp = 0
            for k in MET + ["ablang"]:
                z = y[y.metric == k]
                if not len(z): continue
                r[f"{k}_mean"] = z["mean"].median(); r[f"{k}_r"] = z.r.abs().median(); r[f"{k}_n"] = len(z)
                k_sig += int((z.p < 0.05).sum()); n_cmp += len(z)
            r["share"] = f"{k_sig}/{n_cmp}"; rows.append(r)
    T = pd.DataFrame(rows)
    for m in SETS:   # heavy-chain models have no VL column
        if m != "piggen": T.loc[T.model == MN[m], [c for c in T.columns if c.startswith("VL_")]] = np.nan
    return T


def _fmt(v, dg):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return f"${v:.{dg}f}$" if v < 0 else f"{v:.{dg}f}"


def check(tex):
    txt = Path(tex).read_text(); i = txt.index("\\label{tab:t3_batch}"); j = txt.index("\\bottomrule", i)
    body = txt[txt.index("\\midrule", i) + len("\\midrule"):j]
    T = per_sae_rows(*load()); rev = {v: k for k, v in MN.items()}; model = setl = None; n = bad = 0
    for line in body.split("\\\\"):
        line = re.sub(r"^\\midrule", "", line.strip()).strip()
        if not line: continue
        c = [x.strip() for x in line.split("&")]
        if c[0]: model = c[0]
        sae = c[1]
        if sae in ("Natural", "Unsteered"): setl = sae
        elif c[2]: setl = c[2].replace("Highest $F_1$", "Highest F1")
        m = rev[model]
        t = T[(T.model == m) & (T.set == setl) & ((T.arch == sae) | (sae in ("Natural", "Unsteered")))].iloc[0]
        mine = ["" if sae == "Natural" else str(int(t.n))]
        for k in MET + ["ablang"]:
            if m != "piggen" and k == "VL_hamming_to_test": mine += ["--", "" if sae == "Natural" else "--"]; continue
            mine += [_fmt(t[f"{k}_mean"], DG[k]), "" if sae == "Natural" else f"{t[f'{k}_r']:.2f}"]
        mine.append("" if sae == "Natural" else t.share)
        for a, b in zip(c[3:], mine):
            n += 1
            if a.replace(" ", "") != b.replace(" ", ""): bad += 1; print("MISMATCH", model, sae, setl, repr(a), repr(b))
    print(f"Supplementary Table 11: {n} cells compared, {bad} mismatches"); return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("mode", choices=["pool", "check"])
    ap.add_argument("--out", default=None); ap.add_argument("--tex", default=None); a = ap.parse_args(argv)
    if a.mode == "check":
        if not a.tex: ap.error("check needs --tex")
        sys.exit(1 if check(a.tex) else 0)
    PL, D, nat = load(); X = long_table(PL, D); T = pooled(X, PL, nat)
    o = Path(a.out) if a.out else P.work("quality", "by_set"); o.mkdir(parents=True, exist_ok=True)
    X.to_csv(o / "per_library_long.csv", index=False); T.to_csv(o / "quality_by_set.csv", index=False)
    print(T[["model", "set", "n", "VH_hamming_to_test_mean", "ablang_mean", "share"]].to_string(index=False)); print("wrote", o)


if __name__ == "__main__":
    main()
