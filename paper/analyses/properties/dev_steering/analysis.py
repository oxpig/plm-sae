"""Steering the PSH-associated latents of p-IgGen (Supplementary Section A.10, "Steering associated latents"): TopK
latent 980 and Batch TopK latent 1088, steered in the benchmark (set developability), scored with FlashTAP
(paper/scripts/properties/dev_score.sh: <work_dir>/benchmark/dev_scores/f<latent>/).
  trend      per seed and seeds pooled, Spearman's rho of the median PSH against alpha over alpha 0 .. valid_amax (the
             last alpha with >= 80% numbered human heavy chains, final_latents.csv), the medians at both ends and the
             fewest scored sequences per library -> dev_trend.csv
  controls   baseline latents of p-IgGen TopK and Batch TopK steered from position 100 or later, as 980 and 1088
             (starts 109 and 110), each to be scored at its valid_amax, seeds 0-2 -> dev_ctrl_list.csv
             (scored by paper/scripts/properties/dev_ctrl.sh into dev_scores/ctrl_<latent>/)
  compare    shift of the median PSH at valid_amax from the median of the unsteered libraries (all alpha-0 sequences
             scored for 980 and 1088), for 980, 1088 and the baselines -> dev_ctrl_compare.csv
Inputs are read from <work_dir>/benchmark where present, else from the record; outputs go to <work_dir>/benchmark.
Usage: python -m paper.analyses.properties.dev_steering.analysis trend|controls|compare"""
import sys

import pandas as pd
from scipy import stats

from plm_sae import paths as P
from paper.benchmark._inputs import inp

F = P.work("benchmark")
F.mkdir(parents=True, exist_ok=True)
B = lambda *p: inp("benchmark", *p)
DEV = {980: "TopK", 1088: "Batch TopK"}


def valid_amax(R, latent):
    return R[(R.target == "dev") & (R.latent == latent)].valid_amax.iloc[0]


def seeds_of(feat, S, pairs):
    """Seed of each scored pair: the seed file whose line idx starts with the pair's VH."""
    W, raw = B("dev_scores", f"f{feat}"), {}
    for f in W.glob(f"steered_l3_f{feat}_a*_seed*_seq1000.txt"):
        raw[(float(f.name.split("_a")[1].split("_")[0]), int(f.name.split("seed")[1].split("_")[0]))] = \
            [l.strip() for l in open(f) if l.strip()]
    out = []
    for a, i, vh in zip(pairs.alpha, pairs.idx, pairs.VH):
        hit = [s for s in (0, 1, 2) if i < len(raw.get((a, s), [])) and raw[(a, s)][i].startswith(vh)]
        out.append(hit[0] if len(hit) == 1 else -1)
    return out


def trend():
    R, rows = pd.read_csv(B("final_latents.csv")), []
    for feat in DEV:
        W = B("dev_scores", f"f{feat}")
        S, pairs = pd.read_csv(W / f"scores_f{feat}.csv"), pd.read_csv(W / "pairs.csv")
        assert len(S) == len(pairs) and (S.idx.values == pairs.idx.values).all()
        S["seed"] = seeds_of(feat, S, pairs)
        S = S[S.alpha <= valid_amax(R, feat) + 1e-9]
        for sd in (0, 1, 2, "pooled"):
            x = S if sd == "pooled" else S[S.seed == sd]
            med, n = x.groupby("alpha").PSH.median(), x.groupby("alpha").size()
            q = stats.spearmanr(med.index, med.values)
            rows.append(dict(latent=feat, seed=sd, n_alpha=len(med), rho=round(q.statistic, 3), p=q.pvalue,
                             median_alpha0=round(med.iloc[0], 2), median_last=round(med.iloc[-1], 2),
                             alpha_last=med.index.max(), min_scored=int(n.min())))
    D = pd.DataFrame(rows)
    D.to_csv(F / "dev_trend.csv", index=False)
    print(D.to_string(index=False))


def controls():
    R = pd.read_csv(B("final_latents.csv"))
    C = R[(R.model == "piggen") & (R.set == "baseline") & R.arch.isin(["TopK", "Batch TopK"]) & (R.start >= 100)]
    rows = [dict(arch=e.arch.replace(" ", ""), target=e.target, latent=e.latent, start=e.start, alpha=e.valid_amax,
                 dir=str(F / f"runs/piggen/{e.arch.replace(' ', '')}/baseline/gated/{e.target}")) for e in C.itertuples()]
    pd.DataFrame(rows).to_csv(F / "dev_ctrl_list.csv", index=False)
    print(len(rows), "baselines")


def compare():
    R = pd.read_csv(B("final_latents.csv"))
    S = {f: pd.read_csv(B("dev_scores", f"f{f}", f"scores_f{f}.csv")) for f in DEV}
    b0 = pd.concat([S[f][S[f].alpha == 0].PSH for f in S]).median()
    rows = []
    for f, x in S.items():
        va = valid_amax(R, f)
        rows.append(dict(kind="developability", latent=f, alpha=va, shift=x[x.alpha == va].PSH.median() - x[x.alpha == 0].PSH.median()))
    for e in pd.read_csv(B("dev_ctrl_list.csv")).itertuples():
        x = pd.read_csv(B("dev_scores", f"ctrl_{e.latent}", f"scores_f{e.latent}.csv"))
        rows.append(dict(kind="baseline", latent=e.latent, start=e.start, alpha=e.alpha, shift=x.PSH.median() - b0))
    D = pd.DataFrame(rows)
    D.to_csv(F / "dev_ctrl_compare.csv", index=False)
    C, sh = D[D.kind == "baseline"], D[(D.kind == "developability") & (D.latent == 980)]["shift"].iloc[0]
    print(D.round(2).to_string(index=False))
    print(f"baselines (starts {C.start.min()}-{C.start.max()}): shift {C['shift'].min():.1f} to {C['shift'].max():.1f}; "
          f"{(C['shift'] <= sh).sum()} of {len(C)} at or below latent 980's {sh:.1f}")


if __name__ == "__main__":
    {"trend": trend, "controls": controls, "compare": compare}[sys.argv[1]]()
