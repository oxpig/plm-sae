"""Supplementary Figure 4: generation quality of the libraries of Supplementary Table 11, by set. One violin per set
(highest F1, baseline, output-selected) of the library means at alpha* (seed 0) of the latents passing the correlation
criterion, pooled over the three SAEs; white dots medians, bars interquartile ranges; solid line natural sequences,
dashed line the unsteered library. Rows: models; columns: metrics (VL Hamming only for p-IgGen). Checks that each
set's median equals quality_by_set.csv.

Reads <in>/per_library_long.csv and <in>/quality_by_set.csv (python -m paper.analyses.quality.by_set pool; default
<work_dir>/quality/by_set). Usage: python -m paper.figures.fig_quality_violins OUT.png [--in DIR]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plm_sae import paths as P

MODELS = [("piggen", "p-IgGen"), ("IgLM", "IgLM"), ("progen2_oas", "ProGen2-OAS")]
SETS = ["Highest F1", "Baseline", "Output-selected"]
TICK = {"Highest F1": "F$_1$", "Baseline": "Base", "Output-selected": "Out"}
COL = {"Highest F1": "#0072B2", "Baseline": "#8C8C8C", "Output-selected": "#D55E00"}
MET = [("VH_hamming_to_test", "VH Hamming to test"), ("VL_hamming_to_test", "VL Hamming to test"), ("diversity", "Diversity"),
       ("ESM2_loglikelihood", "ESM-2 log-lik."), ("IgLM_loglikelihood", "IgLM log-lik."), ("ablang", "AbLang-2 PLL")]
STYLE = {"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 0,
         "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False}


def figure(X, T, out):
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(3, 6, figsize=(7.0, 4.3))
    for i, (mk, mname) in enumerate(MODELS):
        t = T[T.model == mname].set_index("set"); x = X[(X.model == mk) & (X.set != "unsteered")]
        for j, (m, lab) in enumerate(MET):
            ax = axs[i, j]
            data = [x[(x.metric == m) & (x.set_label == s)]["mean"].dropna().values for s in SETS]
            if not all(len(d) for d in data): ax.axis("off"); continue
            for s, d in zip(SETS, data):
                assert abs(np.median(d) - t.loc[s, f"{m}_mean"]) < 1e-3, (mname, m, s)
            parts = ax.violinplot(data, positions=range(3), widths=0.8, showextrema=False)
            for b, s in zip(parts["bodies"], SETS): b.set_facecolor(COL[s]); b.set_edgecolor("none"); b.set_alpha(0.55)
            for p, d in enumerate(data):
                q1, q2, q3 = np.percentile(d, [25, 50, 75])
                ax.plot([p, p], [q1, q3], color="black", lw=1.3, solid_capstyle="butt")
                ax.scatter([p], [q2], s=7, color="white", edgecolor="black", lw=0.5, zorder=3)
            ax.axhline(t.loc["Natural", f"{m}_mean"], color="black", lw=0.7)
            ax.axhline(t.loc["Unsteered", f"{m}_mean"], color="black", lw=0.7, ls=(0, (2.5, 1.5)))
            ax.set_xticks(range(3)); ax.set_xticklabels([TICK[s] for s in SETS])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            if i == 0 or (j == 1): ax.set_title(lab, pad=3)
            if j == 0: ax.set_ylabel(mname, fontsize=7.5, fontweight="bold")
    h = [Line2D([], [], color="black", lw=0.7), Line2D([], [], color="black", lw=0.7, ls=(0, (2.5, 1.5)))]
    fig.legend(h, ["Natural sequences", "Unsteered library"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995),
               handlelength=2.0, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.05, top=0.92, wspace=0.62, hspace=0.38)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("out"); ap.add_argument("--in", dest="inp", default=None)
    a = ap.parse_args(argv); d = Path(a.inp) if a.inp else P.work("quality", "by_set")
    figure(pd.read_csv(d / "per_library_long.csv"), pd.read_csv(d / "quality_by_set.csv"), a.out)
    print("set medians match quality_by_set.csv; wrote", a.out)


if __name__ == "__main__":
    main()
