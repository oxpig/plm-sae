"""J-gene steering figures (Figures 2 and 3, Supplementary Figure 7), each panel read from the library table of the run the paper reports
(p-IgGen gated and IgLM without the position gate: benchmark; ProGen2-OAS at temperature 1.0 with the human FR1 prompt:
progen2_t10). Proportion of each heavy J gene among numbered human heavy chains with at least 50% J germline identity,
mean of three seeds of 1,000 sequences (line) and range (band), from alpha = 0 to alpha* (median over seeds, from the
latent table of the same run). Prints, per panel, the target-gene proportion at alpha = 0 and alpha* and the per-seed
alpha* mean of Supplementary Table 7.

    python -m paper.figures.fig_steering OUT.png WIDTH_IN HEIGHT_IN NCOL "TITLE|RUN|MODEL|RUNDIR|LATENT|GENE" ...

RUN is benchmark (benchmark/final_libraries.csv.gz, final_latents.csv) or progen2_t10 (progen2_t10/score/libraries.csv.gz,
score/latents.csv), from <work_dir> where present, else <results_dir>; RUNDIR is the library directory below
runs/<MODEL>/ (e.g. BatchTopK/f1/gated). paper/scripts/figures/make_steering_figures.sh draws the three figures."""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper.benchmark._inputs import inp
from paper.benchmark.analyze_final import rel_run_dir

RUNS = {"benchmark": ("benchmark", "final_libraries.csv.gz", "final_latents.csv"), "progen2_t10": ("progen2_t10", "score/libraries.csv.gz", "score/latents.csv")}
GENES = ["IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]   # tab10
STYLE = {"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 7,
         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
         "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
         "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.titlepad": 3, "axes.labelpad": 2}
_cache = {}


def tables(run):
    if run not in _cache:
        stage, lib, lat = RUNS[run]
        # germline-filtered J calls (IGHJx_g) over germline-filtered heavy chains (n_Hj), as in Methods 3.3.1 and Supp. Table 7
        L = pd.read_csv(inp(stage, lib), usecols=["dir", "latent", "target", "alpha", "seed", "n_Hj"] + [g + "_g" for g in GENES])
        L["dir"] = L["dir"].map(rel_run_dir)   # library directories relative to the stage root
        _cache[run] = (L.rename(columns={g + "_g": g for g in GENES}), pd.read_csv(inp(stage, lat)))
    return _cache[run]


def load(run, model, rundir, lat, gene):
    LIB, LAT = tables(run); d = f"runs/{model}/{rundir}/{gene}"
    x = LIB[(LIB.dir == d) & (LIB.latent == lat) & (LIB.target == gene)].copy()
    if x.empty: raise SystemExit(f"no library rows for {d} latent {lat}")
    for g in GENES: x[g] = 100 * x[g] / x.n_Hj.where(x.n_Hj > 0)
    s = x.groupby("alpha")[GENES].agg(["mean", "min", "max"])
    tasks = {t for t in LAT[(LAT.latent == lat) & (LAT.target == gene)].task}
    a = LAT[(LAT.latent == lat) & (LAT.target == gene) & LAT.task.isin(tasks)]
    a = a[a.arch.str.replace(" ", "") == rundir.split("/")[0]] if "arch" in a else a
    if "model" in a: a = a[a.model == model]
    p = rundir.split("/")
    if run == "benchmark":   # runs/<model>/<arch>/<set>/<rule>: keep rows of that rule and set
        a = a[a["rule"].astype(str) == p[-1]]
        if len(p) == 3 and (a["sub"].astype(str) == p[1]).any(): a = a[a["sub"].astype(str) == p[1]]
    else: a = a[a.set == p[1]]
    astar = a.paper_astar_med.unique(); assert len(astar) == 1, (rundir, lat, astar)
    # Supp. Table 7 convention for the text: each seed at its own alpha* (largest alpha with n_Hj >= 800), then the mean
    ach = []
    for _, xs in x.groupby("seed"):
        k = xs[xs.n_Hj >= 800].alpha.max(); ach.append(float(xs[xs.alpha == k][gene].iloc[0]))
    print(f"    {gene} per-seed alpha* mean (Supp. Table 7): {s[(gene, 'mean')].iloc[0]:.1f} -> {np.mean(ach):.1f}")
    return s[s.index <= astar[0] + 1e-9], float(astar[0])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    plt.rcParams.update(STYLE)
    out, W, H, ncol = argv[0], float(argv[1]), float(argv[2]), int(argv[3])
    specs = [x.split("|") for x in argv[4:]]; nrow = -(-len(specs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, H), squeeze=False, sharey=True)
    for k, (ax, (title, run, model, rundir, lat, gene)) in enumerate(zip(axes.flat, specs)):
        s, astar = load(run, model, rundir, int(lat), gene)
        for g, c in zip(GENES, COL):
            tgt = g == gene
            ax.fill_between(s.index, s[(g, "min")], s[(g, "max")], color=c, alpha=0.15, lw=0)
            ax.plot(s.index, s[(g, "mean")], color=c, lw=1.5 if tgt else 0.9, marker="o", ms=2.6 if tgt else 1.9, label=g, zorder=3 if tgt else 2)
        ax.set_title(title, fontsize=6 if len(title) > 26 else 7); ax.set_xlabel(r"Steering factor ($\alpha$)")
        if k % ncol == 0: ax.set_ylabel("Proportion (%)")
        ax.set_ylim(-2, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlim(-0.04 * astar, 1.04 * astar)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        t = s[(gene, "mean")]
        print(f"{title}: {gene} {t.iloc[0]:.1f} -> {t.iloc[-1]:.1f} at alpha* {astar:g}")
    for ax in list(axes.flat)[len(specs):]: ax.axis("off")
    # neutral legend: the target line is drawn thicker in each panel, so no panel's target is singled out
    h = [Line2D([], [], color=c, lw=1.2, marker="o", ms=2.2) for c in COL]; l = GENES
    fig.legend(h, l, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2, columnspacing=0.7, handletextpad=0.25, borderaxespad=0.1)
    fig.tight_layout(pad=0.25, rect=(0, 0, 1, 1 - 0.2 / H), w_pad=0.8, h_pad=0.6)
    fig.savefig(out, dpi=400); print("wrote", out)


if __name__ == "__main__":
    main()
