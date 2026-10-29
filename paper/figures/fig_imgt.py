"""IMGT activation figures (Supplementary Figures 8 and 9): bars over IMGT positions, the most active position
highlighted and named in a legend, "Activation positions for latent N" titles. Quantity: percentage of the latent's
total activation at each IMGT position over validation sequences of the gene (the quantity of the localisation table).
Data: identification/imgt/mass_<sae>.npz (p-IgGen; paper.analyses.identification.localisation) and
identification/imgt/imgt_position_counts.json (IgLM, ProGen2-OAS; paper.analyses.identification.imgt_position_counts),
from <work_dir> where present, else <results_dir>.
Usage: python -m paper.figures.fig_imgt OUTDIR
Outputs: OUTDIR/fig_imgt_ordered_topk.png (Supplementary Figure 9), OUTDIR/fig_imgt_xmodel.png (Supplementary
Figure 8); prints the top position and its share for every panel."""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.analyses.identification.imgt import CANONICAL_NUMBERING
from paper.benchmark._inputs import inp

LAB = [l.strip() for l in CANONICAL_NUMBERING]
TICKS = ["1", "14", "28", "37", "47", "56", "61", "69B", "79", "85A", "95", "108", "111K", "112", "125"]
TIX = [LAB.index(t) for t in TICKS if t in LAB]
TIX3 = [LAB.index(t) for t in ["1", "28", "47", "61", "79", "95", "111K", "125"]]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6})


def panel(ax, mass, lid, ylab, ticks=TIX):
    pct = 100 * np.asarray(mass, float) / np.sum(mass); t = int(np.argmax(pct))
    ax.bar(range(len(pct)), pct, width=1.0, color="skyblue", lw=0)
    ax.bar([t], [pct[t]], width=1.6, color="#d62728", lw=0, label=f"Position {LAB[t]}")
    ax.set_xticks(ticks); ax.set_xticklabels([LAB[i] for i in ticks], rotation=0)
    ax.set_xlim(-1, len(pct)); ax.set_ylim(0, 105)
    ax.set_title(f"Activation positions for latent {lid}", pad=3); ax.set_xlabel("IMGT position", labelpad=1)
    if ylab: ax.set_ylabel("Percentage of activation (%)", labelpad=1)
    ax.legend(loc="upper left", frameon=True, handlelength=1.0, borderpad=0.3)
    ax.tick_params(length=2, pad=1.5)
    return LAB[t], pct[t]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    # Ordered vs TopK, highest-F1 p-IgGen latent of IGHJ2, IGHJ4, IGHJ5 (rows); (a) Ordered left, (b) TopK right
    mass = {a: np.load(inp("identification", "imgt", f"mass_{sae}.npz"))
            for a, sae in (("Ordered", "piggen_ordered_r8"), ("TopK", "piggen_topk_r8"))}
    fig = plt.figure(figsize=(6.9, 4.8), layout="constrained")
    for c, (sub, arch, title) in enumerate(zip(fig.subfigures(1, 2, wspace=0.03), ("Ordered", "TopK"),
                                               (r"(a) Ordered, top $F_1$", r"(b) TopK, top $F_1$"))):
        sub.suptitle(title, fontsize=8); axes = sub.subplots(3, 1)
        for r, (gene, o, t) in enumerate([("IGHJ2", 2979, 902), ("IGHJ4", 4900, 5815), ("IGHJ5", 5575, 3938)]):
            lid = o if arch == "Ordered" else t
            print(gene, arch, lid, panel(axes[r], mass[arch][f"{gene}_{lid}"], lid, c == 0))
    fig.savefig(f"{out}/fig_imgt_ordered_topk.png", dpi=400); plt.close(fig)
    # (a) IgLM (top) and (b) ProGen2-OAS (bottom), three highest-F1 TopK IGHJ4 latents
    d = json.load(open(inp("identification", "imgt", "imgt_position_counts.json")))
    fig = plt.figure(figsize=(6.9, 3.4), layout="constrained")
    for sub, (model, title, lats) in zip(fig.subfigures(2, 1, hspace=0.04),
                                         [("IgLM", "(a) IgLM", [3036, 1565, 2705]),
                                          ("ProGen2-OAS", "(b) ProGen2-OAS", [8218, 2303, 9879])]):
        sub.suptitle(title, fontsize=8); axes = sub.subplots(1, 3)
        for c, lid in enumerate(lats):
            print(model, lid, panel(axes[c], d[f"{model}|{lid}"]["mass"], lid, c == 0, TIX3))
    fig.savefig(f"{out}/fig_imgt_xmodel.png", dpi=400); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
