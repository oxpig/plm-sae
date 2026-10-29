"""Table 3 and Supplementary Table 10 rows, and the library medians quoted in Sections 4.2.3 and 4.3, from the steered
generation analysis (paper.analyses.quality): quality/per_latent.csv, quality/per_library.csv and
quality/ablang/scores/natural_{paired,heavy}.npy (<work_dir> where present, else <results_dir>).

Table 3: Batch TopK latent 733 (IGHJ6, highest F1, alpha* 48); Supplementary Table 10: TopK latent 5141 (IGHJ6,
output-selected, alpha* 12), both p-IgGen. Rows: VH and VL Hamming distance to the test sequences, diversity, ESM-2 and
IgLM log-likelihood, AbLang-2 pseudo-log-likelihood. Cells: steered mean, natural mean, difference, two-sided
Mann-Whitney p ("<0.001" below 0.001), |rank-biserial r|. Also prints the numbered count of each library, the unsteered
p-IgGen IgLM and AbLang-2 means and, per model, the medians over the steered libraries of the VH Hamming distance and
the AbLang-2 PLL with the range over metrics of the median |r|.
Writes <work_dir>/tables/table3_rows.tex and t3_output_rows.tex. Usage: python -m paper.tables.table3_rows"""
import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

NM = [("VH_hamming_to_test", "VH Hamming", 2), ("VL_hamming_to_test", "VL Hamming", 2), ("diversity", "Diversity", 3),
      ("ESM2_loglikelihood", "ESM-2 log-lik.", 3), ("IgLM_loglikelihood", "IgLM log-lik.", 3)]
LIBS = [("table3_rows.tex", "piggen_BatchTopK_f1_IGHJ6_f733_a48"),
        ("t3_output_rows.tex", "piggen_TopK_output_IGHJ6_f5141_a12")]
SETS = ["f1", "baseline", "output"]


def cell_row(name, ms, mn, p, r, d):
    f = lambda v: f"${v:.{d}f}$" if v < 0 else f"{v:.{d}f}"
    pp = "$<0.001$" if p < 0.001 else f"{p:.3f}"
    return f"{name} & {f(ms)} & {f(mn)} & ${ms - mn:+.{d}f}$ & {pp} & {abs(r):.2f} \\\\"


def main():
    PL = pd.read_csv(inp("quality", "per_latent.csv"))
    lib = pd.read_csv(inp("quality", "per_library.csv"))
    nat = {k: np.load(inp("quality", "ablang", "scores", f"natural_{k}.npy")) for k in ("paired", "heavy")}
    count = PL[PL.metric == "ANARCI_numbered_count"].set_index("tag")   # numbered (mean_steered) of the library (mean_ref)
    PL = PL[PL.ref == "natural"]
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    for fn, tag in LIBS:
        t = PL[PL.tag == tag].set_index("metric")
        rows = [cell_row(name, t.loc[k].mean_steered, t.loc[k].mean_ref, t.loc[k].p, t.loc[k].rank_biserial, d) for k, name, d in NM]
        x = lib[lib.tag == tag].iloc[0]
        rows.append(cell_row("AbLang-2 PLL", x["mean"], np.nanmean(nat["paired"]), x.p, x.r, 3))
        (o / fn).write_text("\n".join(rows) + "\n")
        print(f"% {tag}: both chains numbered in {int(count.mean_steered[tag])} of the library's {int(count.mean_ref[tag])} "
              f"sequences; AbLang-2 n {x.n:.0f}; "
              f"max p {max(t.loc[[k for k, _, _ in NM]].p.max(), x.p):.3g}; "
              f"max |r| {max(t.loc[[k for k, _, _ in NM]].rank_biserial.abs().max(), abs(x.r)):.2f}")
        print("\n".join(rows))
    u = PL[(PL.tag == "piggen_unsteered_a0") & (PL.metric == "IgLM_loglikelihood")].mean_steered.iloc[0]
    print(f"% unsteered p-IgGen (alpha 0): IgLM {u:.3f}; AbLang-2 {lib[lib.tag == 'piggen_unsteered_a0']['mean'].iloc[0]:.3f}")
    for m in ("piggen", "IgLM", "progen2_oas"):
        dd = lib[(lib.model == m) & lib.set.isin(SETS)].dropna(subset=["mean"])
        p = PL[(PL.model == m) & PL.set.isin(SETS)]
        u = lib[(lib.model == m) & (lib.set == "unsteered")]
        pu = PL[(PL.model == m) & (PL.set == "unsteered")]
        vh = p[p.metric == "VH_hamming_to_test"]
        rr = [p[p.metric == k].rank_biserial.abs().median() for k, _, _ in NM if (p.metric == k).any()] + [dd.r.abs().median()]
        print(f"% {m}: {len(dd)} AbLang-2-scored libraries ({vh.tag.nunique()} with metrics); median VH Hamming "
              f"{vh.mean_steered.median():.4f} (natural {vh.mean_ref.iloc[0]:.4f}; unsteered "
              f"{pu[pu.metric == 'VH_hamming_to_test'].mean_steered.iloc[0]:.4f}); median AbLang-2 {dd['mean'].median():.4f} "
              f"(natural {np.nanmean(nat['paired' if m == 'piggen' else 'heavy']):.4f}; unsteered {u['mean'].iloc[0]:.4f}); "
              f"median |r| over metrics {min(rr):.3f} to {max(rr):.3f}")


if __name__ == "__main__":
    main()
