"""Supplementary Table 8 (tab:rl_depth): latents passing among the top one, two and three latents per heavy J gene, as
LaTeX. Sets of paper.benchmark.rsets.existing_sets (each model under its reported protocol: p-IgGen gated, IgLM without the
position gate, ProGen2-OAS at temperature 1.0): highest F1 (ranks 1-3), baseline (draws 1-3) and output-selected
(ranks 1-3). Cell: latents passing the correlation criterion among ranks / draws 1, 1-2 and 1-3 (both criteria in
parentheses). Prints the table and writes <work_dir>/tables/rl_depth.tex. Usage: python -m paper.tables.rl_depth"""

from plm_sae import paths as P
from paper.benchmark import rsets

SETS = ["f1", "baseline", "output"]


def main():
    T = rsets.existing_sets().copy()
    T["c"] = rsets.ok(T.paper_corr).astype(int)
    T["b"] = (rsets.ok(T.paper_corr) & rsets.ok(T.paper_mag)).astype(int)
    MN = {"piggen": "p-IgGen", "IgLM": "IgLM", "progen2_oas": "ProGen2-OAS"}
    L = [r"\begin{table*}[ht]", r"\centering\revcolor\footnotesize",
         r"\caption{\rev{Latents passing the correlation criterion among the top one, two and three latents per heavy J gene (of 6, 12 and 18; both criteria in parentheses). IgLM without the position gate, ProGen2-OAS at temperature 1.0.}}",
         r"\label{tab:rl_depth}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llccccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Highest $F_1$} & \multicolumn{3}{c}{Baseline} & \multicolumn{3}{c}{Output-selected} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Model & SAE & Top 1 & Top 2 & Top 3 & 1 & 2 & 3 & Top 1 & Top 2 & Top 3 \\"]
    for mi, m in enumerate(rsets.M):
        L.append(r"\midrule")
        for ai, a in enumerate(rsets.ARCH):
            cells = []
            for s in SETS:
                x = T[(T.model == m) & (T.arch == a) & (T.set == s)]
                cells += [f"{x[x['rank'] <= r].c.sum()} ({x[x['rank'] <= r].b.sum()})" for r in (1, 2, 3)]
            L.append(f"{MN[m] if ai == 0 else ''} & {a} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    (o / "rl_depth.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
