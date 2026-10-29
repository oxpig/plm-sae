"""Supplementary Table 19 (tab:dev_counts): latents associated with each FlashTAP developability metric in the nine
hyperparameter-matched SAEs, as a LaTeX table body. Each cell is the number of live latents with |Pearson r| >= 0.20
and Bonferroni-significant correlation with the metric over the validation sequences, with the number at |r| >= 0.40
in parentheses when it is not zero (paper.analyses.properties.developability.counts).
Reads developability/counts/dev_latent_counts.csv; writes <work_dir>/tables/dev_counts.tex and prints it.
Usage: python -m paper.tables.dev_counts"""
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

METRICS = ["PSH", "PNC", "PPC", "SFvCSP"]


def rows(C):
    out = []
    for i, (model, g) in enumerate(C.groupby("model", sort=False)):
        if i:
            out.append(r"\midrule")
        for j, r in enumerate(g.itertuples()):
            cells = []
            for m in METRICS:
                n20, n40 = int(getattr(r, f"{m}_020")), int(getattr(r, f"{m}_040"))
                cells.append(f"{n20} ({n40})" if n40 else f"{n20}")
            out.append(f"{model if j == 0 else ''} & {r.sae} & " + " & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def main():
    body = rows(pd.read_csv(inp("developability", "counts", "dev_latent_counts.csv")))
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    (o / "dev_counts.tex").write_text(body)
    print(body, end="")


if __name__ == "__main__":
    main()
