"""Steering with and without the position gate (Supplementary Table 9, tab:gate) and the numbers of its paragraph:
latents passing the correlation criterion of 54 (three latents per SAE-gene pair: highest-F1 ranks 1-3, baseline draws
1-3, output-selected ranks 1-3), the same latents, strengths and seeds with and without the gate.
Per model and set: passes gated / ungated, exact two-sided McNemar p over the latents run both ways, and the median
total variation distance of the FR1-FR3 amino-acid composition from alpha = 0 at alpha* (vtv_valid_amax).
Reads benchmark/final_latents.csv and the rank tables of paper.benchmark.rsets; prints the table body and the numbers
and writes <work_dir>/tables/gate.tex. Usage: python -m paper.tables.gate"""
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import rsets
from paper.benchmark._inputs import inp
from paper.tables.common import MN, mcnemar, ok, pfmt

KEY = rsets.KEY
ROWS = [("Highest $F_1$", "f1"), ("Baseline", "baseline"), ("Output-selected", "output")]


def main():
    R = pd.read_csv(inp("benchmark", "final_latents.csv"))
    POS, RD, OR = rsets.rank_tables()
    RK = {"f1": POS, "baseline": RD, "output": OR}

    def pick(m, s, rk):
        x = R[(R.model == m) & (R.set == s)].drop(columns=["rank"]).merge(RK[rk], on=KEY)
        return x[x["rank"] <= 3].drop_duplicates(KEY)

    body = []
    for m in ("piggen", "IgLM"):
        if body:
            body.append(r"\midrule")
        first = True
        for label, s in ROWS:
            g, u = pick(m, s, s), pick(m, s + " ungated", s)
            if not len(g) or not len(u):   # p-IgGen output-selected latents were run with the gate only
                continue
            gi, ui = g.set_index(KEY), u.set_index(KEY)
            common = sorted(set(gi.index) & set(ui.index))
            p = mcnemar(ok(gi.loc[common].paper_corr), ok(ui.loc[common].paper_corr))
            body.append(f"{MN[m] if first else ''} & {label} & {ok(g.paper_corr).sum()} & {ok(u.paper_corr).sum()} \\\\")
            print(f"% {m} {label}: gated {ok(g.paper_corr).sum()}/{len(g)}, ungated {ok(u.paper_corr).sum()}/{len(u)}, "
                  f"McNemar p {pfmt(p)} over {len(common)} latents; median TV gated {g.vtv_valid_amax.median():.3f}, "
                  f"ungated {u.vtv_valid_amax.median():.3f}")
            first = False
    body = "\n".join(body)
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    (o / "gate.tex").write_text(body + "\n")
    print(body)


if __name__ == "__main__":
    main()
