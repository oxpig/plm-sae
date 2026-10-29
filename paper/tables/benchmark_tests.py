"""Tests of the steering benchmark quoted in the text (Sections 4.2.3 and 4.3; Supplementary Section A.4): latents
passing the correlation criterion, k of 54 (three per SAE-gene pair), and the stratified exact test of each set against
the baseline latents (Mantel-Haenszel type, stratified by SAE-gene pair; paper.benchmark.rsets.strat_exact).
  benchmark   p-IgGen gated, IgLM ungated, ProGen2-OAS at temperature 1.0 (the sets of Table 2)
  windowed    p-IgGen and IgLM, steering inside the window at natural scale; a pair whose highest-F1 latent never fires
              on sequences of its gene has no window and counts as failing for every set
Reads benchmark (final_latents.csv, positions/, probe/, win/win_latents.csv) and progen2_t10/score/latents.csv
(<work_dir> where present, else <results_dir>). Usage: python -m paper.tables.benchmark_tests"""
import pandas as pd

from paper.benchmark import rsets as R
from paper.benchmark._inputs import inp

SETS = ["f1", "baseline", "output"]


def show(name, T, col):
    s = R.summarise(T, SETS, col, ["arch", "target"], "baseline")
    cells = [f"{x} {s[x]['k']}" + (f" (p = {s[x]['p_vs_baseline']:.2g})" if "p_vs_baseline" in s[x] else "") for x in SETS]
    print(f"{name}: " + ", ".join(cells) + ("" if s["complete"] else f"  INCOMPLETE {s.get('missing')}"))


def windowed():
    POS, RD, OR = R.rank_tables()
    W = pd.read_csv(inp("benchmark", "win", "win_latents.csv"))
    K = R.KEY
    T = pd.concat([W[W.set == s].merge(rk, on=K) for s, rk in (("f1", POS), ("baseline", RD), ("output", OR))],
                  ignore_index=True)
    none = []
    for m in ("piggen", "IgLM"):
        p = pd.read_csv(inp("benchmark", "positions", f"{m}_positions.csv"))
        for r in p[(p["rank"] == 1) & (p.peak_p95 < 0)].itertuples():
            none += [dict(model=m, arch=r.arch, target=r.gene, latent=-1, set=s, rank=k, bench="fail")
                     for s in SETS for k in (1, 2, 3)]
    return pd.concat([T, pd.DataFrame(none)], ignore_index=True)


def main():
    E = R.existing_sets()
    for m in R.M:
        show(f"benchmark {m} ({R.PROTOCOL[m]})", E[E.model == m], "paper_corr")
    T = windowed()
    for m in ("piggen", "IgLM"):
        show(f"windowed {m}", T[T.model == m], "bench")


if __name__ == "__main__":
    main()
