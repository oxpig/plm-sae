"""Steering of the expansion-32 TopK SAE of Section 4.1 (Section 4.2.1, first paragraph): the three highest-F1 latents
of each heavy J gene (18 latents), gated, from benchmark/final_latents.csv (set section41; the set also holds the
case-study latent 463 of Supplementary Figure 7). Prints the latents passing the correlation criterion (of 18), the genes
they steer, and the passes per rank.
Usage: python -m paper.tables.section41_steering"""
import pandas as pd

from paper.benchmark._inputs import inp, section41_latents
from paper.tables.common import ok


def main():
    R = pd.read_csv(inp("benchmark", "final_latents.csv"))
    S = R[R.set == "section41"].drop(columns=["rank"])
    ranks = pd.DataFrame([dict(target=g, latent=l, rank=k) for g, l, k in section41_latents() if k <= 3])
    S = S.merge(ranks, on=["target", "latent"])
    assert len(S) == 18, len(S)
    p = S[ok(S.paper_corr)]
    print(f"correlation criterion: {len(p)} of {len(S)}; both criteria: {int((ok(S.paper_corr) & ok(S.paper_mag)).sum())}")
    print("genes steered:", ", ".join(sorted(p.target.unique())))
    print("passes by rank:", p.groupby("rank").size().reindex([1, 2, 3], fill_value=0).to_dict())
    print(S.sort_values(["target", "rank"])[["target", "rank", "latent", "paper_base", "paper_ach", "paper_corr",
                                               "paper_mag"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
