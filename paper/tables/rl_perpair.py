"""Supplementary Table 7 (tab:rl_perpair): per-pair detail of the highest-F1 latent of each SAE and heavy J gene, as a
LaTeX table body. Each model under its reported protocol (paper.benchmark.rsets: p-IgGen gated, IgLM without the
position gate, ProGen2-OAS at temperature 1.0). Columns: gene, latent, start position, alpha* (median over seeds),
target-gene proportion (mean +- SD over 3 seeds) at alpha = 0 and at each seed's alpha*, correlation criterion, both
criteria. Reads benchmark/final_latents.csv and progen2_t10/score/latents.csv; writes <work_dir>/tables/rl_perpair.tex.
Usage: python -m paper.tables.rl_perpair"""
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import rsets
from paper.benchmark._inputs import inp

MN = {"piggen": "p-IgGen", "IgLM": "IgLM", "progen2_oas": "ProGen2-OAS"}
COLS = ["task", "latent", "target", "start", "paper_astar_med", "paper_base", "paper_base_sd", "paper_ach", "paper_ach_sd",
        "paper_corr", "paper_mag"]


def main():
    FL = pd.read_csv(inp("benchmark", "final_latents.csv"))
    T10 = pd.read_csv(inp("progen2_t10", "score", "latents.csv"))
    E = rsets.existing_sets()
    x = E[(E.set == "f1") & (E["rank"] == 1)][["model", "arch", "target", "latent", "task"]]
    x = x.merge(pd.concat([FL[COLS], T10[COLS]]).drop_duplicates(["task", "latent", "target"]),
                on=["task", "latent", "target"], how="left")
    ok = lambda s: str(s) in ("pass", "True")
    out = []
    for m in rsets.M:
        for ai, a in enumerate(rsets.ARCH):
            y = x[(x.model == m) & (x.arch == a)].sort_values("target")
            assert len(y) == 6, (m, a, len(y))
            if out:
                out.append(r"\midrule")
            for gi, r in enumerate(y.itertuples()):
                corr, both = ok(r.paper_corr), ok(r.paper_corr) and ok(r.paper_mag)
                out.append(f"{MN[m] if ai == 0 and gi == 0 else ''} & {a if gi == 0 else ''} & {r.target} & {int(r.latent)} & "
                           f"{int(r.start)} & {r.paper_astar_med:g} & {r.paper_base:.3f} $\\pm$ {r.paper_base_sd:.3f} & "
                           f"{r.paper_ach:.3f} $\\pm$ {r.paper_ach_sd:.3f} & {'pass' if corr else '--'} & "
                           f"{'pass' if both else '--'} \\\\")
    body = "\n".join(out)
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    (o / "rl_perpair.tex").write_text(body + "\n")
    print(body)


if __name__ == "__main__":
    main()
