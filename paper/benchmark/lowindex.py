"""Low-index selection for the Ordered SAEs (Supplementary Section A.11). The Ordered objective puts
coarse structure in the lowest-index latents, so the index is itself a selection criterion: for each J gene, the three
lowest-index latents among its 50 highest-F1 latents (identification/gates/<SAE>/f1_scores/, the ANARCI ranking of the
benchmark). They are steered under each model's reported protocol (p-IgGen gated, IgLM ungated, ProGen2-OAS at temperature
1.0 with the human FR1 prompt) from the start of the gene's highest-F1 latent, as the baseline and output-selected sets,
and scored with the benchmark's criteria.
  select    -> <work_dir>/benchmark/lowindex/selection.csv
  tasks     p-IgGen and IgLM tasks in <work_dir>/benchmark (set directory lowidx); ProGen2-OAS tasks t10d_* in
            <work_dir>/progen2_t10 (rows appended to selected.csv with set lowidx)
  summary   passes per model, k of 54, and the stratified exact test (paper.benchmark.rsets) against the highest-F1 and
            the baseline latents of the Ordered SAEs
Usage: python -m paper.benchmark.lowindex select|tasks|summary"""
import argparse

import pandas as pd

from plm_sae import paths as P
from paper.benchmark import make_tasks as MT, rsets as R
from paper.benchmark._inputs import inp
from paper.benchmark.progen2_t10 import make_tasks as T10

N_TOP, K = 50, 3
SAE = {"piggen": "piggen_ordered_r8", "IgLM": "iglm_ordered_r8", "progen2_oas": "progen2oas_ordered_r8"}
GENES = [f"IGHJ{i}" for i in range(1, 7)]
OUT = P.work("benchmark", "lowindex")


def top_onset(model):
    pos = pd.read_csv(inp("benchmark", "positions", f"{model}_positions.csv"))
    return pos[(pos.arch == "Ordered") & (pos["rank"] == 1)].set_index("gene").onset_p05


def select():
    rows = []
    for model, sae in SAE.items():
        pos = pd.read_csv(inp("benchmark", "positions", f"{model}_positions.csv"))
        for g in GENES:
            f = pd.read_csv(inp("identification", "gates", sae, "f1_scores", f"f1_scores_class_{g}.csv"))
            f = f.sort_values(["best_f1", "feature_idx"], ascending=[False, True]).reset_index(drop=True)
            f["f1_rank"] = f.index + 1
            top3 = set(f.feature_idx.head(3))
            paper3 = set(pos[(pos.arch == "Ordered") & (pos.gene == g) & pos["rank"].between(1, 3)].latent)
            if top3 != paper3:
                print(f"note {model} {g}: top 3 of the F1 table {sorted(top3)} differ from positions ranks 1-3 {sorted(paper3)}")
            low = f.head(N_TOP).sort_values("feature_idx").head(K)
            for k, r in enumerate(low.itertuples(), 1):
                rows.append(dict(model=model, sae=sae, arch="Ordered", gene=g, rank=k, latent=int(r.feature_idx),
                                 f1=float(r.best_f1), f1_rank=int(r.f1_rank)))
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "selection.csv", index=False)
    print(S.groupby("model").apply(lambda d: f"{len(d)} latents, indices {d.latent.min()}-{d.latent.max()}, "
                                             f"F1 ranks {d.f1_rank.min()}-{d.f1_rank.max()}").to_string())


def tasks():
    S = pd.read_csv(OUT / "selection.csv")
    for model, rule in (("piggen", "gated"), ("IgLM", "ungated")):
        on = top_onset(model)
        for g in GENES:
            x = S[(S.model == model) & (S.gene == g)].sort_values("rank")
            st = MT.onset_start(on[g]) if rule == "gated" else 0
            tag = "li" if rule == "gated" else "liu"
            MT.write_task(f"{tag}_{model}_Ordered_{g}", model, "Ordered", [(g, int(r.latent), rule, st) for r in x.itertuples()],
                          MT.LADDER[model], "lowidx")
    B, M, on = P.work("progen2_t10"), T10.M, top_onset("progen2_oas")
    sel_path, new = B / "selected.csv", []
    for r in S[S.model == M].itertuples():
        name, out, st = f"t10d_Ordered_{r.gene}_r{r.rank}", B / "runs" / M / "Ordered", max(0, int(on[r.gene]) - 5)
        (B / "cfg" / f"{name}.txt").write_text("".join(f"{r.latent} {a:.2f} {st} lowidx/{r.gene}\n" for a in T10.LADDER))
        (B / "units" / f"{name}.txt").write_text("".join(f"{out}/lowidx/{r.gene} {r.latent} {r.gene} {a:.2f} {s}\n"
                                                        for a in T10.LADDER for s in (0, 1, 2)))
        (B / "tasks/todo" / name).write_text(f'MODEL={M}\nCK={T10.SAE["Ordered"]}\nCFG="{B}/cfg/{name}.txt {T10.FLAGS}"\n'
                                             f'OUT={out}\nUNITS={B}/units/{name}.txt\nNLIB={3 * len(T10.LADDER)}\n')
        new.append(dict(set="lowidx", arch="Ordered", target=r.gene, rank=int(r.rank), latent=int(r.latent), start=st, task=name))
    old = pd.read_csv(sel_path) if sel_path.exists() else pd.DataFrame()
    old = old[old.set != "lowidx"] if len(old) else old
    pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(sel_path, index=False)
    print(f"tasks: p-IgGen 6 (gated), IgLM 6 (ungated), ProGen2-OAS {len(new)}")


def summary():
    S = pd.read_csv(inp("benchmark", "lowindex", "selection.csv")).rename(columns={"gene": "target"})
    FL = pd.read_csv(inp("benchmark", "final_latents.csv"))
    T10L = pd.read_csv(inp("progen2_t10", "score", "latents.csv"))
    li = pd.concat([FL[(FL.model == "piggen") & (FL.set == "low-index") & (FL.rule == "gated")],
                    FL[(FL.model == "IgLM") & (FL.set == "low-index") & (FL.rule == "ungated")],
                    T10L[T10L.set == "lowidx"].assign(model="progen2_oas")])
    li = li.drop(columns=["rank"], errors="ignore").merge(S[R.KEY + ["rank"]], on=R.KEY).assign(set="lowidx")
    ex = R.existing_sets(FL, T10L)
    ex = ex[(ex.arch == "Ordered") & ex.set.isin(["f1", "baseline", "output"])]   # ProGen2-OAS latents.csv also holds the lowidx rows
    T = pd.concat([ex, li[R.KEY + ["rank", "set", "paper_corr", "paper_mag"]]], ignore_index=True)
    for col, name in (("paper_corr", "correlation criterion"), ("paper_mag", "both criteria")):
        per = T.assign(p=R.ok(T[col])).groupby(["model", "set"]).p.sum().unstack().reindex(R.M)
        print(f"{name}, of 18 per model (Ordered SAEs):\n{per[['lowidx', 'f1', 'baseline', 'output']].to_string()}")
        for ref in ("f1", "baseline"):
            s = R.summarise(T, ["lowidx", ref], col, ["model", "target"], ref)
            print(f"  low-index {s['lowidx']['k']} vs {ref} {s[ref]['k']} of {s['n']}: p = {s['lowidx'].get('p_vs_' + ref, float('nan')):.3g}"
                  + ("" if s["complete"] else f"  INCOMPLETE {s.get('missing')}"))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["select", "tasks", "summary"])
    {"select": select, "tasks": tasks, "summary": summary}[p.parse_args().step]()


if __name__ == "__main__":
    main()
