"""Windowed steering at natural scale (Supplementary Section A.4, "Steering at natural scale"), p-IgGen and IgLM.

Every latent of a gene and SAE is steered only inside the window of the gene's highest-F1 latent, [start, end) with
start = max(0, peak_p05 - 5) and end = peak_p95 + 1 (positions/<model>_positions.csv), at alpha = MULT x raw_p95 of the
highest-F1 latent (its natural contribution, natural/natural_<model>.csv), so every latent of the gene gets the same
window and strengths. One task per SAE and gene: alpha 0 once, then the three sets of paper.benchmark.make_tasks
(f1, baseline, output; three latents each). A gene whose highest-F1 latent never fires on its sequences has no window.
Tasks: <work_dir>/benchmark/win/tasks/todo (configs with a fifth field, the window end); win_configs.csv lists them.
Usage: python -m paper.benchmark.windowed.make_win"""
import os

import pandas as pd

from plm_sae import paths as P

F = P.work("benchmark")
W = F / "win"
MULT = [0.5, 1, 2, 4, 8, 16, 32, 64]
SAE = {("piggen", "TopK"): "piggen_topk_r8", ("piggen", "Batch TopK"): "piggen_batchtopk_r8",
       ("piggen", "Ordered"): "piggen_ordered_r8", ("IgLM", "TopK"): "iglm_topk_r8",
       ("IgLM", "Batch TopK"): "iglm_batchtopk_r8", ("IgLM", "Ordered"): "iglm_ordered_r8"}


def add_task(name, model, arch, cfgs, rows):
    a = arch.replace(" ", "")
    out, cf, un = W / "runs" / model / a, W / "cfg" / f"{name}.txt", W / "units" / f"{name}.txt"
    cf.write_text("".join(f"{c['latent']} {c['alpha']:.2f} {c['start']} {c['set']}/{c['gene']} {c['end']}\n" for c in cfgs))
    un.write_text("".join(f"{out}/{c['set']}/{c['gene']} {c['latent']} {c['gene']} {c['alpha']:.2f} {s}\n"
                          for c in cfgs for s in (0, 1, 2)))
    (W / "tasks/todo" / name).write_text(f'MODEL={model}\nCK={SAE[(model, arch)]}\nCFG="{cf}"\nOUT={out}\nUNITS={un}\n'
                                         f'NLIB={3 * len(cfgs)}\n')
    rows += [dict(model=model, arch=arch, task=name, **c) for c in cfgs]


def main():
    from paper.benchmark.make_tasks import sets
    rows = []
    for model in ("piggen", "IgLM"):
        pos = pd.read_csv(F / f"positions/{model}_positions.csv")
        nat = pd.read_csv(F / f"natural/natural_{model}.csv").set_index(["arch", "latent"])
        S = sets(model)
        for _, t in pos[pos["rank"] == 1].iterrows():
            if t.peak_p95 < 0:
                print("no window", model, t.arch, t.gene, t.latent)
                continue
            st, en = max(0, int(t.peak_p05) - 5), int(t.peak_p95) + 1
            al = [round(k * float(nat.loc[(t.arch, t.latent), "raw_p95"]), 2) for k in MULT]
            cfgs = [dict(set="f1", gene=t.gene, latent=int(t.latent), start=st, end=en, alpha=0.0, mult=0.0)]
            for name in ("f1", "baseline", "output"):
                x = S[name][(S[name].arch == t.arch) & (S[name].gene == t.gene)].sort_values("rank")
                cfgs += [dict(set=name, gene=t.gene, latent=int(r.latent), start=st, end=en, alpha=a, mult=k)
                         for r in x.itertuples() for k, a in zip(MULT, al)]
            add_task(f"win_{model}_{t.arch.replace(' ', '')}_{t.gene}", model, t.arch, cfgs, rows)
    pd.DataFrame(rows).to_csv(W / "win_configs.csv", index=False)
    (W / "ALL_TASKS_ADDED").touch()   # every windowed task is queued; workers exit once the queue is empty
    print(pd.DataFrame(rows).groupby(["model", "set"]).latent.nunique())


if __name__ == "__main__":
    for d in ("cfg", "units", "tasks/todo", "tasks/claimed", "tasks/done", "tasks/failed"):
        os.makedirs(W / d, exist_ok=True)
    main()
