"""Steering lambda light chains in p-IgGen (Supplementary Section A.12, "Lambda light chains": 0 of 9).
  tasks   the three lambda latents of each p-IgGen SAE (identification/pairs/pairs_<sae>_lct.txt), steered from the
          start of the SAE's highest-F1 lambda latent (positions/piggen_generic_positions.csv), alpha 0-120 in steps of
          8, seeds 0-2: one task per SAE in <work_dir>/lambda/tasks/ (run with scripts/generation/run_queued_task.sh)
  score   readout = share of lambda among light chains with a classified type (paper.analyses.properties.props.lc_type,
          from the last residue of the light-chain FR4); coverage per seed = classified light chains / 1,000; the
          paper's criteria (paper.benchmark.analyze_final.paper) -> <work_dir>/lambda/lambda_latents.csv
Usage: python -m paper.analyses.properties.lambda_steering tasks|score"""
import os
import sys

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.analyses.properties.props import lc_type
from paper.benchmark import analyze_final as A
from paper.benchmark._inputs import inp, ranking

SAES = {"TopK": "piggen_topk_r8", "Batch TopK": "piggen_batchtopk_r8", "Ordered": "piggen_ordered_r8"}
LADDER = [8.0 * i for i in range(16)]
L = P.work("lambda")


def latents(sae):
    return [int(l.split(":")[1]) for l in open(ranking(f"pairs_{sae}_lct.txt")) if l.startswith("lambda:")]


def tasks():
    G = pd.read_csv(inp("benchmark", "positions", "piggen_generic_positions.csv"))
    for d in ("tasks", "cfg", "units"):
        os.makedirs(L / d, exist_ok=True)
    for arch, sae in SAES.items():
        lats = latents(sae)
        p = G[(G.label == "lct") & (G.arch == arch) & (G.latent == lats[0]) & (G.gene == "lambda")].iloc[0]
        st, a = max(0, int(p.onset_p05) - 5), arch.replace(" ", "")
        out, name = L / "runs" / a, f"lambda_{a}"
        (L / "cfg" / f"{name}.txt").write_text("".join(f"{l} {al:.2f} {st} lambda/f1\n" for l in lats for al in LADDER))
        (L / "units" / f"{name}.txt").write_text("".join(f"{out}/lambda/f1 {l} lambda {al:.2f} {s}\n"
                                                         for l in lats for al in LADDER for s in (0, 1, 2)))
        (L / "tasks" / name).write_text(f'MODEL=piggen\nCK={sae}\nCFG="{L}/cfg/{name}.txt"\nOUT={out}\n'
                                        f'UNITS={L}/units/{name}.txt\nNLIB={9 * len(LADDER)}\n')
        print(name, lats, "start", st)


def lib(path):
    n = lam = 0
    if os.path.exists(path):
        for line in open(path):
            t = lc_type(line.strip()) if line.strip() else None
            if t:
                n += 1
                lam += t == "lambda"
    return n, lam


def score():
    rows = []
    for arch, sae in SAES.items():
        d = L / "runs" / arch.replace(" ", "") / "lambda" / "f1"
        for lat in latents(sae):
            X = np.array([[lib(f"{d}/steered_l3_f{lat}_a{al:.2f}_seed{s}_seq1000.txt") for al in LADDER] for s in (0, 1, 2)], float)
            with np.errstate(all="ignore"):
                pb = A.paper(np.array(LADDER), X[..., 1] / X[..., 0], X[..., 0] / 1000)
            rows.append(dict(arch=arch, latent=lat, corr=pb["paper_corr"], mag=pb["paper_mag"],
                             base=pb["paper_base"], at_astar=pb["paper_ach"], astar=pb["paper_astar_med"]))
    D = pd.DataFrame(rows)
    D.to_csv(L / "lambda_latents.csv", index=False)
    print(D.to_string(index=False))
    print(f"lambda latents passing the correlation criterion: {(D['corr'] == 'pass').sum()} of {len(D)}")


if __name__ == "__main__":
    {"tasks": tasks, "score": score}[sys.argv[1]]()
