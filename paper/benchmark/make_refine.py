"""Ladder refinement (Supplementary Section A.2): a latent whose valid range (contiguous alphas from 0 with >= 80% pooled
coverage) holds fewer than five strengths gets seven evenly spaced strengths between 0 and the first invalid alpha
(a_hi * k/8, k = 1..7, minus those already run). Same SAE, site, start, seeds and output directory as the latent's
existing libraries (analyze_final merges alphas per directory, latent and target). Tasks r_* (up to 4 latents
each); extend.py does not extend them again.
Usage: python -m paper.benchmark.make_refine   (state <work_dir>/benchmark)"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import analyze_final as af

F = P.work("benchmark")


def kv(p):
    return dict(re.findall(r'^(\w+)="?(.*?)"?$', p.read_text(), re.M))


def main():
    R = pd.read_csv(F / "final_latents.csv")
    X = af.experiments().set_index(["dir", "latent", "target"])
    todo = []
    for r in R.itertuples():
        if r.target == "dev" or r.valid_amax != r.valid_amax:
            continue
        d = next((k[0] for k in X.index if k[1] == r.latent and k[2] == r.target and X.loc[k, "task"] == r.task), None)
        if d is None:
            continue
        al = np.array(X.loc[(d, r.latent, r.target), "alphas"])
        npts = int((al <= r.valid_amax + 1e-9).sum())
        if npts >= 5 or r.valid_amax >= al.max() - 1e-9:
            continue
        a_hi = al[al > r.valid_amax + 1e-9].min()
        new = [a for a in (round(a_hi * k / 8, 2) for k in range(1, 8)) if not np.any(np.abs(al - a) < 1e-6)]
        if new:
            todo.append(dict(task=r.task, dir=d, latent=int(r.latent), target=r.target, start=int(r.start), new=new,
                             model=r.model, set=r.set, npts=npts, a_hi=a_hi))
    T = pd.DataFrame(todo)
    if not len(T):
        print("0 latents to refine")
        return
    print(T.groupby(["model", "set"]).size().to_string(), "\n", len(T), "latents")
    n = 0
    for task, g in T.groupby("task"):
        k = kv(F / "tasks/done" / task)
        flags = k["CFG"].split(" ", 1)
        flags = (" " + flags[1]) if len(flags) > 1 else ""
        rows = g.to_dict("records")
        for i in range(0, len(rows), 4):
            name = f"r_{task}" + (f"_{i // 4}" if len(rows) > 4 else "")
            cfg, units = [], []
            for e in rows[i:i + 4]:
                tag = "/".join(Path(e["dir"]).parts[-2:])
                for a in e["new"]:
                    cfg.append(f"{e['latent']} {a:.2f} {e['start']} {tag}")
                    units += [f"{e['dir']} {e['latent']} {e['target']} {a:.2f} {s}" for s in (0, 1, 2)]
            (F / "cfg" / f"{name}.txt").write_text("\n".join(cfg) + "\n")
            (F / "units" / f"{name}.txt").write_text("\n".join(units) + "\n")
            (F / "tasks/todo" / name).write_text(f'MODEL={k["MODEL"]}\nCK={k["CK"]}\nCFG="{F}/cfg/{name}.txt{flags}"\n'
                                                 f'OUT={k["OUT"]}\nUNITS={F}/units/{name}.txt\nNLIB={len(units)}\n')
            n += 1
            print(name, len(units))
    T.to_csv(F / "refine_list.csv", index=False)
    print(n, "tasks")


if __name__ == "__main__":
    main()
