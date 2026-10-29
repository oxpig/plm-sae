"""The libraries of the steered generation analysis: for every latent of the paper's three sets (paper.benchmark.rsets:
p-IgGen gated, IgLM without the position gate, ProGen2-OAS at temperature 1.0) that passed the correlation criterion,
its seed-0 library at alpha* (median over seeds), and each model's unsteered seed-0 library (alpha 0; set unsteered;
UNSTEERED below).
Writes <work_dir>/quality/list.csv (tag, model, arch, set, rank, target, latent, alpha, both, lib); rows are processed
by paper.analyses.quality.run_one, the unsteered rows first.
Usage: python -m paper.analyses.quality.libraries   (reads <work_dir>/benchmark and <work_dir>/progen2_t10)"""
import os

import pandas as pd

from plm_sae import paths as P
from paper.benchmark import analyze_final as A
from paper.benchmark import rsets
from paper.benchmark.progen2_t10 import score as T10

SH = {"TopK": "TopK", "Batch TopK": "BatchTopK", "Ordered": "Ordered"}
# the unsteered reference of each model: the alpha-0 seed-0 library of one task (SAE, gene, latent, subdirectory)
UNSTEERED = {"piggen": ("Batch TopK", "IGHJ5", 1824, "f1/gated"), "IgLM": ("Batch TopK", "IGHJ3", 899, "f1/gated"),
             "progen2_oas": ("Batch TopK", "IGHJ1", 7925, "output")}


def main():
    FL = pd.read_csv(P.work("benchmark", "final_latents.csv"))
    T = rsets.existing_sets(FL, pd.read_csv(P.work("progen2_t10", "score", "latents.csv")))
    T = T[rsets.ok(T.paper_corr)].merge(pd.concat([FL, pd.read_csv(P.work("progen2_t10", "score", "latents.csv")).assign(
        model="progen2_oas")])[["model", "task", "latent", "target", "paper_astar_med"]].drop_duplicates(), on=["model", "task", "latent", "target"])
    dirs = {(e.task, e.latent, e.target): e.dir for e in A.experiments().itertuples()}
    dirs.update({(e.task, e.latent, e.target): e.dir for e in T10.experiments().itertuples()})
    rows = []
    for r in T.itertuples():
        a = float(r.paper_astar_med)
        lib = f"{dirs[(r.task, r.latent, r.target)]}/steered_l3_f{r.latent}_a{a:.2f}_seed0_seq1000.txt"
        rows.append(dict(tag=f"{r.model}_{SH[r.arch]}_{r.set}_{r.target}_f{r.latent}_a{a:g}", model=r.model, arch=r.arch,
                         set=r.set, rank=int(r.rank), target=r.target, latent=int(r.latent), alpha=a,
                         both=r.paper_mag == "pass", lib=lib))
    for m, (arch, target, latent, sub) in UNSTEERED.items():   # alpha 0: the unsteered model
        d = [v for (t, l, g), v in dirs.items() if l == latent and g == target and f"/{SH[arch]}/{sub}/" in v][0]
        rows.insert(0, dict(tag=f"{m}_unsteered_a0", model=m, arch="none", set="unsteered", rank=0, target="none",
                            latent=latent, alpha=0.0, both=False, lib=f"{d}/steered_l3_f{latent}_a0.00_seed0_seq1000.txt"))
    L = pd.DataFrame(rows)
    missing = [l for l in L.lib if not os.path.exists(l)]
    assert not missing, missing[:5]
    os.makedirs(P.work("quality"), exist_ok=True)
    L.to_csv(P.work("quality", "list.csv"), index=False)
    print(L.groupby(["model", "set"]).size().to_string())


if __name__ == "__main__":
    main()
