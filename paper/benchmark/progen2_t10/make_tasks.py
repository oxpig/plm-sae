"""GPU tasks of the ProGen2-OAS benchmark (Supplementary Section A.2): temperature 1.0, top-p 0.95, prompt
1EVQLVESGGGLVQPGGSLRLSCAAS, TF32 matrix products. 162 latents, one per task: output-selected ranks 1-3 (t10a),
baseline draws 1-3 (t10b) and highest-F1 ranks 1-3 (t10c) of every SAE and J gene, base ladder 0-30 in steps of 3,
seeds 0-2. Start: max(0, onset_p05 - 5) of the latent (highest-F1) or of the gene's highest-F1 latent (baseline and
output-selected latents; plm_sae.steering.pick_output on probe/scores_progen2_oas.csv).
Reads positions/ and probe/ of the benchmark (<work_dir>/benchmark, else <results_dir>/benchmark).
Libraries: <work_dir>/progen2_t10/runs/progen2_oas/<SAE>/<set>/<gene>/.
Usage: python -m paper.benchmark.progen2_t10.make_tasks"""
import pandas as pd

from plm_sae import paths as P
from plm_sae.steering.pick_output import picks
from paper.benchmark._inputs import inp

M = "progen2_oas"
SH = {"TopK": "TopK", "Batch TopK": "BatchTopK", "Ordered": "Ordered"}
SAE = {"TopK": "progen2oas_topk_r8", "Batch TopK": "progen2oas_batchtopk_r8", "Ordered": "progen2oas_ordered_r8"}
LADDER = [3.0 * i for i in range(11)]
FLAGS = "--tf32 --temp 1.0 --prompt 1EVQLVESGGGLVQPGGSLRLSCAAS"
K = ["arch", "target", "latent"]


def sets():
    S = inp("benchmark", "positions", f"{M}_positions.csv").parent
    pos = pd.read_csv(S / f"{M}_positions.csv").rename(columns={"gene": "target"})
    pos = pos[pos["rank"].between(1, 3)].assign(start=lambda d: (d.onset_p05 - 5).clip(lower=0))
    c1 = pd.read_csv(S / f"{M}_controls.csv").assign(rank=1)
    c2 = pd.read_csv(S / f"{M}_controls2.csv")
    c2["rank"] = c2.groupby(["arch", "gene"]).cumcount() + 2
    # baseline files carry the onset of the gene's highest-F1 latent
    rd = pd.concat([c1, c2]).rename(columns={"gene": "target"}).assign(start=lambda d: (d.onset_p05 - 5).clip(lower=0))
    top = pos[pos["rank"] == 1].set_index(["arch", "target"]).onset_p05
    op = picks(pd.read_csv(inp("benchmark", "probe", f"scores_{M}.csv"))).rename(columns={"gene": "target"})
    op["start"] = [max(0, int(top[(r.arch, r.target)]) - 5) for r in op.itertuples()]
    out = {"a_output": op, "b_baseline": rd, "c_f1": pos}
    return {k: v[v["rank"] <= 3][K + ["rank", "start"]].drop_duplicates(K) for k, v in out.items()}


def main():
    B = P.work("progen2_t10")
    for d in ("cfg", "units", "tasks/todo", "tasks/claimed", "tasks/done", "tasks/failed", "anarci", "logs", "score", "runs"):
        (B / d).mkdir(parents=True, exist_ok=True)
    sel = []
    for key, T in sets().items():
        assert len(T) == 54, (key, len(T))
        name_set = key.split("_")[1]
        for r in T.itertuples():
            name, out = f"t10{key[0]}_{SH[r.arch]}_{r.target}_r{r.rank}", B / "runs" / M / SH[r.arch]
            (B / "cfg" / f"{name}.txt").write_text("".join(f"{r.latent} {a:.2f} {int(r.start)} {name_set}/{r.target}\n"
                                                          for a in LADDER))
            (B / "units" / f"{name}.txt").write_text("".join(f"{out}/{name_set}/{r.target} {r.latent} {r.target} {a:.2f} {s}\n"
                                                            for a in LADDER for s in (0, 1, 2)))
            (B / "tasks/todo" / name).write_text(f'MODEL={M}\nCK={SAE[r.arch]}\nCFG="{B}/cfg/{name}.txt {FLAGS}"\n'
                                                 f'OUT={out}\nUNITS={B}/units/{name}.txt\nNLIB={3 * len(LADDER)}\n')
            sel.append(dict(set=name_set, arch=r.arch, target=r.target, rank=int(r.rank), latent=int(r.latent),
                            start=int(r.start), task=name))
    pd.DataFrame(sel).to_csv(B / "selected.csv", index=False)
    print(len(sel), "tasks;", pd.DataFrame(sel).groupby("set").size().to_dict())


if __name__ == "__main__":
    main()
