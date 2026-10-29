"""Latent sets and GPU tasks for steering CDRH3 properties (Yrich, positive, long) and developability metrics (PPC, PNC,
PSH, SFvCSP) in p-IgGen (Supplementary Sections A.10 and A.12). CPU.

Per SAE and target, three sets of three latents, all steered from one start:
  id_r1-3    CDRH3 properties only: the latents with the highest sequence-level F1 for the property
             (<work_dir>/outsel_properties/ident_scores.csv, paper.analyses.properties.ident_simple)
  out_r1-3   the three highest output scores (scores/out_scores.csv) among latents firing in >= 1% of the selection pool
  base1-3    baseline latents: firing rate 5-50% on the selection pool, within 1.25-fold of it for every J gene
             (scores/pool_stats.csv), not in baseline_exclusions.csv (latents steered in other runs of this study) or in
             the identification and output sets of the SAE; drawn with numpy default_rng without replacement from the
             sorted eligible latents and assigned in target order:
               TopK, Batch TopK   seed 261013 + SAE index (0, 1), 21 latents for Yrich, positive, long, PPC, PNC, PSH, SFvCSP
               Ordered            seed 261015, 9 latents for the CDRH3 properties; seed 261033, 12 latents for the
                                  developability metrics, also excluding the Ordered CDRH3 sets
Start: max(0, onset_p05 - 5) of the property's highest-F1 latent (onset over the sequences with the property) or of the
latent most correlated with the metric (highest |Pearson r| among live latents of the developability gate; onset over
all sequences), from scores/latent_positions.csv.
Writes manifest.csv (target, kind, arch, set, latent, start, run), runs.csv (one run per SAE, latent and start) and one
task per run (base ladder 0-120 in steps of 8, seeds 0-2) in <work_dir>/outsel_properties/tasks/; libraries in
runs/piggen/<SAE>/s<start>/lib/. Usage: python -m paper.analyses.properties.outsel.manifest"""
import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

ARCHS = ["TopK", "Batch TopK", "Ordered"]
SAE = {"TopK": "piggen_topk_r8", "Batch TopK": "piggen_batchtopk_r8", "Ordered": "piggen_ordered_r8"}
LAB, DEV = ["Yrich", "positive", "long"], ["PPC", "PNC", "PSH", "SFvCSP"]
LADDER = np.arange(0, 120.001, 8)


def start(o):
    return max(0, o - 5) if o >= 0 else 0


def draw(PL, arch, excluded, n, seed):
    p = PL[(PL.arch == arch) & (PL.rate >= 0.05) & (PL.rate <= 0.5) & (PL.max_enrichment <= 1.25)]
    elig = sorted(set(p.latent.astype(int)) - excluded)
    return [int(x) for x in np.random.default_rng(seed).choice(elig, size=n, replace=False)]


def main():
    S = P.work("outsel_properties")
    sc = S / "scores" if (S / "scores").exists() else P.results("outsel_properties", "scores")
    LP = pd.read_csv(sc / "latent_positions.csv").set_index(["arch", "subset", "latent"])
    OS, PL = pd.read_csv(sc / "out_scores.csv"), pd.read_csv(sc / "pool_stats.csv")
    SF = pd.read_csv(inp("benchmark", "probe", "scores_piggen.csv")).drop_duplicates(["arch", "latent"])[["arch", "latent", "seq_freq"]]
    OS = OS.merge(SF, on=["arch", "latent"])
    ID = pd.read_csv(inp("outsel_properties", "ident_scores.csv"))
    EX = pd.read_csv(inp("outsel_properties", "baseline_exclusions.csv"))
    starts, ids, outs = {}, {}, {}
    for arch in ARCHS:
        o = OS[(OS.arch == arch) & (OS.seq_freq >= 0.01)]
        for t in LAB:
            f1r = ID[(ID.arch == arch) & (ID.label == t)].sort_values("f1", ascending=False, kind="stable").latent.astype(int)
            ids[(arch, t)] = f1r.tolist()[:3]
            starts[(arch, t)] = start(int(LP.loc[(arch, t, ids[(arch, t)][0])].onset_p05))
        for t in DEV:
            d = pd.read_csv(inp("developability", "gate", SAE[arch], f"developability_{t}_per_latent.csv"))
            live = d[~d.dead.astype(bool)]
            starts[(arch, t)] = start(int(LP.loc[(arch, "all", int(live.loc[live.pearson_r.abs().idxmax()].latent))].onset_p05))
        for t in LAB + DEV:
            outs[(arch, t)] = o.sort_values(t, ascending=False, kind="stable").latent.astype(int).tolist()[:3]
    base = {}
    for i, arch in enumerate(ARCHS):
        ex = set(EX[EX.arch == arch].latent.astype(int))
        groups = [(LAB + DEV, 261013 + i)] if arch != "Ordered" else [(LAB, 261015), (DEV, 261033)]
        for targets, seed in groups:
            used = ex | {l for t in targets for l in ids.get((arch, t), []) + outs[(arch, t)]}
            if targets == DEV and arch == "Ordered":
                used |= {l for t in LAB for l in ids[(arch, t)] + outs[(arch, t)] + base[(arch, t)]}
            pick = draw(PL, arch, used, 3 * len(targets), seed)
            for j, t in enumerate(targets):
                base[(arch, t)] = pick[3 * j:3 * j + 3]
    rows = []
    for arch in ARCHS:
        for t in LAB + DEV:
            sets = [(f"id_r{k + 1}", l) for k, l in enumerate(ids.get((arch, t), []))] if t in LAB else []
            sets += [(f"out_r{k + 1}", l) for k, l in enumerate(outs[(arch, t)])]
            sets += [(f"base{k + 1}", l) for k, l in enumerate(base[(arch, t)])]
            st = starts[(arch, t)]
            rows += [dict(target=t, kind="graded" if t in LAB else "dev", arch=arch, set=s, latent=l, start=st,
                          run=f"uo_{arch.replace(' ', '')}_f{l}_s{st}") for s, l in sets]
    M = pd.DataFrame(rows)
    M.to_csv(S / "manifest.csv", index=False)
    R = M.drop_duplicates(["run"])[["run", "arch", "latent", "start"]]
    R.to_csv(S / "runs.csv", index=False)
    for d in ("tasks/done", "tasks/failed", "cfg", "units"):
        (S / d).mkdir(parents=True, exist_ok=True)
    for r in R.itertuples():
        out, tag = S / "runs/piggen" / r.arch.replace(" ", ""), f"s{r.start}/lib"
        (S / "cfg" / f"{r.run}.txt").write_text("".join(f"{r.latent} {a:.2f} {r.start} {tag}\n" for a in LADDER))
        (S / "units" / f"{r.run}.txt").write_text("".join(f"{out / tag} {r.latent} lib {a:.2f} {s}\n" for a in LADDER for s in (0, 1, 2)))
        (S / "tasks" / r.run).write_text(f'MODEL=piggen\nCK={SAE[r.arch]}\nCFG="{S}/cfg/{r.run}.txt"\nOUT={out}\n'
                                         f'UNITS={S}/units/{r.run}.txt\nNLIB={3 * len(LADDER)}\n')
    (S / "tasks/base.lst").write_text("".join(f"{t}\n" for t in R.run))
    print(M.groupby(["kind", "arch"]).size().to_string(), f"\n{len(R)} runs")


if __name__ == "__main__":
    main()
