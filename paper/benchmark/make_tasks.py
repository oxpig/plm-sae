"""GPU tasks of the steering benchmark, p-IgGen and IgLM (ProGen2-OAS: paper.benchmark.progen2_t10).

Latents (three per SAE and J gene in every set):
  f1         highest-F1 ranks 1-3 (positions/<model>_positions.csv)
  baseline   baseline draws 1-3 (positions/<model>_controls.csv, _controls2.csv; paper.benchmark.baselines)
  output     output-selected ranks 1-3 (plm_sae.steering.pick_output on probe/scores_<model>.csv)
Start (Supplementary Section A.2): max(0, onset_p05 - 5) of the gene's highest-F1 latent for the baseline and output
sets, of the latent itself for the f1 set; ungated tasks steer from the first position. Strengths: LADDER (the base
range; paper.benchmark.extend and make_refine add strengths later). Seeds 0-2, 1,000 sequences.

Modes (each writes one task per SAE and gene unless noted):
  gated      f1, baseline and output sets of p-IgGen and IgLM, gated
  ungated    the same sets ungated (p-IgGen: f1 and baseline; IgLM: all three; IgLM is reported ungated)
  joint      the 22 co-activation groups (p-IgGen TopK and Batch TopK), gated; one task per SAE and phi
  other      p-IgGen light-chain type (one task per SAE), developability latents 980 and 1088, and the three
             highest-F1 latents per gene of the TopK SAE of Section 4.1 (output of block 3, strengths 0-240)
A task = configs file (cfg/<task>.txt) + units file (units/<task>.txt) + task file (tasks/todo/<task>), in
<work_dir>/benchmark. Libraries: runs/<model>/<SAE>/<set>/<gated|ungated>/<gene>/ (joint sets joint_phi05,
joint_phi07; other sets lct, dev, section41).
Run: python -m paper.benchmark.make_tasks MODE"""
import argparse

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from plm_sae import paths as P, sae as S
from plm_sae.steering.pick_output import picks
from paper.benchmark import _inputs

F = P.work("benchmark")
SAE = {("piggen", "TopK"): "piggen_topk_r8", ("piggen", "Batch TopK"): "piggen_batchtopk_r8",
       ("piggen", "Ordered"): "piggen_ordered_r8", ("IgLM", "TopK"): "iglm_topk_r8",
       ("IgLM", "Batch TopK"): "iglm_batchtopk_r8", ("IgLM", "Ordered"): "iglm_ordered_r8"}
ARCH = ["TopK", "Batch TopK", "Ordered"]
GENES = [f"IGHJ{i}" for i in range(1, 7)]
# base ranges: the largest alpha shifts the amino-acid logits by a similar amount relative to each model's temperature
LADDER = {"piggen": np.arange(0, 120.001, 8), "IgLM": np.arange(0, 60.001, 4)}
SECTION41_LADDER = np.arange(0, 240.001, 16)   # block output: 2.2 times the norm of the final-layer-norm output


def short(arch):
    return arch.replace("Batch TopK", "BatchTopK")


def onset_start(onset_p05):
    return int(max(0, onset_p05 - 5))


def write_task(name, model, arch, rows, alphas, sub, sae=None, extra=""):
    """rows: (gene, latent, rule, start); rule = gated or ungated, the directory below sub. sae overrides the SAE (a
    name or a .safetensors direction file); extra = generator flags appended to the configs path."""
    for d in ("cfg", "units", "tasks/todo", "tasks/claimed", "tasks/done", "tasks/failed"):
        (F / d).mkdir(parents=True, exist_ok=True)
    out = F / "runs" / model / short(arch) / sub
    cfg, units = [], []
    for g, lat, rule, st in rows:
        for a in alphas:
            cfg.append(f"{lat} {a:.2f} {st} {rule}/{g}")
            units += [f"{out}/{rule}/{g} {lat} {g} {a:.2f} {s}" for s in (0, 1, 2)]
    (F / "cfg" / f"{name}.txt").write_text("\n".join(cfg) + "\n")
    (F / "units" / f"{name}.txt").write_text("\n".join(units) + "\n")
    (F / "tasks/todo" / name).write_text(f'MODEL={model}\nCK={sae or SAE[(model, arch)]}\nCFG="{F}/cfg/{name}.txt{extra}"\n'
                                         f'OUT={out}\nUNITS={F}/units/{name}.txt\nNLIB={len(units)}\n')
    return [dict(task=name, model=model, arch=arch, sub=sub, gene=g, latent=lat, rule=rule, start=st) for g, lat, rule, st in rows]


def sets(model):
    """{set: DataFrame(arch, gene, latent, rank, onset_p05)} of the three sets; onset_p05 of the latent itself (f1) or
    of the gene's highest-F1 latent (baseline, output)."""
    pos = pd.read_csv(F / f"positions/{model}_positions.csv")
    f1 = pos[pos["rank"].between(1, 3)]
    top = pos[pos["rank"] == 1].set_index(["arch", "gene"]).onset_p05
    c1 = pd.read_csv(F / f"positions/{model}_controls.csv").assign(rank=1)
    c2 = pd.read_csv(F / f"positions/{model}_controls2.csv")
    c2["rank"] = c2.groupby(["arch", "gene"]).cumcount() + 2
    out = picks(pd.read_csv(_inputs.inp("benchmark", "probe", f"scores_{model}.csv")))
    out["onset_p05"] = [top[(r.arch, r.gene)] for r in out.itertuples()]
    bl = pd.concat([c1, c2])
    bl["onset_p05"] = [top[(r.arch, r.gene)] for r in bl.itertuples()]
    return {"f1": f1, "baseline": bl, "output": out}


def gated_or_ungated(rule):
    m = []
    for model in ("piggen", "IgLM"):
        S_ = sets(model)
        keep = ("f1", "baseline", "output") if rule == "gated" or model == "IgLM" else ("f1", "baseline")
        for name in keep:
            for arch in ARCH:
                for g in GENES:
                    x = S_[name][(S_[name].arch == arch) & (S_[name].gene == g)].sort_values("rank")
                    rows = [(g, int(r.latent), rule, onset_start(r.onset_p05) if rule == "gated" else 0) for r in x.itertuples()]
                    tag = {"f1": "f1", "baseline": "bl", "output": "out"}[name] + ("" if rule == "gated" else "u")
                    m += write_task(f"{tag}_{model}_{short(arch)}_{g}", model, arch, rows, LADDER[model], name)
    return m


def joint():
    """Direction of a group: normalised sum of its members' unit decoder vectors, written as the row of the group's
    highest-F1 latent in a direction file per SAE and phi; start from that latent's positions."""
    pos = pd.read_csv(F / "positions/piggen_positions.csv")
    groups, m = sorted(_inputs.coactivation_groups()), []
    for phi in ("0.5", "0.7"):
        for arch in ("TopK", "Batch TopK"):
            sae = SAE[("piggen", arch)]
            W = S.decoder(sae)
            J, rows, members = torch.zeros_like(W), [], {}
            for _, _, g, mem in (x for x in groups if x[0] == phi and x[1] == sae):
                members[g] = "+".join(map(str, mem))
                v = sum(W[i] / W[i].norm() for i in mem)
                J[mem[0]] = v / v.norm()
                p = pos[(pos.arch == arch) & (pos.gene == g) & (pos["rank"] == 1)].iloc[0]
                assert int(p.latent) == mem[0], (g, mem[0], p.latent)
                rows.append((g, mem[0], "gated", onset_start(p.onset_p05)))
            tag = f"phi0{phi[-1]}"
            (F / "joint").mkdir(parents=True, exist_ok=True)
            path = F / "joint" / f"{sae}_joint_{tag}.safetensors"
            save_file({"W_dec": J.contiguous()}, str(path))
            m += [dict(r, phi=tag, members=members[r["gene"]]) for r in
                  write_task(f"joint_{short(arch)}_{tag}", "piggen", arch, rows, LADDER["piggen"], f"joint_{tag}", sae=path)]
    return m


def other():
    G = pd.read_csv(F / "positions/piggen_generic_positions.csv")
    m = []
    for label, sub in (("lct", "lct"), ("dev", "dev")):
        for arch, ga in G[G.label == label].groupby("arch", sort=False):
            rows = [(p.gene, int(p.latent), "gated", onset_start(p.onset_p05)) for p in ga.itertuples()]
            m += write_task(f"{label}_{short(arch)}", "piggen", arch, rows, LADDER["piggen"], sub)
    s41 = G[G.label == "section41"]
    for g, ga in s41.groupby("gene"):
        rows = [(g, int(p.latent), "gated", onset_start(p.onset_p05)) for p in ga.itertuples()]
        m += write_task(f"s41_{g}", "piggen", "TopK", rows, SECTION41_LADDER, "section41", sae="piggen_topk_r32_layer3",
                        extra=" --site blockout")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["gated", "ungated", "joint", "other"])
    mode = ap.parse_args().mode
    m = {"gated": lambda: gated_or_ungated("gated"), "ungated": lambda: gated_or_ungated("ungated"),
         "joint": joint, "other": other}[mode]()
    M = pd.DataFrame(m)
    M.to_csv(F / f"manifest_{mode}.csv", index=False)
    print(M.groupby(["model", "sub"]).agg(tasks=("task", "nunique"), latents=("latent", "size")).to_string())


if __name__ == "__main__":
    main()
