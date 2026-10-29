"""Baseline latents (Methods; Supplementary Section A.2): three per SAE and J gene, steered with the start position and
strengths of that gene's highest-F1 latent and scored for that gene. Their pass rate is the pipeline's false-positive
rate.
Eligible: fires (activation above zero at any position) in 5 to 50% of the selection pool
(plm_sae.steering.selection_pool, 1,800 validation heavy chains), at a rate within 1.25-fold of its overall rate for
every J gene, and not steered in any other set: the top-5 and low-index latents of paper.benchmark.positions, and for
p-IgGen the light-chain-type latents, the co-activation group members and the developability latents 980 and 1088.
  --set 1   one latent per gene, numpy default_rng(0), on the GPU      -> positions/<MODEL>_controls.csv
  --set 2   two more per gene, default_rng(1), on the CPU, excluding set 1 -> positions/<MODEL>_controls2.csv
(the firing rates were computed on these devices; picks are drawn without replacement within an SAE).

Run: python -m paper.benchmark.baselines MODEL --set 1|2   (writes to <work_dir>/benchmark/positions/)"""
import argparse

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from plm_sae.steering.selection_pool import validation_pool
from paper.benchmark import _inputs
from paper.benchmark.positions import SAES


def excluded(model, pos):
    """(arch, latent) steered in another set."""
    out = {(p.arch, int(p.latent)) for p in pos.itertuples()}
    if model == "piggen":
        arch = {"piggen_topk_r8": "TopK", "piggen_batchtopk_r8": "Batch TopK", "piggen_ordered_r8": "Ordered"}
        for sae, a in arch.items():
            for line in open(_inputs.ranking(f"pairs_{sae}_lct.txt")):
                if ":" in line:
                    out.add((a, int(line.strip().split(":")[1])))
        for _, sae, _, lats in _inputs.coactivation_groups():
            out |= {(arch[sae], l) for l in lats}
        out |= {("TopK", 980), ("Batch TopK", 1088)}
    return out


def firing_rates(model, sae, plm, lnf, tok, val, dev):
    """[genes, latents] share of each gene's pool sequences in which each latent fires."""
    fire, n = {}, {}
    for g, dg in val.groupby("gene"):
        seqs = dg.VHorVHH.astype(str).tolist()
        fire[g], n[g] = 0, 0
        for b in range(0, len(seqs), 32):
            tk = {k: v.to(dev) for k, v in models.tokenize(model, tok, seqs[b:b + 32]).items()}
            st = {}
            h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
            with torch.no_grad():
                plm(**tk)
            h.remove()
            x, msk = st["x"] / models.NORM_FACTOR[model], tk["attention_mask"].bool()
            with torch.no_grad():
                z = torch.stack([sae.encode(x[j]) for j in range(x.shape[0])]).clamp(min=0) * msk[..., None]
            fire[g] = fire[g] + (z.amax(1) > 0).float().sum(0).cpu()
            n[g] += x.shape[0]
    return sorted(fire), torch.stack([fire[g] / n[g] for g in sorted(fire)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("--set", type=int, choices=[1, 2], required=True)
    ap.add_argument("--device", default=None, help="default: cuda for set 1, cpu for set 2")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    F = P.work("benchmark", "positions")
    dev = torch.device(a.device or ("cuda" if a.set == 1 else "cpu"))
    torch.set_num_threads(a.threads)
    plm, tok = models.load(a.model, dev)
    core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
    lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
    val = validation_pool()
    pos = pd.read_csv(F / f"{a.model}_positions.csv")
    skip = excluded(a.model, pos)
    if a.set == 2:
        skip |= {(r.arch, int(r.latent)) for r in pd.read_csv(F / f"{a.model}_controls.csv").itertuples()}
    rng, per_gene = np.random.default_rng(a.set - 1), a.set
    rows = []
    for arch, name in SAES[a.model]:
        sae = S.load(name).to(dev)
        genes, rate_g = firing_rates(a.model, sae, plm, lnf, tok, val, dev)
        rate = rate_g.mean(0)
        enr = (rate_g / rate.clamp(min=1e-9)).amax(0)
        ok = [i for i in range(len(rate)) if 0.05 <= rate[i] <= 0.5 and enr[i] <= 1.25 and (arch, i) not in skip]
        pick = rng.choice(ok, size=6 * per_gene, replace=False)
        print(arch, "eligible", len(ok), "of", len(rate), "picked", list(map(int, pick)), flush=True)
        for g, lat in zip(genes * per_gene, pick):
            top = pos[(pos.arch == arch) & (pos.gene == g) & (pos["rank"] == 1)].iloc[0]
            rows.append(dict(model=a.model, arch=arch, gene=g, rank=-1, latent=int(lat), matched_to=int(top.latent),
                             frac_fire=round(float(rate[lat]), 3), max_enrichment=round(float(enr[lat]), 3),
                             peak_p05=int(top.peak_p05), onset_p05=int(top.onset_p05), lowidx=False))
    out = F / f"{a.model}_controls{'' if a.set == 1 else '2'}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
