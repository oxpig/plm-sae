"""Steering at the input of the final layer does not act through the steered latent (Supplementary Section A.3,
"Steering site"). Teacher-forced, for the highest-F1 latent of each gene and matched SAE, on 48 human validation heavy
chains: the vector alpha * d (d = unit decoder vector; alpha 6.5 p-IgGen, 12 IgLM, 24 ProGen2-OAS) is added at the input
of the final layer and
  FULL   the next-token distribution with that steering,
  CLAMP  the same, minus the latent's own change at the SAE's input (delta a_i * W_dec[i] * norm factor),
are compared with the unsteered model by KL divergence over the J region (the last 15 residues). KL_clamp_J / KL_full_J
is the share of the shift left after removing the latent's own change.
Output: <work_dir>/mediation/<MODEL>.csv. Usage: python -m paper.benchmark.mediation MODEL   (GPU)"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark import _inputs
from paper.benchmark.positions import SAES

LAYER = {"piggen": 3, "IgLM": 3, "progen2_oas": 26}
ALPHA = {"piggen": 6.5, "IgLM": 12.0, "progen2_oas": 24.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=models.MODELS)
    model = ap.parse_args().model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    nf, amax = models.NORM_FACTOR[model], ALPHA[model]
    plm, tok = models.load(model, dev)
    core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
    block = (core.layers if hasattr(core, "layers") else core.h)[LAYER[model]]
    lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH", "j_gene_anarci_heavy", "j_gene_species_anarci_heavy"])
    val = val[val.j_gene_species_anarci_heavy == "human"].dropna(subset=["j_gene_anarci_heavy"])
    rng = np.random.default_rng(1)
    rng.choice(len(val), 1200, replace=False)   # first draw of the paper's run (not used); keeps the 48 below identical
    med = val.iloc[rng.choice(len(val), 48, replace=False)]
    tk = {k: v.to(dev) for k, v in models.tokenize(model, tok, med.VHorVHH.tolist(),
                                                     max_len=175 if model == "piggen" else 160).items()}
    mm = tk["attention_mask"].bool()
    jmask = torch.zeros_like(mm)
    for b, n in enumerate(mm.sum(1).tolist()):
        jmask[b, max(0, n - 16):n - 1] = True

    def run(pre=None, post=None):
        """Logits and final-layer-norm output with optional additions at the final layer's input and the norm's output."""
        st, hs = {}, []
        if pre is not None:
            hs.append(block.register_forward_pre_hook(lambda m, i: (i[0] + pre.to(i[0].dtype),) + tuple(i[1:])))

        def lno(m, i, o):
            o = o - post.to(o.dtype) if post is not None else o
            st["ln"] = o.detach().float()
            return o
        hs.append(lnf.register_forward_hook(lno))
        with torch.no_grad():
            lo = plm(**tk).logits.float()
        for h in hs:
            h.remove()
        return lo, st["ln"]

    lo0, ln0 = run()
    lp0 = torch.log_softmax(lo0, -1)

    def kl(lo):
        return float(((lp0.exp() * (lp0 - torch.log_softmax(lo, -1))).sum(-1))[jmask].mean())

    out = []
    for arch, name in SAES[model]:
        sae, W = S.load(name).to(dev), S.decoder(name)
        top1 = {}
        for line in open(_inputs.ranking(f"pairs_{name}_top5.txt")):
            g, lat = line.strip().split(":")
            top1.setdefault(g, int(lat))
        with torch.no_grad():
            z0 = torch.stack([sae.encode(x) for x in ln0 / nf])
            for g, i in sorted(top1.items()):
                d, col = (W[i] / W[i].norm()).to(dev), W[i].to(dev)
                lo1, ln1 = run(pre=amax * d)
                z1 = torch.stack([sae.encode(x) for x in ln1 / nf])
                contrib = (z1[..., i] - z0[..., i])[..., None] * col * nf   # the latent's change, raw units
                loC, _ = run(pre=amax * d, post=contrib)
                out.append(dict(model=model, arch=arch, gene=g, latent=i, KL_full_J=kl(lo1), KL_clamp_J=kl(loC)))
                print(out[-1], flush=True)
    D = pd.DataFrame(out)
    D["share_left"] = D.KL_clamp_J / D.KL_full_J
    os.makedirs(P.work("mediation"), exist_ok=True)
    D.to_csv(P.work("mediation", f"{model}.csv"), index=False)
    print(f"{model}: median share of the J-region KL left after removing the latent's own change {D.share_left.median():.3f}")


if __name__ == "__main__":
    main()
