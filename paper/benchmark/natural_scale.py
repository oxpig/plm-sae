"""Natural scale of each latent's contribution at the steering site (Supplementary Section A.4, "Natural scale").

Steering adds alpha times the unit decoder vector to the final-layer-norm output. A latent's natural contribution there
is nf * z * ||W_dec[i]|| (the SAE encodes the output divided by the norm factor nf). On 300 validation heavy chains per
J gene (random_state 2), per latent: the share of sequences in which it fires and the median, 95th percentile and
maximum of its per-sequence peak, in raw units (raw_p50, raw_p95, raw_max).

  python -m paper.benchmark.natural_scale measure MODEL   (GPU) -> <work_dir>/benchmark/natural/natural_<MODEL>.csv
  python -m paper.benchmark.natural_scale compare         p-IgGen highest-F1 latents (ranks 1-3): median of
                                                          alpha* / raw_p95 (reads final_latents.csv and positions/)"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark._inputs import inp
from paper.benchmark.positions import SAES

GENES = [f"IGHJ{i}" for i in range(1, 7)]


def measure(model):
    out = P.work("benchmark", "natural")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = models.NORM_FACTOR[model]
    plm, tok = models.load(model, dev)
    core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
    lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH", "j_gene_anarci_heavy"])
    val["gene"] = val.j_gene_anarci_heavy.astype(str).str.split("*").str[0]
    val = val[val.gene.isin(GENES)].groupby("gene", group_keys=False).apply(lambda d: d.sample(n=min(300, len(d)), random_state=2))
    saes = [(arch, S.load(name).to(dev), S.decoder(name)) for arch, name in SAES[model]]
    peak = {arch: [] for arch, _, _ in saes}
    seqs = val.VHorVHH.astype(str).tolist()
    for b in range(0, len(seqs), 64):
        tk = {k: v.to(dev) for k, v in models.tokenize(model, tok, seqs[b:b + 64]).items()}
        st = {}
        h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
        with torch.no_grad():
            plm(**tk)
        h.remove()
        msk, x = tk["attention_mask"].bool(), st["x"] / nf
        for arch, sae, _ in saes:
            with torch.no_grad():
                z = torch.stack([sae.encode(x[j]) for j in range(x.shape[0])]).clamp(min=0) * msk[..., None]
            peak[arch].append(z.max(1).values.cpu().numpy())   # [B, latents] per-sequence peak
    rows = []
    for arch, _, Wd in saes:
        M = np.concatenate(peak[arch])
        dn, fire = Wd.norm(dim=1).numpy(), M > 0
        p50 = np.array([np.median(M[fire[:, i], i]) if fire[:, i].any() else 0. for i in range(M.shape[1])])
        p95 = np.array([np.percentile(M[fire[:, i], i], 95) if fire[:, i].any() else 0. for i in range(M.shape[1])])
        zmax = M.max(0)
        rows.append(pd.DataFrame(dict(model=model, arch=arch, latent=np.arange(M.shape[1]), frac_seq_fire=fire.mean(0),
                                      dec_norm=dn, z_max=zmax, z_seqpeak_p50=p50, z_seqpeak_p95=p95, raw_max=nf * zmax * dn,
                                      raw_p50=nf * p50 * dn, raw_p95=nf * p95 * dn)))
    pd.concat(rows).to_csv(out / f"natural_{model}.csv", index=False)
    print("wrote", out / f"natural_{model}.csv")


def compare():
    R = pd.read_csv(inp("benchmark", "final_latents.csv"))
    pos = pd.read_csv(inp("benchmark", "positions", "piggen_positions.csv"))
    N = pd.read_csv(inp("benchmark", "natural", "natural_piggen.csv"))
    X = R[(R.model == "piggen") & (R.set == "f1")].drop(columns=["rank"])
    X = X.merge(pos[pos["rank"].between(1, 3)][["arch", "gene", "latent", "rank"]].rename(columns={"gene": "target"}),
                on=["arch", "target", "latent"]).merge(N, on=["model", "arch", "latent"])
    X["astar_over_p95"] = X.paper_astar_med / X.raw_p95
    print(f"p-IgGen highest-F1 ranks 1-3: n {len(X)}, median alpha* / raw_p95 = {X.astar_over_p95.median():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["measure", "compare"])
    ap.add_argument("model", nargs="?", choices=models.MODELS)
    a = ap.parse_args()
    measure(a.model) if a.mode == "measure" else compare()


if __name__ == "__main__":
    main()
