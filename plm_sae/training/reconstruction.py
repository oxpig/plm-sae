"""FVU, L0 and dead fraction of the hyperparameter-matched SAEs on the 30,754 validation heavy chains
(vj_val_anarci.csv, VHorVHH): Supplementary Tables 4 and 5, and the matched expansion-32 TopK SAE of the
Supplementary Table 4 caption. Input: final-layer-norm output / norm factor, tokenised as for feature identification;
one sequence per encode() call. Statistics over real tokens (tokens=real, the reported rows) and over all padded
positions (tokens=all). FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2.

Run: python -m plm_sae.training.reconstruction {piggen,IgLM,progen2_oas,piggen_exp32}   (GPU)
Writes <work_dir>/identification/reconstruction/recon_val_<key>.csv and fire_counts_<key>.npz (per-latent firing
counts, arrays '<SAE>|real' and '<SAE>|all', read by the redundancy and developability stages)."""
import os
import sys

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S

SAES = {
    "piggen": [("TopK", "piggen_topk_r8"), ("Batch TopK", "piggen_batchtopk_r8"), ("Ordered", "piggen_ordered_r8")],
    "IgLM": [("TopK", "iglm_topk_r8"), ("Batch TopK", "iglm_batchtopk_r8"), ("Ordered", "iglm_ordered_r8")],
    "progen2_oas": [("TopK", "progen2oas_topk_r8"), ("Batch TopK", "progen2oas_batchtopk_r8"),
                    ("Ordered", "progen2oas_ordered_r8")],
    "piggen_exp32": [("TopK exp32", "piggen_topk_r32_matched")],
}


def _new(D, dev):
    return {m: dict(sse=0., sx=0., sxx=0., n=0, l0=0., fire=torch.zeros(D, device=dev)) for m in ("real", "all")}


def _add(s, x, xh, f):
    s["sse"] += ((x - xh) ** 2).sum().item()
    s["sx"] = s["sx"] + x.sum(0)
    s["sxx"] += (x ** 2).sum().item()
    s["n"] += x.shape[0]
    nz = f != 0
    s["l0"] += nz.sum().item()
    s["fire"] += nz.sum(0).float()


def main():
    key = sys.argv[1]
    model = "piggen" if key == "piggen_exp32" else key
    out = P.work("identification", "reconstruction")
    dev = torch.device("cuda")
    plm, tok = models.load(model, dev)
    saes = [(arch, name, S.load(name).to(dev), S.info(name)["dictionary_size"]) for arch, name in SAES[key]]
    st = {arch: _new(D, dev) for arch, _, _, D in saes}
    seqs = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH"])["VHorVHH"].astype(str).tolist()
    B = 32
    for b in range(0, len(seqs), B):
        tk = {k: v.to(dev) for k, v in models.tokenize(model, tok, seqs[b:b + B]).items()}
        mask = tk["attention_mask"].bool().reshape(-1)
        with torch.no_grad():
            x = plm(**tk, output_hidden_states=True).hidden_states[-1].float() / models.NORM_FACTOR[model]
            xf = x.reshape(-1, x.shape[-1])
            for arch, _, sae, _ in saes:
                f = torch.cat([sae.encode(x[i]) for i in range(x.shape[0])])
                xh = sae.decode(f)
                _add(st[arch]["all"], xf, xh, f)
                _add(st[arch]["real"], xf[mask], xh[mask], f[mask])
        if (b // B) % 100 == 0:
            print(f"[{b}/{len(seqs)}]", flush=True)

    rows = []
    for arch, name, _, D in saes:
        for m, s in st[arch].items():
            n = s["n"]
            var = s["sxx"] - n * ((s["sx"] / n) ** 2).sum().item()
            rows.append(dict(model=model, arch=arch, sae=name, dict=D, tokens=m, n_tokens=n, fvu=s["sse"] / var,
                             l0=s["l0"] / n, dead_frac=float((s["fire"] == 0).float().mean().item())))
            print(rows[-1], flush=True)
    os.makedirs(out, exist_ok=True)
    np.savez(out / f"fire_counts_{key}.npz", **{f"{arch}|{m}".replace(" ", "_"): st[arch][m]["fire"].cpu().numpy()
                                                for arch, _, _, _ in saes for m in ("real", "all")})
    pd.DataFrame(rows).to_csv(out / f"recon_val_{key}.csv", index=False)
    print("wrote", out / f"recon_val_{key}.csv")


if __name__ == "__main__":
    main()
