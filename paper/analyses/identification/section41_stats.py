"""FVU, L0 and never-fired share of the TopK SAE of Section 4.1 (p-IgGen block 3) on the 30,754 validation heavy chains
('1' + VH + '2'), real tokens only, input = output of block 3 (Supplementary Table 4 caption: FVU 0.0088, L0 32.0,
33.0% never fired). FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2.
Usage: python -m paper.analyses.identification.section41_stats   (GPU)
Output: <work_dir>/identification/reconstruction/section41_stats.csv"""
import os

import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S


def main():
    dev = "cuda"
    out = P.work("identification", "reconstruction", "section41_stats.csv")
    seqs = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH"])["VHorVHH"].astype(str).tolist()
    model, tok = models.load("piggen", dev)
    sae = S.load("piggen_topk_r32_layer3").to(dev)
    D = sae.W_enc.shape[0]
    cap = []
    model.gpt_neox.layers[3].register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float()))
    fire, st = torch.zeros(D, device=dev), dict(l0=0., n=0, num=0., sx=0., sxx=0.)
    for b in range(0, len(seqs), 64):
        ids = models.tokenize("piggen", tok, seqs[b:b + 64])["input_ids"].to(dev)
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids)
            x = cap[0][ids != tok.pad_token_id]
            for c in range(0, x.shape[0], 16384):
                xc = x[c:c + 16384]
                v, i = sae.topk(xc)
                fire.index_add_(0, i[v > 0], torch.ones_like(v[v > 0]))
                st["l0"] += (v > 0).sum().item()
                st["n"] += xc.shape[0]
                xh = (v.unsqueeze(-1) * sae.W_dec[i]).sum(1) + sae.b_dec
                st["num"] += ((xc - xh) ** 2).sum().item()
                st["sx"] = st["sx"] + xc.sum(0)
                st["sxx"] += (xc ** 2).sum().item()
    n = st["n"]
    fvu = st["num"] / (st["sxx"] - n * ((st["sx"] / n) ** 2).sum().item())
    r = dict(sae="piggen_topk_r32_layer3", dict=D, n_sequences=len(seqs), n_tokens=n, l0=st["l0"] / n, fvu=fvu,
             dead_frac=(fire == 0).float().mean().item())
    print(r)
    os.makedirs(out.parent, exist_ok=True)
    pd.DataFrame([r]).to_csv(out, index=False)


if __name__ == "__main__":
    main()
