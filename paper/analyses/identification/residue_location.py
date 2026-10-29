"""Where on the heavy chain latents of the TopK SAE of Section 4.1 fire (Supplementary Section A.5: IGHJ4 latents 463,
4720 and 6276 peak at IMGT 120, 119 and 117).
  measure   4,000 validation heavy chains (random_state 0) with a J call, IMGT-numbered with ANARCI, encoded as
            '1' + VH + '2' by p-IgGen; the SAE reads the output of block 3. Per gene, IMGT position (0-128) and latent,
            the sum of the top-32 activations -> <work_dir>/identification/residue_location/residue_location.npz (M)
  report    per latent, the IMGT position holding most of its activation over the IGHJ4 sequences and its share
Usage: python -m paper.analyses.identification.residue_location measure [N]   (GPU; ANARCI on 8 CPUs)
       python -m paper.analyses.identification.residue_location report [--latents 463 4720 6276]"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark._inputs import inp

GENES = ["IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6"]
PN = 129


def measure(n):
    from anarci import anarci
    out = P.work("identification", "residue_location", "residue_location.npz")
    os.makedirs(out.parent, exist_ok=True)
    dev = "cuda"
    df = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["orig_row", "VHorVHH", "heavy_j_gene"])
    df = df.dropna().sample(n=min(n, len(df)), random_state=0).reset_index(drop=True)
    df["gene"] = df.heavy_j_gene.str.extract(r"(IGHJ\d)")[0]
    df = df[df.gene.notna()].reset_index(drop=True)
    nums = anarci([(str(i), s) for i, s in enumerate(df.VHorVHH)], scheme="imgt", output=False, ncpu=8)[0]
    posmap, keep = [], []
    for i, (num, s) in enumerate(zip(nums, df.VHorVHH)):
        if num is None:
            continue
        nl, st, _ = num[0]
        pm, k = np.full(len(s), -1, np.int64), st
        for (p, _), aa in nl:
            if aa != "-":
                pm[k] = p
                k += 1
        posmap.append(pm)
        keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    model, tok = models.load("piggen", dev)
    sae = S.load("piggen_topk_r32_layer3").to(dev)
    D = sae.W_enc.shape[0]
    cap = []
    model.gpt_neox.layers[3].register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float()))
    M = torch.zeros(len(GENES) * PN * D, device=dev)
    gi = {g: i for i, g in enumerate(GENES)}
    for b in range(0, len(df), 32):
        sub = df.iloc[b:b + 32]
        t = models.tokenize("piggen", tok, list(sub.VHorVHH))
        ids, am = t["input_ids"].to(dev), t["attention_mask"].to(dev)
        start = am.argmax(1)
        pos = torch.full(ids.shape, -1, dtype=torch.long, device=dev)
        for j, (s, pm) in enumerate(zip(sub.VHorVHH, posmap[b:b + 32])):
            if int(am[j].sum()) == len(s) + 2:
                pos[j, start[j] + 1:start[j] + 1 + len(s)] = torch.tensor(pm, device=dev)
        pos = torch.where(pos > PN - 1, torch.full_like(pos, -1), pos)
        g = torch.tensor([gi[x] for x in sub.gene], device=dev)[:, None].expand_as(pos)
        valid = pos >= 0
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids, attention_mask=am)
            v, i = sae.topk(cap[0][valid])
            M.index_add_(0, (((g[valid] * PN + pos[valid]) * D)[:, None] + i).reshape(-1), v.reshape(-1))
    np.savez_compressed(out, M=M.view(len(GENES), PN, D).cpu().numpy())
    print("wrote", out)


def report(latents, gene="IGHJ4"):
    M = np.load(inp("identification", "residue_location", "residue_location.npz"))["M"][GENES.index(gene)]
    for l in latents:
        m = M[:, l]
        print(f"{gene} latent {l}: peak at IMGT {int(m.argmax())}, {100 * m.max() / m.sum():.1f}% of its activation")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["measure", "report"])
    ap.add_argument("n", nargs="?", type=int, default=4000)
    ap.add_argument("--latents", nargs="+", type=int, default=[463, 4720, 6276])
    a = ap.parse_args()
    measure(a.n) if a.mode == "measure" else report(a.latents)


if __name__ == "__main__":
    main()
