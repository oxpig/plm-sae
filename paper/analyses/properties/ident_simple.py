"""Sequence-level F1 of every latent of the three p-IgGen matched SAEs for the CDRH3 properties (Supplementary
Section A.12): 3,000 validation heavy chains (random_state 0) with a parsed CDRH3 (paper.analyses.properties.props)
through p-IgGen ('1' + VH + '2', final-layer-norm output / norm factor, one sequence per encode() call), labelled
Y-rich (tyrosine share >= 0.2), positive (K + R - D - E >= +1) and long (>= 18 residues). A latent predicts the property
if it fires at any of the tokens 0 .. len(VH).
Writes <work_dir>/outsel_properties/ident_scores.csv (arch, label, latent, frac_fire, f1, prec, rec); prints the share of
sequences with each property.
Usage: python -m paper.analyses.properties.ident_simple   (GPU or CPU)"""
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.analyses.properties.props import props

SAES = [("TopK", "piggen_topk_r8"), ("Batch TopK", "piggen_batchtopk_r8"), ("Ordered", "piggen_ordered_r8")]


def main():
    out = P.work("outsel_properties")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plm, tok = models.load("piggen", dev)
    lnf = plm.gpt_neox.final_layer_norm
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH"]).VHorVHH.astype(str).sample(3000, random_state=0)
    pr = [props(s) for s in val]
    keep = [i for i, p in enumerate(pr) if p]
    seqs, pr = [val.iloc[i] for i in keep], pd.DataFrame([pr[i] for i in keep])
    lab = {"Yrich": (pr.Y >= 0.2).values, "positive": (pr.charge >= 1).values, "long": (pr.len >= 18).values}
    print({k: round(v.mean(), 3) for k, v in lab.items()}, len(seqs), "sequences with a parsed CDRH3", flush=True)
    saes = [(arch, S.load(name).to(dev)) for arch, name in SAES]
    mx = {a: [] for a, _ in saes}
    for b in range(0, len(seqs), 32):
        Sb = seqs[b:b + 32]
        tk = {k: v.to(dev) for k, v in models.tokenize("piggen", tok, Sb, max_len=200).items()}
        st = {}
        h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
        with torch.no_grad():
            plm(**tk)
        h.remove()
        x = st["x"] / models.NORM_FACTOR["piggen"]
        for arch, sae in saes:
            for j, s in enumerate(Sb):
                with torch.no_grad():
                    mx[arch].append(sae.encode(x[j, :1 + len(s)]).clamp(min=0).max(0).values.cpu().numpy())
    rows = []
    for arch, _ in saes:
        fire = np.stack(mx[arch]) > 0
        nf = fire.sum(0)
        for name, y in lab.items():
            tp = (fire & y[:, None]).sum(0)
            prec = np.divide(tp, nf, out=np.zeros(fire.shape[1]), where=nf > 0)
            rec = tp / y.sum()
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(fire.shape[1]), where=(prec + rec) > 0)
            rows.append(pd.DataFrame(dict(arch=arch, label=name, latent=np.arange(fire.shape[1]), frac_fire=fire.mean(0),
                                          f1=f1, prec=prec, rec=rec)))
    pd.concat(rows).to_csv(out / "ident_scores.csv", index=False)
    print("wrote", out / "ident_scores.csv")


if __name__ == "__main__":
    main()
