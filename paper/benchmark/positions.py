"""Start positions of the benchmark latents: where each latent fires in natural heavy chains of its gene.

Latents: the top 5 per J gene of each matched SAE (identification/pairs) and, for the Ordered SAE, the low-index
latents (identification/lowidx/selection.csv). Up to 1,500 validation heavy chains per gene (random_state 0) are
encoded as in feature identification (final-layer-norm output / norm factor, one sequence per encode() call); positions
are token indices of the generated sequence, so they can gate steering directly.
Per latent: frac_fire (share of sequences in which it fires), the 5th and 95th percentiles of the per-sequence peak
position (peak_p05, peak_p95) and the 5th percentile of the per-sequence onset (onset_p05; onset = first position at
which the latent reaches 10% of that sequence's maximum). The benchmark steers from max(0, onset_p05 - 5).

Run: python -m paper.benchmark.positions MODEL   (GPU) -> <work_dir>/benchmark/positions/<MODEL>_positions.csv"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark import _inputs

SAES = {m: [("TopK", f"{p}_topk_r8"), ("Batch TopK", f"{p}_batchtopk_r8"), ("Ordered", f"{p}_ordered_r8")]
        for m, p in (("piggen", "piggen"), ("IgLM", "iglm"), ("progen2_oas", "progen2oas"))}


def ranked(sae):
    """{gene: [latents in F1 rank order]} of the SAE's top-5 list."""
    out = {}
    for line in open(_inputs.ranking(f"pairs_{sae}_top5.txt")):
        if ":" in line:
            g, i = line.strip().split(":")
            out.setdefault(g, []).append(int(i))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("--per_gene", type=int, default=1500)
    a = ap.parse_args()
    out = P.work("benchmark", "positions")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")
    plm, tok = models.load(a.model, dev)
    core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
    lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH", "j_gene_anarci_heavy"])
    val["gene"] = val.j_gene_anarci_heavy.astype(str).str.split("*").str[0]
    sel = pd.read_csv(_inputs.lowidx_selection())
    saes, lowidx = [], set()
    for arch, name in SAES[a.model]:
        lat = {g: v[:5] for g, v in ranked(name).items()}
        if arch == "Ordered":
            lowidx = sorted(set((r.gene, int(r.latent)) for r in sel[sel.sae == name].itertuples()))
            for g, i in lowidx:
                if i not in lat.setdefault(g, []):
                    lat[g].append(i)
        saes.append((arch, S.load(name).to(dev), lat))
    rows = []
    for gene in sorted(val.gene.unique()):
        if not gene.startswith("IGHJ") or gene not in saes[0][2]:
            continue
        seqs = val[val.gene == gene].VHorVHH.astype(str).sample(n=min(a.per_gene, (val.gene == gene).sum()),
                                                                 random_state=0).tolist()
        peaks = {(arch, i): [] for arch, _, lat in saes for i in lat.get(gene, [])}
        onsets = {k: [] for k in peaks}
        nfire = {k: 0 for k in peaks}
        for b in range(0, len(seqs), 64):
            tk = {k: v.to(dev) for k, v in models.tokenize(a.model, tok, seqs[b:b + 64]).items()}
            st = {}
            h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
            with torch.no_grad():
                plm(**tk)
            h.remove()
            x, msk = st["x"] / models.NORM_FACTOR[a.model], tk["attention_mask"].bool()
            for arch, sae, lat in saes:
                ids = lat.get(gene, [])
                if not ids:
                    continue
                with torch.no_grad():
                    z = torch.stack([sae.encode(x[j]) for j in range(x.shape[0])])[..., ids]
                z = (z.clamp(min=0) * msk[..., None]).cpu().numpy()   # [B, T, latents]
                for k, i in enumerate(ids):
                    zz = z[:, :, k]
                    zf = zz[zz.max(1) > 0]
                    peaks[(arch, i)] += zf.argmax(1).tolist()
                    onsets[(arch, i)] += (zf >= 0.1 * zf.max(1, keepdims=True)).argmax(1).tolist()
                    nfire[(arch, i)] += len(zf)
        for (arch, i), pk in peaks.items():
            lat = saes[[s[0] for s in saes].index(arch)][2][gene]
            rank = lat.index(i) + 1
            q = np.percentile(pk, [5, 95]).astype(int) if pk else [-1, -1]
            rows.append(dict(model=a.model, arch=arch, gene=gene, rank=rank if rank <= 5 else 0, latent=i,
                             n_seqs=len(seqs), frac_fire=round(nfire[(arch, i)] / len(seqs), 3), peak_p05=q[0],
                             peak_p95=q[1], lowidx=arch == "Ordered" and (gene, i) in lowidx,
                             onset_p05=int(np.percentile(onsets[(arch, i)] or [-1], 5))))   # -1: never fires
        print(gene, "done", flush=True)
    df = pd.DataFrame(rows).sort_values(["arch", "gene", "rank"])
    df.to_csv(out / f"{a.model}_positions.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
