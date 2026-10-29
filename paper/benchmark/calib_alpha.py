"""Steering strengths across models (Supplementary Section A.2, "Steering strengths"). At the final-layer-norm output,
steering adds alpha * W_U d to the next-token logits, so alpha is compared across models by the spread of the amino-acid
logit shift per unit alpha, range = max - min over the 20 amino-acid tokens of W_U d, for the highest-F1 latent of every
SAE and gene. A vector added at the output of the last block (the Section 4.1 SAE) passes the final layer norm, which
divides by the hidden state's scale, so its strengths scale by |h_block output| / |h_final-layer-norm output|
(median over tokens of 64 validation heavy chains).
Output: <work_dir>/benchmark/calib_alpha.csv, printed medians and norm ratios.
Usage: python -m paper.benchmark.calib_alpha   (CPU)"""
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark.positions import SAES

AA = list("ACDEFGHIKLMNPQRSTVWY")


def main():
    F = P.work("benchmark")
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH"]).sample(n=200, random_state=0)
    val = val.VHorVHH.astype(str).tolist()
    rows, norms = [], {}
    for m in models.MODELS:
        plm, tok = models.load(m, "cpu")
        WU = (plm.get_output_embeddings() or plm.lm_head).weight.detach().float()
        WA = WU[tok.convert_tokens_to_ids(AA)]   # [20, d_model]
        core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
        lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
        last = (core.layers if hasattr(core, "layers") else core.h)[-1]
        st = {}
        h1 = lnf.register_forward_hook(lambda mod, i, o: st.__setitem__("post", o.detach()))
        h2 = last.register_forward_hook(lambda mod, i, o: st.__setitem__("blk", (o[0] if isinstance(o, tuple) else o).detach()))
        tk = models.tokenize(m, tok, val[:64])
        with torch.no_grad():
            plm(**tk)
        h1.remove()
        h2.remove()
        msk = tk["attention_mask"].bool()
        norms[m] = dict(post=float(st["post"][msk].norm(dim=-1).median()), blockout=float(st["blk"][msk].norm(dim=-1).median()))
        pos = pd.read_csv(F / f"positions/{m}_positions.csv")
        for arch, name in SAES[m]:
            W = S.decoder(name)
            for p in pos[(pos.arch == arch) & (pos["rank"] == 1)].itertuples():
                s = WA @ (W[int(p.latent)] / W[int(p.latent)].norm())
                rows.append(dict(model=m, arch=arch, gene=p.gene, latent=int(p.latent), range=float(s.max() - s.min())))
        del plm
    C = pd.DataFrame(rows)
    C.to_csv(F / "calib_alpha.csv", index=False)
    med = C.groupby("model")["range"].median()
    print("median amino-acid logit range per unit alpha:\n" + med.round(4).to_string())
    print("hidden-state norms (median per token):", norms)
    print(f"p-IgGen block output / final-layer-norm output norm ratio {norms['piggen']['blockout'] / norms['piggen']['post']:.2f}")


if __name__ == "__main__":
    main()
