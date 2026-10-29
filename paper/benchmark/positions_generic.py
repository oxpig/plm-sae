"""Start positions of the p-IgGen latents whose target is not a heavy J gene (light-chain type, developability) and of
the TopK SAE of Section 4.1 (its three highest-F1 latents per J gene and the case-study latent 463), measured on 3,000
unsteered p-IgGen sequences (libraries/piggen_alpha0_reference, seeds 0-2) with the statistics of
paper.benchmark.positions. Matched SAEs read the final-layer-norm output / norm factor; the Section 4.1 SAE reads the
output of block 3.
Run: python -m paper.benchmark.positions_generic   (GPU) -> <work_dir>/benchmark/positions/piggen_generic_positions.csv"""
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from paper.benchmark import _inputs

def jobs():
    """(label, arch, site, sae, target, latent) of every latent."""
    out = []
    for arch, sae in (("TopK", "piggen_topk_r8"), ("Batch TopK", "piggen_batchtopk_r8"), ("Ordered", "piggen_ordered_r8")):
        for cls, lats in ranked_lct(sae).items():
            out += [("lct", arch, "post", sae, cls, l) for l in lats]
    out += [("dev", "TopK", "post", "piggen_topk_r8", "dev", 980), ("dev", "Batch TopK", "post", "piggen_batchtopk_r8", "dev", 1088)]
    out += [("section41", "TopK exp32", "blockout", "piggen_topk_r32_layer3", g, l) for g, l, _ in _inputs.section41_latents()]
    return out


def ranked_lct(sae):
    out = {}
    for line in open(_inputs.ranking(f"pairs_{sae}_lct.txt")):
        if ":" in line:
            g, i = line.strip().split(":")
            out.setdefault(g, []).append(int(i))
    return out


def main():
    dev = torch.device("cuda")
    out = P.work("benchmark", "positions")
    os.makedirs(out, exist_ok=True)
    seqs = [l.strip() for f in _inputs.alpha0_reference() for l in open(f) if l.strip()]
    plm, tok = models.load("piggen", dev)
    lnf, blk = plm.gpt_neox.final_layer_norm, plm.gpt_neox.layers[3]
    J = jobs()
    saes = {sae: S.load(sae).to(dev) for _, _, _, sae, _, _ in J}
    peaks, onsets, nfire = ({j: [] for j in range(len(J))}, {j: [] for j in range(len(J))}, {j: 0 for j in range(len(J))})
    for b in range(0, len(seqs), 32):
        tk = {k: v.to(dev) for k, v in models.tokenize("piggen", tok, seqs[b:b + 32], max_len=300).items()}
        st = {}
        h1 = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("post", o.detach().float()))
        h2 = blk.register_forward_hook(lambda m, i, o: st.__setitem__("blockout", (o[0] if isinstance(o, tuple) else o).detach().float()))
        with torch.no_grad():
            plm(**tk)
        h1.remove()
        h2.remove()
        msk, cache = tk["attention_mask"].bool(), {}
        for j, (_, _, site, sae, _, lat) in enumerate(J):
            if sae not in cache:
                x = st["post"] / models.NORM_FACTOR["piggen"] if site == "post" else st["blockout"]
                with torch.no_grad():   # one sequence per encode() call
                    z = torch.stack([saes[sae].encode(x[i]) for i in range(x.shape[0])])
                cache[sae] = (z.clamp(min=0) * msk[..., None]).cpu()
            zz = cache[sae][:, :, lat].numpy()
            zf = zz[zz.max(1) > 0]
            nfire[j] += len(zf)
            peaks[j] += zf.argmax(1).tolist()
            onsets[j] += (zf >= 0.1 * zf.max(1, keepdims=True)).argmax(1).tolist()
    rows = []
    for j, (label, arch, site, _, tgt, lat) in enumerate(J):
        q = np.percentile(peaks[j], [5, 95]).astype(int) if peaks[j] else [-1, -1]
        rows.append(dict(label=label, arch=arch, site=site, gene=tgt, latent=lat, n_seqs=len(seqs),
                         frac_fire=round(nfire[j] / len(seqs), 3), peak_p05=q[0], peak_p95=q[1],
                         onset_p05=int(np.percentile(onsets[j], 5)) if onsets[j] else -1))
    df = pd.DataFrame(rows)
    df.to_csv(out / "piggen_generic_positions.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
