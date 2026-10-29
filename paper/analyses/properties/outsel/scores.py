"""Latent statistics for steering CDRH3 properties and developability metrics in p-IgGen (Supplementary Sections A.10
and A.12). GPU, one job. Per latent of the three hyperparameter-matched p-IgGen SAEs (final-layer-norm output / norm
factor, one sequence per encode() call):
  latent_positions.csv  on 3,000 validation heavy chains (random_state 0, the sample of ident_simple), the 5th
                        percentile of the per-sequence onset and peak positions over the sequences in which the latent
                        fires, for all sequences (subset all) and for those with each CDRH3 property (Yrich, positive,
                        long; paper.analyses.properties.props)
  pool_stats.csv        firing rate on the selection pool (plm_sae.steering.selection_pool), per gene and overall, and
                        the largest per-gene enrichment: the eligibility statistics of the baseline latents
  out_scores.csv        output scores: b = W_U d (unit decoder vector, the 20 amino-acid rows of the unembedding,
                        centred over them); per target the bias in the steered direction (Yrich b[Y]; positive and
                        SFvCSP mean(b[K], b[R]) - mean(b[D], b[E]); long -b[W]; PPC -mean(b[K], b[R]); PNC
                        -mean(b[D], b[E]); PSH -mean(b[A, I, L, M, F, V, W, Y]))
Output: <work_dir>/outsel_properties/scores/. Usage: python -m paper.analyses.properties.outsel.scores"""
import os

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from plm_sae.steering.selection_pool import GENES, validation_pool
from paper.analyses.properties.props import props

SAES = [("TopK", "piggen_topk_r8"), ("Batch TopK", "piggen_batchtopk_r8"), ("Ordered", "piggen_ordered_r8")]
AA = list("ACDEFGHIKLMNPQRSTVWY")


def main():
    out = P.work("outsel_properties", "scores")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")
    nf = models.NORM_FACTOR["piggen"]
    plm, tok = models.load("piggen", dev)
    lnf = plm.gpt_neox.final_layer_norm
    WU = plm.get_output_embeddings().weight.detach().float()[[tok.convert_tokens_to_ids(a) for a in AA]].to(dev)
    saes = [(arch, S.load(name).to(dev), S.decoder(name)) for arch, name in SAES]
    NL = saes[0][2].shape[0]

    def run(seqs, bs=32):
        """Per SAE: per-sequence maximum, peak position and onset position of every latent, [sequences, latents]."""
        res = {a: dict(mx=[], pk=[], on=[]) for a, _, _ in saes}
        for b in range(0, len(seqs), bs):
            tk = {k: v.to(dev) for k, v in models.tokenize("piggen", tok, seqs[b:b + bs]).items()}
            st = {}
            h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
            with torch.no_grad():
                plm(**tk)
            h.remove()
            x, msk = st["x"] / nf, tk["attention_mask"].bool()
            for arch, sae, _ in saes:
                with torch.no_grad():
                    z = torch.stack([sae.encode(x[j]) for j in range(x.shape[0])]).clamp(min=0) * msk[..., None]
                    mx, pk = z.max(1)
                    on = (z >= 0.1 * mx[:, None, :]).int().argmax(1)
                res[arch]["mx"].append(mx.cpu().numpy().astype(np.float32))
                res[arch]["pk"].append(pk.cpu().numpy().astype(np.int16))
                res[arch]["on"].append(on.cpu().numpy().astype(np.int16))
        return {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in res.items()}

    def pct(v):
        return int(np.percentile(v, 5)) if len(v) else -1

    # positions on 3,000 validation heavy chains
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VHorVHH", "j_gene_anarci_heavy"])
    val["gene"] = val.j_gene_anarci_heavy.astype(str).str.split("*").str[0]
    seqs = val.sample(3000, random_state=0).VHorVHH.astype(str).tolist()
    pr = [props(s) for s in seqs]
    sub = {"all": np.ones(len(seqs), bool), "Yrich": np.array([bool(p) and p["Y"] >= 0.2 for p in pr]),
           "positive": np.array([bool(p) and p["charge"] >= 1 for p in pr]),
           "long": np.array([bool(p) and p["len"] >= 18 for p in pr])}
    r3, rows = run(seqs), []
    for arch, _, _ in saes:
        fire = r3[arch]["mx"] > 0
        for name, m in sub.items():
            for l in range(NL):
                f = fire[:, l] & m
                rows.append(dict(arch=arch, subset=name, latent=l, n_seqs=int(m.sum()), n_fire=int(f.sum()),
                                 onset_p05=pct(r3[arch]["on"][f, l]), peak_p05=pct(r3[arch]["pk"][f, l])))
    pd.DataFrame(rows).to_csv(out / "latent_positions.csv", index=False)

    # firing statistics on the selection pool
    pool, fire_g, rows = validation_pool(val), {}, []
    for g, dg in pool.groupby("gene"):
        for arch, r in run(dg.VHorVHH.astype(str).tolist()).items():
            fire_g[(arch, g)] = (r["mx"] > 0).mean(0)
    for arch, _, _ in saes:
        rate_g = np.stack([fire_g[(arch, g)] for g in GENES])
        rate = rate_g.mean(0)
        rows.append(pd.DataFrame(dict(arch=arch, latent=np.arange(NL), rate=rate,
                                      max_enrichment=(rate_g / np.maximum(rate, 1e-9)).max(0),
                                      **{f"rate_{g}": rate_g[i] for i, g in enumerate(GENES)})))
    pd.concat(rows).to_csv(out / "pool_stats.csv", index=False)

    # output scores
    ai, rows = {c: k for k, c in enumerate(AA)}, []
    for arch, _, W in saes:
        b = ((W / W.norm(dim=1, keepdim=True)).to(dev) @ WU.T).cpu().numpy()
        b -= b.mean(1, keepdims=True)
        KR, DE = (b[:, ai["K"]] + b[:, ai["R"]]) / 2, (b[:, ai["D"]] + b[:, ai["E"]]) / 2
        HYD = b[:, [ai[c] for c in "AILMFVWY"]].mean(1)
        rows.append(pd.DataFrame(dict(arch=arch, latent=np.arange(NL), Yrich=b[:, ai["Y"]], positive=KR - DE,
                                      long=-b[:, ai["W"]], PPC=-KR, PNC=-DE, PSH=-HYD, SFvCSP=KR - DE)))
    pd.concat(rows).to_csv(out / "out_scores.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
