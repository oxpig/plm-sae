"""Output scores of every latent of the three hyperparameter-matched SAEs of one base model (Methods, selection by
output effect).

  ll_pred   logit-lens score: the change in J-gene log-odds per unit alpha predicted from the decoder vector. The unit
            decoder vector d is projected onto the 20 amino-acid rows of the unembedding (W_U d), centred over amino
            acids, summed over the gene's germline J residues and centred over the six genes.
  seq_freq  share of the selection pool (plm_sae.steering.selection_pool) in which the latent fires at any position;
            output selection keeps latents with seq_freq >= 0.01 (plm_sae.steering.pick_output).
Also written, per latent and gene: firing counts at the tokens that predict the gene's CDR3-end residues (the m_g
residues before the FR4 W of the last W[GA]xG motif, m_g = length of the germline motif: J1 AEYFQH 6, J2 YWYFDL 6,
J3 DAFDI 5, J4 YFDY 4, J5 NWFDP 5, J6 YYYYYGMDV 9) and sequence-level precision, recall and F1 of the gene call.

Run: python -m plm_sae.steering.output_scores MODEL   -> <work_dir>/benchmark/probe/scores_<MODEL>.csv  (GPU)"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from plm_sae import models, paths as P, sae as S
from plm_sae.steering.selection_pool import validation_pool

AA = list("ACDEFGHIKLMNPQRSTVWY")
GENES = [f"IGHJ{i}" for i in range(1, 7)]
MOTIF_LEN = {"IGHJ1": 6, "IGHJ2": 6, "IGHJ3": 5, "IGHJ4": 4, "IGHJ5": 5, "IGHJ6": 9}
JSEQ = {"IGHJ1": "AEYFQHWGQGTLVTVSS", "IGHJ2": "YWYFDLWGRGTLVTVSS", "IGHJ3": "DAFDIWGQGTMVTVSS",
        "IGHJ4": "YFDYWGQGTLVTVSS", "IGHJ5": "NWFDPWGQGTLVTVSS", "IGHJ6": "YYYYYGMDVWGQGTTVTVSS"}
SAES = {"piggen": [("TopK", "piggen_topk_r8"), ("Batch TopK", "piggen_batchtopk_r8"), ("Ordered", "piggen_ordered_r8")],
        "IgLM": [("TopK", "iglm_topk_r8"), ("Batch TopK", "iglm_batchtopk_r8"), ("Ordered", "iglm_ordered_r8")],
        "progen2_oas": [("TopK", "progen2oas_topk_r8"), ("Batch TopK", "progen2oas_batchtopk_r8"),
                        ("Ordered", "progen2oas_ordered_r8")]}
OFFSET = {"piggen": 1, "IgLM": 2, "progen2_oas": 1}   # tokens before the first residue


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=models.MODELS)
    a = ap.parse_args()
    model = a.model
    out = P.work("benchmark", "probe")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plm, tok = models.load(model, dev)
    core = plm.gpt_neox if hasattr(plm, "gpt_neox") else plm.transformer
    lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
    emb = plm.get_output_embeddings() if plm.get_output_embeddings() is not None else plm.lm_head
    WU = emb.weight.detach().float()[[tok.convert_tokens_to_ids(c) for c in AA]].to(dev)   # [20, d_model]
    val = validation_pool()

    saes = [(arch, name, S.load(name).to(dev), S.decoder(name)) for arch, name in SAES[model]]
    acc = {arch: dict(tp=torch.zeros(6, W.shape[0], device=dev), fire=torch.zeros(W.shape[0], device=dev),
                      sfire=torch.zeros(6, W.shape[0], device=dev)) for arch, _, _, W in saes}
    npos, nseq, nskip = np.zeros(6), np.zeros(6), 0
    seqs, gidx = val.VHorVHH.astype(str).tolist(), val.gene.map(GENES.index).tolist()
    for b in range(0, len(seqs), 64):
        Sb, G = seqs[b:b + 64], gidx[b:b + 64]
        tk = {k: v.to(dev) for k, v in models.tokenize(model, tok, Sb).items()}
        st = {}
        h = lnf.register_forward_hook(lambda m, i, o: st.__setitem__("x", o.detach().float()))
        with torch.no_grad():
            plm(**tk)
        h.remove()
        x, msk = st["x"] / models.NORM_FACTOR[model], tk["attention_mask"].bool()
        T = x.shape[1]
        pos = torch.zeros(len(Sb), T, dtype=torch.bool, device=dev)
        for j, (s, g) in enumerate(zip(Sb, G)):
            m = list(re.finditer(r"W[GA].G", s))
            if not m:
                nskip += 1
                continue
            w = m[-1].start()
            r0 = max(0, w - MOTIF_LEN[GENES[g]])
            t0, t1 = OFFSET[model] + r0 - 1, OFFSET[model] + w - 1   # tokens predicting residues r0 .. w-1
            if t1 > T or t0 < 0:
                nskip += 1
                continue
            pos[j, t0:t1] = True
            npos[g] += t1 - t0
        for g in G:
            nseq[g] += 1
        Gt = torch.tensor(G, device=dev)
        for arch, _, sae, _ in saes:
            with torch.no_grad():   # one sequence per encode() call (batch-pooled classes)
                z = torch.stack([sae.encode(x[j]) for j in range(x.shape[0])]).clamp(min=0)
            f = (z > 0) & msk[..., None]                     # [B, T, L]
            ac = acc[arch]
            ac["fire"] += f.sum((0, 1))
            sf = f.any(1).float()                            # [B, L] fires anywhere
            tpb = (f & pos[..., None]).sum(1).float()        # [B, L]
            for g in range(6):
                sel = Gt == g
                if sel.any():
                    ac["tp"][g] += tpb[sel].sum(0)
                    ac["sfire"][g] += sf[sel].sum(0)
        print(b, flush=True)

    rows, ai = [], {c: k for k, c in enumerate(AA)}
    for arch, name, _, Wd in saes:
        ac = {k: v.cpu().numpy() for k, v in acc[arch].items()}
        d = (Wd / Wd.norm(dim=1, keepdim=True)).to(dev)
        bias = (d @ WU.T).cpu().numpy()
        bias -= bias.mean(1, keepdims=True)                                       # [L, 20]
        pred = np.stack([bias[:, [ai[c] for c in JSEQ[g]]].sum(1) for g in GENES], 1)
        pred -= pred.mean(1, keepdims=True)                                       # [L, 6]
        sf_all = ac["sfire"].sum(0)

        def ratio(n, d_):
            return np.divide(n, d_, out=np.zeros_like(n), where=d_ > 0)
        for g in range(6):
            tp = ac["tp"][g]
            prec, rec = ratio(tp, ac["fire"]), tp / max(npos[g], 1)
            sp, sr = ratio(ac["sfire"][g], sf_all), ac["sfire"][g] / nseq[g]
            rows.append(pd.DataFrame(dict(
                model=model, arch=arch, ckpt=name, gene=GENES[g], latent=np.arange(len(tp)), live=ac["fire"] > 0,
                fire_tokens=ac["fire"], pre_tp=tp, pre_prec=prec, pre_rec=rec, pre_f1=ratio(2 * prec * rec, prec + rec),
                seq_prec=sp, seq_rec=sr, seq_f1=ratio(2 * sp * sr, sp + sr), seq_freq=sf_all / nseq.sum(),
                ll_pred=pred[:, g], ll_norm=np.linalg.norm(bias, axis=1),
                ll_top=["".join(np.array(AA)[np.argsort(-bias[i])[:4]]) for i in range(len(tp))])))
    pd.concat(rows).to_csv(f"{out}/scores_{model}.csv", index=False)
    print("DONE", model, "skipped", nskip, "positives per gene", npos.tolist())


if __name__ == "__main__":
    main()
