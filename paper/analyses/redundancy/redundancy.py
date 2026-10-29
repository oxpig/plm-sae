"""Dictionary redundancy of the nine hyperparameter-matched SAEs (Supplementary Table 16, Section 4.2.2, Supplementary
Section A.8). Dead latents are those never nonzero at a real-token position of the 30,754 validation sequences (the
'<arch>|real' firing counts of identification/reconstruction/fire_counts_<model>.npz, plm_sae.training.reconstruction).
Geometric redundancy: median over non-dead latents of the largest decoder cosine to any other non-dead latent.
Co-firing redundancy: each latent is binarised at the threshold of its best-F1 gene in the J-gene identification gate
(gate scaler, clipped to [0, 1]) on a 10,000-sequence seed-42 sample of the validation set; per latent the largest
positive phi to any other latent, over non-dead latents that cross their threshold at least once in the sample.
Inputs: identification/gates/<sae>/ (best_model_pipeline.pkl, f1_scores/), <work_dir>/activations/<sae>/val/, the SAE
checkpoints and the firing counts.
Writes <work_dir>/redundancy/redundancy.csv. Usage: python -m paper.analyses.redundancy.redundancy   (CPU)"""
import pickle

import numpy as np
import pandas as pd
import torch

from plm_sae import paths as P
from plm_sae import sae as S
from plm_sae.features.gate import load_latents
from paper.benchmark._inputs import inp

# (model, architecture, SAE, firing-count file, firing-count key)
SAES = [
    ("p-IgGen", "TopK", "piggen_topk_r8", "piggen", "TopK"),
    ("p-IgGen", "Batch TopK", "piggen_batchtopk_r8", "piggen", "Batch_TopK"),
    ("p-IgGen", "Ordered", "piggen_ordered_r8", "piggen", "Ordered"),
    ("IgLM", "TopK", "iglm_topk_r8", "IgLM", "TopK"),
    ("IgLM", "Batch TopK", "iglm_batchtopk_r8", "IgLM", "Batch_TopK"),
    ("IgLM", "Ordered", "iglm_ordered_r8", "IgLM", "Ordered"),
    ("ProGen2-OAS", "TopK", "progen2oas_topk_r8", "progen2_oas", "TopK"),
    ("ProGen2-OAS", "Batch TopK", "progen2oas_batchtopk_r8", "progen2_oas", "Batch_TopK"),
    ("ProGen2-OAS", "Ordered", "progen2oas_ordered_r8", "progen2_oas", "Ordered"),
]
N_SAMPLE = 10000
GENES = [f"IGHJ{i}" for i in range(1, 7)]


def decoder_redundancy(name, dead):
    """Number of non-dead latents and the median of their largest decoder cosine to any other non-dead latent."""
    W = S.decoder(name)
    W = (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12))[torch.from_numpy(~dead)]
    mx = torch.empty(W.shape[0])
    for i in range(0, W.shape[0], 2048):
        c = W[i:i + 2048] @ W.T
        j = torch.arange(c.shape[0])
        c[j, j + i] = -np.inf
        mx[i:i + 2048] = c.max(1).values
    return W.shape[0], float(mx.median())


def cofiring_redundancy(name, dead):
    """Dictionary size, number of live latents and, per live latent, its largest phi to any other live latent."""
    gate = inp("identification", "gates", name)
    X, _ = load_latents(P.activations(name, "val"), sample_n=N_SAMPLE, seed=42)
    sc = pickle.load(open(gate / "best_model_pipeline.pkl", "rb")).named_steps["scaler"]
    X = np.clip(sc.transform(X), 0, 1)
    n_lat = X.shape[1]
    best = np.full(n_lat, -np.inf, np.float32)
    thr = np.full(n_lat, np.inf, np.float32)
    seen = np.zeros(n_lat, bool)
    for g in GENES:
        f = pd.read_csv(gate / "f1_scores" / f"f1_scores_class_{g}.csv")
        i = f.feature_idx.to_numpy()
        b = f.best_f1.to_numpy() > best[i]
        best[i[b]] = f.best_f1.to_numpy()[b]
        thr[i[b]] = f.best_threshold.to_numpy()[b]
        seen[i] = True
    B = (X > thr[None, :]).astype(np.float32)
    fc = B.sum(0)
    live = seen & ~dead & (fc >= 1) & (fc < N_SAMPLE)
    Z = B[:, live]
    Z = (Z - Z.mean(0)) / Z.std(0)
    C = Z.T @ Z / N_SAMPLE
    np.fill_diagonal(C, 0)
    return n_lat, int(live.sum()), C.max(1)


def main():
    rows = []
    for model, arch, name, fc_file, fc_key in SAES:
        dead = np.load(inp("identification", "reconstruction", f"fire_counts_{fc_file}.npz"))[f"{fc_key}|real"] == 0
        n_dec, dec_cos = decoder_redundancy(name, dead)
        n_lat, n_cofire, m = cofiring_redundancy(name, dead)
        rows.append(dict(model=model, arch=arch, dict=n_lat, n_dead_real=int(dead.sum()), n_dec=n_dec, dec_cos_med=dec_cos,
                         n_cofire=n_cofire, median_max_phi=float(np.median(m)), frac_phi_ge_0p5=float((m >= 0.5).mean()),
                         frac_phi_ge_0p7=float((m >= 0.7).mean())))
        print(rows[-1], flush=True)
    out = P.work("redundancy")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "redundancy.csv", index=False)


if __name__ == "__main__":
    main()
