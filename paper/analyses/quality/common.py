"""Steered generation metrics (Table 3, Supplementary Tables 10 and 11): per sequence, the mean VH (and VL) Hamming
distance to the first 1,000 sequences of the paired OAS test set (IMGT-aligned), diversity (nearest-neighbour 3-mer
cosine distance), the ESM-2 (35M) mean per-residue log-likelihood and the IgLM log-likelihood (heavy plus light chain
for paired sequences); two-sided Mann-Whitney U test and rank-biserial r = 1 - 2U/(n1 n2) against a reference.
B = <work_dir>/quality, the working directory of the analysis."""
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from plm_sae import paths as P
from paper.analyses.quality.align import chain_to_aligned   # noqa: F401 (used by run_one)
from paper.analyses.quality.seq_metrics import compute_pairwise_cosine_distances, hamming_dist

B = P.work("quality")
ESM2 = "facebook/esm2_t12_35M_UR50D"
ESM2_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"


def first1000(split):
    """First 1,000 rows of the aligned paired OAS test or training set."""
    return pd.read_csv(P.dataset(f"paired_oas_human_{split}_aligned.csv")).iloc[:1000]


def esm_ll(seqs, bs=64):
    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(ESM2, revision=ESM2_REVISION)
    m = EsmForMaskedLM.from_pretrained(ESM2, revision=ESM2_REVISION).eval()
    out = []
    for i in range(0, len(seqs), bs):
        x = tok(seqs[i:i + bs], return_tensors="pt", padding=True)
        with torch.no_grad():
            lp = torch.log_softmax(m(**x).logits, -1)
        for j in range(lp.shape[0]):
            k = x["input_ids"][j] != tok.pad_token_id
            out.append(lp[j, k, x["input_ids"][j, k]].mean().item())
    return np.array(out)


def iglm_ll(heavy, light=None):
    from iglm import IgLM
    g, out = IgLM(model_name="IgLM"), []
    for i, h in enumerate(heavy):
        try:
            out.append(g.log_likelihood(h, "[HEAVY]", "[HUMAN]") +
                       (g.log_likelihood(light[i], "[LIGHT]", "[HUMAN]") if light is not None else 0.0))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def per_sequence(vh_al, vl_al, vh, vl):
    """Per-sequence metric arrays; vl_al and vl None for heavy chains only."""
    test = first1000("test")
    d = {"VH_hamming_to_test": hamming_dist(vh_al, test["aligned_VHorVHH"].tolist()).mean(axis=1)}
    if vl_al is not None:
        d["VL_hamming_to_test"] = hamming_dist(vl_al, test["aligned_VL"].tolist()).mean(axis=1)
    txt = [h + l for h, l in zip(vh, vl)] if vl is not None else list(vh)
    d["diversity"] = np.asarray(compute_pairwise_cosine_distances(txt))
    d["ESM2_loglikelihood"] = esm_ll(txt)
    d["IgLM_loglikelihood"] = iglm_ll(vh, vl)
    return d


def mwu(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return dict(n_steered=len(x), n_ref=len(y), mean_steered=float(x.mean()), mean_ref=float(y.mean()), U=float(u),
                p=float(p), rank_biserial=float(1 - 2 * u / (len(x) * len(y))))
