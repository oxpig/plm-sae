"""Developability gate of one hyperparameter-matched SAE (Supplementary Section A.10, Supplementary Table 19): the
Pearson correlation of every latent's pooled activation (mean over the positions of each of the 30,754 validation
sequences; <work_dir>/activations/<sae>/val/) with each FlashTAP metric of the paired sequence
(developability/flashtap/flashtap_val_scores.csv, paper.analyses.properties.developability.flashtap_val), with its
two-sided p-value. Latents with a constant pooled activation are flagged dead.
Writes <work_dir>/developability/gate/<sae>/developability_<metric>_per_latent.csv (latent, pearson_r, pearson_p, dead),
sorted by |r|. Usage: python -m paper.analyses.properties.developability.gate SAE   (CPU)"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from plm_sae import paths as P
from plm_sae.features.gate import load_latents
from paper.benchmark._inputs import inp

METRICS = ["PSH", "PPC", "PNC", "SFvCSP"]


def pearson(Z, y):
    """Per-column Pearson r of Z [n, latents] with y, two-sided p (t test); NaN for constant columns."""
    n = len(y)
    Zc, yc = Z - Z.mean(axis=0, keepdims=True), y - y.mean()
    den = np.sqrt((Zc ** 2).sum(axis=0) * (yc ** 2).sum())
    r = (Zc.T @ yc) / np.where(den == 0, np.nan, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return r, 2 * stats.t.sf(np.abs(t), df=n - 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sae")
    a = ap.parse_args()
    out = P.work("developability", "gate", a.sae)
    os.makedirs(out, exist_ok=True)
    Z, rows = load_latents(P.activations(a.sae, "val"))
    assert rows == list(range(len(rows))), "validation latents must cover orig_row 0 .. n-1"
    labels = pd.read_csv(inp("developability", "flashtap", "flashtap_val_scores.csv"))
    Z = Z[labels.orig_row.to_numpy()]
    for m in METRICS:
        ok = labels[m].notna().to_numpy()
        r, p = pearson(Z[ok], labels.loc[ok, m].to_numpy(dtype=float))
        d = pd.DataFrame(dict(latent=np.arange(Z.shape[1]), pearson_r=r, pearson_p=p, dead=np.isnan(r)))
        d.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False).to_csv(
            out / f"developability_{m}_per_latent.csv", index=False)
        print(m, "latents with |r| >= 0.40 (Bonferroni):", int(((d.pearson_r.abs() >= 0.4) & (d.pearson_p < 0.05 / len(d))).sum()))


if __name__ == "__main__":
    main()
