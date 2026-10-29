"""Latents associated with each FlashTAP metric in the nine hyperparameter-matched SAEs (Supplementary Table 19):
|Pearson r| >= 0.40 (and >= 0.20) with Bonferroni-corrected p < 0.05 over the live latents. A latent is dead if it
never fires at a residue of the 30,754 validation sequences (firing counts '<SAE>|real' of
identification/reconstruction/fire_counts_<model>.npz, plm_sae.training.reconstruction).
Reads <work_dir>/developability/gate/<sae>/ (paper.analyses.properties.developability.gate).
Writes <work_dir>/developability/counts/dev_latent_counts.csv and dev_latents_r040.csv.
Usage: python -m paper.analyses.properties.developability.counts"""
import os

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

SAES = [("p-IgGen", "TopK", "piggen_topk_r8"), ("p-IgGen", "Batch TopK", "piggen_batchtopk_r8"),
        ("p-IgGen", "Ordered", "piggen_ordered_r8"), ("IgLM", "TopK", "iglm_topk_r8"),
        ("IgLM", "Batch TopK", "iglm_batchtopk_r8"), ("IgLM", "Ordered", "iglm_ordered_r8"),
        ("ProGen2-OAS", "TopK", "progen2oas_topk_r8"), ("ProGen2-OAS", "Batch TopK", "progen2oas_batchtopk_r8"),
        ("ProGen2-OAS", "Ordered", "progen2oas_ordered_r8")]
METRICS = ["PSH", "PNC", "PPC", "SFvCSP"]
FIRE = {"p-IgGen": "piggen", "IgLM": "IgLM", "ProGen2-OAS": "progen2_oas"}


def main():
    out = P.work("developability", "counts")
    rows, hits = [], []
    for model, arch, sae in SAES:
        dead = np.load(inp("identification", "reconstruction", f"fire_counts_{FIRE[model]}.npz"))[f"{arch.replace(' ', '_')}|real"] == 0
        r = dict(model=model, sae=arch)
        for m in METRICS:
            df = pd.read_csv(inp("developability", "gate", sae, f"developability_{m}_per_latent.csv"))
            live = df[~dead[df["latent"].values]]
            r["live"] = len(live)
            sig = live["pearson_p"] < 0.05 / len(live)
            for t in (0.40, 0.20):
                r[f"{m}_{int(t * 100):03d}"] = int(((live["pearson_r"].abs() >= t) & sig).sum())
            hits += [dict(model=model, sae=arch, metric=m, latent=int(h.latent), r=round(float(h.pearson_r), 3))
                     for h in live[(live["pearson_r"].abs() >= 0.40) & sig].itertuples()]
        rows.append(r)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dev_latent_counts.csv", index=False)
    pd.DataFrame(hits).to_csv(out / "dev_latents_r040.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(hits).to_string(index=False))


if __name__ == "__main__":
    main()
