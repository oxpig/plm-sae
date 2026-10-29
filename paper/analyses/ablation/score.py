"""Single-latent and co-activation group ablation, p-IgGen (Supplementary Tables 17 and 18): the target gene's share of
the ANARCI heavy chains (chain_type H) at strength 0 (control) and 1 (ablated), Yates chi-squared on the 2x2 table.
Reads the libraries of paper/scripts/ablation/ablate.sh (<work_dir>/ablation/piggen/ablation_{single,group}/, or
--libraries DIR holding ablation_single/ and ablation_group/). Writes <work_dir>/ablation/results/ablation.csv
(base, abl: proportions; dpp: change in percentage points; pct: relative change in %; p).
Run: python -m paper.analyses.ablation.score [--libraries DIR]"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

from plm_sae import paths as P
from plm_sae.numbering.number import paths
from paper.analyses.ablation.make_tasks import tasks

ARCH = {"piggen_topk_r8": "TopK", "piggen_batchtopk_r8": "Batch TopK", "piggen_ordered_r8": "Ordered"}


def counts(p, gene):
    h = pd.read_csv(p)
    h = h[h.chain_type == "H"]
    return int((h["j_gene"].astype(str).str.extract(r"(IGHJ\d)")[0] == gene).sum()), len(h)


def score(d, gene, lats):
    label = "+".join(map(str, lats))
    f0, f1 = (Path(paths(d, label, gene, a, 0, features=lats)[2]) for a in ("0.00", "1.00"))
    if not (f0.exists() and f1.exists()):
        return None
    (c0, n0), (c1, n1) = counts(f0, gene), counts(f1, gene)
    b, a = c0 / n0, c1 / n1
    p = chi2_contingency([[c0, n0 - c0], [c1, n1 - c1]], correction=True)[1]
    return dict(n0=n0, n1=n1, base=b, abl=a, dpp=100 * (a - b), pct=100 * (a - b) / b if b else float("nan"), p=p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--libraries", default=str(P.work("ablation", "piggen")))
    a = ap.parse_args()
    rows = []
    for kind, phi, sae, gene, lats in tasks():
        r = dict(kind=kind, phi=phi, sae=ARCH[sae], gene=gene, top_latent=lats[0], size=len(lats))
        s = score(f"{a.libraries}/ablation_{kind}", gene, lats)
        r.update(s if s else {"status": "missing"})
        rows.append(r)
    df = pd.DataFrame(rows)
    out = P.work("ablation", "results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
