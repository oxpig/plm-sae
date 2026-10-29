"""The selection pool: 300 validation heavy chains per heavy J gene (IGHJ1-6), 1,800 in all.

The same draw gives the firing statistics of both latent selections:
  * baseline latents (paper.benchmark.baselines): firing rate 5 to 50%, within 1.25-fold of the
    overall rate for every J gene;
  * output-selected latents (plm_sae.steering.output_scores, seq_freq): fire in at least 1% of the pool.
Each gene's chains are drawn from vj_val_anarci.csv (heavy chains with an IGHJ1-6 ANARCI call, allele removed) with
pandas `sample(n=300, random_state=1)`."""
import pandas as pd

from plm_sae import paths as P

SEED = 1
PER_GENE = 300
GENES = [f"IGHJ{i}" for i in range(1, 7)]


def load_validation(usecols=("VHorVHH", "j_gene_anarci_heavy")):
    """vj_val_anarci.csv with a `gene` column (ANARCI heavy J call without allele)."""
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=list(usecols))
    val["gene"] = val.j_gene_anarci_heavy.astype(str).str.split("*").str[0]
    return val


def validation_pool(val=None, per_gene=PER_GENE):
    """per_gene chains per IGHJ1-6 gene of `val` (default load_validation()), each gene drawn with random_state SEED."""
    val = load_validation() if val is None else val
    val = val[val.gene.str.match(r"IGHJ[1-6]$")]
    return val.groupby("gene", group_keys=False).apply(lambda d: d.sample(n=min(per_gene, len(d)), random_state=SEED))
