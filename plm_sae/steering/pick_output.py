#!/usr/bin/env python3
"""Output-selected latents from the output scores (plm_sae.steering.output_scores), without a GPU.

Rule: among latents that fire in at least 1% of the selection pool (seq_freq >= 0.01; plm_sae.steering.selection_pool),
the K with the highest logit-lens score ll_pred per SAE and J gene (K = 3 in the paper). Ties keep the score file's
order. paper.benchmark.check_picks reproduces the paper's picks from the deposited scores.

Usage: python -m plm_sae.steering.pick_output SCORES_CSV [--k 3] [--min_freq 0.01] [--out PICKS_CSV]
Output: one row per pick (model, arch, ckpt, gene, rank, latent, ll_pred, seq_freq), on stdout or in --out; each
latent is then steered with scripts/steering/steer.sh."""
import argparse

import pandas as pd

MIN_FREQ = 0.01


def top3(d, col, k=3):
    """The k latents of `d` with the highest `col` (stable sort, so ties keep the file order)."""
    return d.sort_values(col, ascending=False, kind="stable").latent.astype(int).head(k).tolist()


def picks(scores, k=3, min_freq=MIN_FREQ, col="ll_pred"):
    rows = []
    for (a, g), x in scores.groupby(["arch", "gene"], sort=False):
        x = x[x.seq_freq >= min_freq].sort_values(col, ascending=False, kind="stable").head(k)
        for r, row in enumerate(x.itertuples(), 1):
            rows.append(dict(model=row.model, arch=a, ckpt=row.ckpt, gene=g, rank=r, latent=int(row.latent),
                             ll_pred=getattr(row, col), seq_freq=row.seq_freq))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scores"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min_freq", type=float, default=MIN_FREQ); ap.add_argument("--out")
    a = ap.parse_args()
    D = picks(pd.read_csv(a.scores), a.k, a.min_freq)
    if a.out:
        D.to_csv(a.out, index=False); print(f"wrote {len(D)} picks to {a.out}")
    else:
        print(D.to_string(index=False))


if __name__ == "__main__":
    main()
