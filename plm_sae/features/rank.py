"""Latents to steer: per class, the top_n latents by validation F1 (the gate's f1_scores/f1_scores_class_<C>.csv).

Every latent the gate scored is eligible (no margin), so every SAE contributes exactly top_n latents per class.

Run: python -m plm_sae.features.rank GATE_DIR [--top_n 3] [--classes IGHJ1 ... IGHJ6]   -> CLASS:LATENT lines
     python -m plm_sae.features.rank --rank_of F1_CSV --latents L1 L2 ...           -> F1 rank of the given latents"""
import argparse
from pathlib import Path

import pandas as pd

GENES = ["IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6"]


def load_f1(gate, classes=GENES):
    frames = []
    for g in classes:
        p = Path(gate) / "f1_scores" / f"f1_scores_class_{g}.csv"
        frames.append(pd.read_csv(p).assign(gene=g))
    return pd.concat(frames, ignore_index=True).rename(columns={"feature_idx": "latent"})


def select(gate, top_n=3, classes=GENES):
    """The top_n latents by best_f1 of each class: gene, latent, best_f1, best_threshold."""
    d = load_f1(gate, classes)
    out = [sub.sort_values("best_f1", ascending=False).head(top_n) for _, sub in d.groupby("gene")]
    return (pd.concat(out, ignore_index=True).sort_values(["gene", "best_f1"], ascending=[True, False])
            [["gene", "latent", "best_f1", "best_threshold"]])


def rank_of(f1_csv, latents):
    """F1 rank (1 = highest best_f1) of the given latents in one per-class F1 table."""
    d = pd.read_csv(f1_csv)
    col = "neuron_id" if "neuron_id" in d.columns else "feature_idx"
    d = d.sort_values("best_f1", ascending=False, kind="mergesort").reset_index(drop=True)
    d["rank"] = d.index + 1
    d = d.rename(columns={col: "latent"})
    return d[d["latent"].isin([int(x) for x in latents])][["rank", "latent", "best_threshold", "best_f1"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate_dir", nargs="?")
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--classes", nargs="+", default=GENES)
    ap.add_argument("--rank_of", help="per-class F1 table; with --latents, print their F1 ranks")
    ap.add_argument("--latents", nargs="+", type=int)
    ap.add_argument("--out", help="CSV for --rank_of")
    a = ap.parse_args()
    if a.rank_of:
        r = rank_of(a.rank_of, a.latents)
        if a.out:
            r.to_csv(a.out, index=False)
        print(r.to_csv(index=False), end="")
        return
    if not a.gate_dir:
        ap.error("GATE_DIR is required")
    for _, r in select(a.gate_dir, a.top_n, a.classes).iterrows():
        print(f"{r['gene']}:{int(r['latent'])}")


if __name__ == "__main__":
    main()
