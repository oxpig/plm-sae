"""Margin sweep over the per-latent F1 tables of feature identification (Methods, Eqs 6-7).

  Eq 6  F1_all+ = 2 pi / (1 + pi), pi = validation prevalence of the class
  Eq 7  a latent is a feature of the class if best_f1 > F1_all+ + margin

Reads the per-class F1 tables of a gate: f1_scores_class_<C>.csv (plm_sae.features.gate) or
feature_f1_scores_class_<c>.csv (the Section 4.1 gate, paper.analyses.identification.section41_gate).
Writes to OUT_DIR: prevalence.csv (n, pi and F1_all+ per class), counts_by_margin.csv and summary.csv (features per
class and margin), latents_all_margins.csv and latents_margin_<m>.csv (the features).

Run on existing F1 tables (the gates also call run_margin_sweep themselves):
  python -m plm_sae.features.margin_sweep --f1_dir GATE/f1_scores --label_csv VAL_CSV --label_col heavy_j_gene \
      [--out_dir DIR] [--margins 0.10 0.15 0.20 0.25]"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

MARGINS = [0.10, 0.15, 0.20, 0.25]
PREFIXES = ("f1_scores_class_", "feature_f1_scores_class_")


def f1_all_positive(prevalence):
    """Eq 6: F1 of predicting every sequence as the class."""
    return 2.0 * prevalence / (1.0 + prevalence)


def normalise_label(x):
    """Gene label without allele or suffix: IGHJ4*02 -> IGHJ4 (also cuts at '-', so not for V genes)."""
    return str(x).split("*", 1)[0].split("-", 1)[0]


def _f1_files(f1_dir):
    """(path, class label) of every per-class F1 table in f1_dir."""
    out = []
    for prefix in PREFIXES:
        for path in sorted(glob.glob(os.path.join(f1_dir, prefix + "*.csv"))):
            out.append((path, os.path.basename(path)[len(prefix):-len(".csv")]))
    if not out:
        raise FileNotFoundError(f"no per-class F1 tables in {f1_dir}")
    return out


def _latent_col(df):
    for c in ("feature_idx", "neuron_id", "Index", "feature"):
        if c in df.columns:
            return c
    raise ValueError(f"no latent column in F1 table: {list(df.columns)}")


def run_margin_sweep(f1_dir, y=None, margins=MARGINS, out_dir=None, label_csv=None, label_col=None, verbose=True):
    """Eq 6-7 over the F1 tables of f1_dir; y = validation labels (or read from label_csv / label_col)."""
    if label_csv is not None:
        y = pd.read_csv(label_csv)[label_col].tolist()
    counts = Counter(normalise_label(v) for v in y)
    n = float(sum(counts.values()))
    prevalence = {c: k / n for c, k in counts.items()}
    out_dir = out_dir or os.path.join(f1_dir, "margin_sweep")
    os.makedirs(out_dir, exist_ok=True)
    prev = pd.DataFrame([{"class": c, "n": counts[c], "prevalence": p, "f1_all_plus": f1_all_positive(p)}
                         for c, p in sorted(prevalence.items())])
    prev.to_csv(os.path.join(out_dir, "prevalence.csv"), index=False)

    count_rows, latent_rows = [], []
    upper = {k.upper(): k for k in prevalence}
    for path, label in _f1_files(f1_dir):
        label = upper.get(label.upper(), label)
        if label not in prevalence:
            print(f"[margin_sweep] WARN: class {label} ({os.path.basename(path)}) not in the labels; skipped")
            continue
        df = pd.read_csv(path)
        lat, pi = _latent_col(df), prevalence[label]
        for m in margins:
            sel = df[df["best_f1"] > f1_all_positive(pi) + m].sort_values("best_f1", ascending=False)
            count_rows.append({"class": label, "prevalence": pi, "f1_all_plus": f1_all_positive(pi), "margin": m,
                               "n_features": int(len(sel))})
            for _, r in sel.iterrows():
                latent_rows.append({"class": label, "margin": m, "latent": int(r[lat]), "best_f1": float(r["best_f1"]),
                                    "best_threshold": r["best_threshold"] if "best_threshold" in df.columns else np.nan,
                                    "weight": r["weight"] if "weight" in df.columns else np.nan})

    cnt = pd.DataFrame(count_rows)
    cnt.to_csv(os.path.join(out_dir, "counts_by_margin.csv"), index=False)
    table = cnt.pivot_table(index="class", columns="margin", values="n_features", aggfunc="sum")
    table.reindex(columns=sorted(margins)).reset_index().to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    if latent_rows:
        lat = pd.DataFrame(latent_rows)
        lat.to_csv(os.path.join(out_dir, "latents_all_margins.csv"), index=False)
        for m in sorted(margins):
            lat[lat.margin == m].sort_values(["class", "best_f1"], ascending=[True, False]).to_csv(
                os.path.join(out_dir, f"latents_margin_{m:.2f}.csv"), index=False)
    if verbose:
        print(prev.to_string(index=False))
        print(table.to_string())
    return {"prevalence": prevalence, "counts_by_margin": cnt, "out_dir": out_dir}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--f1_dir", required=True)
    p.add_argument("--label_csv", required=True)
    p.add_argument("--label_col", default="heavy_j_gene")
    p.add_argument("--out_dir")
    p.add_argument("--margins", nargs="+", type=float, default=MARGINS)
    a = p.parse_args()
    run_margin_sweep(a.f1_dir, margins=a.margins, out_dir=a.out_dir, label_csv=a.label_csv, label_col=a.label_col)


if __name__ == "__main__":
    main()
