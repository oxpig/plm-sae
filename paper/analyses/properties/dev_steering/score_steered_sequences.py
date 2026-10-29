"""FlashTAP scores (PSH, PPC, PNC, SFvCSP) of steered p-IgGen sequences, split into VH|VL by split_piggen_steered.py.
Writes OUT_CSV with columns feature, alpha, idx, PSH, PPC, PNC, SFvCSP (NaN for a chunk of 1,000 that FlashTAP cannot
score). FlashTAP environment, GPU:
  "$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.dev_steering.score_steered_sequences PAIRS_CSV FEATURE OUT_CSV"""
import argparse

import pandas as pd
import torch
from flash_abb import pretrained_tap

METRICS = ["PSH", "PPC", "PNC", "SFvCSP"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs_csv")
    ap.add_argument("feature", type=int)
    ap.add_argument("out_csv")
    ap.add_argument("--batch_size", type=int, default=50)
    a = ap.parse_args()
    tap = pretrained_tap(device="cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.pairs_csv)
    df = df[df["feature"] == a.feature]
    seqs = (df["VH"].astype(str) + "|" + df["VL"].astype(str)).tolist()
    scores = []
    for i in range(0, len(seqs), a.batch_size * 20):
        c = seqs[i:i + a.batch_size * 20]
        try:
            scores += tap(c, batch_size=a.batch_size).scores
        except Exception as e:
            print(f"WARNING: chunk failed ({e!r}); NaN for {len(c)} sequences")
            scores += [{m: float("nan") for m in METRICS} for _ in c]
    out = pd.DataFrame(scores)[METRICS]
    out.insert(0, "feature", a.feature)
    out.insert(1, "alpha", df["alpha"].astype(str).values)
    out.insert(2, "idx", df["idx"].values)
    out.to_csv(a.out_csv, index=False)
    print(f"wrote {a.out_csv}: {len(out)} rows")


if __name__ == "__main__":
    main()
