"""FlashTAP scores (PSH, PPC, PNC, SFvCSP; flash_abb 0.0.4) of every VH|VL pair (columns VHorVHH, VL) of a validation
table whose orig_row runs 0 .. n-1, in chunks of 1,000 (NaN for a chunk FlashTAP cannot score). FlashTAP environment, GPU.
Usage: "$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.developability.flashtap_val VAL_CSV OUT_CSV"""
import argparse

import numpy as np
import pandas as pd
import torch
from flash_abb import pretrained_tap

METRICS = ["PSH", "PPC", "PNC", "SFvCSP"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("val_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--batch_size", type=int, default=50)
    a = ap.parse_args()
    df = pd.read_csv(a.val_csv)
    rows = df["orig_row"].to_numpy()
    assert np.array_equal(rows, np.arange(len(df))), "orig_row must run 0 .. n-1"
    seqs = (df["VHorVHH"].astype(str) + "|" + df["VL"].astype(str)).tolist()
    tap = pretrained_tap(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scores, chunk = [], a.batch_size * 20
    for i in range(0, len(seqs), chunk):
        try:
            scores += tap(seqs[i:i + chunk], batch_size=a.batch_size).scores
        except Exception as e:
            print(f"WARNING: rows {i}-{i + chunk} failed ({e!r}); NaN")
            scores += [{m: float("nan") for m in METRICS}] * len(seqs[i:i + chunk])
    out = pd.DataFrame(scores)[METRICS]
    out.insert(0, "orig_row", rows)
    out.to_csv(a.out_csv, index=False)
    print(out[METRICS].describe().to_string())


if __name__ == "__main__":
    main()
