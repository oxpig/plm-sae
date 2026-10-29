"""Collect <work_dir>/quality/out/*/metrics.csv into <work_dir>/quality/per_latent.csv (one row per library, metric and
reference). Usage: python -m paper.analyses.quality.summarise"""
import glob

import pandas as pd

from paper.analyses.quality.common import B


def main():
    D = pd.concat([pd.read_csv(f) for f in glob.glob(str(B / "out" / "*" / "metrics.csv"))], ignore_index=True)
    D.to_csv(B / "per_latent.csv", index=False)
    print(D[D.set != "unsteered"].groupby(["model", "set"]).tag.nunique().to_string())


if __name__ == "__main__":
    main()
