"""Light-chain type labels (Supplementary Section A.7): vj_{train,val}_anarci_lct.csv = vj_{train,val}_anarci.csv with
a column light_chain_type, kappa for light_v_gene IGKV* and lambda for IGLV*; stops if any light_v_gene has neither
prefix. The data record holds these two tables (datasets/); this script derives them.
Usage: python -m paper.analyses.identification.add_light_chain_type [--out_dir DIR]   (default <data_dir>/datasets)"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from plm_sae import paths as P


def add_column(in_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)
    if "light_v_gene" not in df.columns:
        sys.exit(f"ERROR: {in_csv} has no light_v_gene column")

    prefix = df["light_v_gene"].str[:4]
    is_kappa = prefix == "IGKV"
    is_lambda = prefix == "IGLV"
    unmatched = ~(is_kappa | is_lambda)
    if unmatched.any():
        bad = df.loc[unmatched, "light_v_gene"].head(10).tolist()
        sys.exit(
            f"ERROR: {unmatched.sum()} rows in {in_csv} have light_v_gene "
            f"values that are neither IGKV* nor IGLV* -- refusing to guess. "
            f"Examples: {bad}"
        )

    df["light_chain_type"] = is_kappa.map({True: "kappa", False: "lambda"})
    df.to_csv(out_csv, index=False)
    print(
        f"{in_csv.name} -> {out_csv.name}: "
        f"{int(is_kappa.sum())} kappa, {int(is_lambda.sum())} lambda, "
        f"{len(df)} total"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=P.dataset(),
                    help="directory holding vj_{train,val}_anarci.csv (default <data_dir>/datasets)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="where to write vj_{train,val}_anarci_lct.csv (default: --datasets)")
    args = ap.parse_args()

    datasets = args.datasets
    out_dir = args.out_dir or datasets
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        in_csv = datasets / f"vj_{split}_anarci.csv"
        out_csv = out_dir / f"vj_{split}_anarci_lct.csv"
        add_column(in_csv, out_csv)


if __name__ == "__main__":
    main()
