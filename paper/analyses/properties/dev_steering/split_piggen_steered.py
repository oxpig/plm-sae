"""Split steered p-IgGen sequences (VH and VL back to back) into VH|VL pairs for FlashTAP, with ANARCI (IMGT scheme;
main environment). Reads every steered_l3_f<F>_a<A>_seed<S>_seq<N>.txt of INPUT_DIR and writes OUT_CSV with columns
feature, alpha, idx, VH, VL for the sequences ANARCI numbered with a heavy and a light chain (the first chain is VH).
Usage: python -m paper.analyses.properties.dev_steering.split_piggen_steered INPUT_DIR OUT_CSV"""
import argparse
import pathlib
import re

import pandas as pd
from anarci import anarci

PATTERN = re.compile(r"^steered_l3_f(\d+)_a([\-\d\.]+)_seed(\d+)_seq(\d+)\.txt$")


def residues(chain):
    """Amino-acid sequence of one numbered ANARCI domain (gaps removed)."""
    numbered = chain[0] if chain and isinstance(chain[0], (list, tuple)) else chain
    return "".join(aa for *_, aa in numbered if aa not in (None, "", "-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir")
    ap.add_argument("out_csv")
    a = ap.parse_args()
    rows = []
    for p in sorted(pathlib.Path(a.input_dir).glob("steered_*.txt")):
        m = PATTERN.match(p.name)
        if not m:
            continue
        seqs = [l.strip() for l in open(p) if l.strip()]
        if not seqs:
            continue
        for i, chains in enumerate(anarci(list(enumerate(seqs)), scheme="imgt", output=False)[0]):
            if chains is None or len(chains) < 2:
                continue
            vh, vl = residues(chains[0]), residues(chains[1])
            if vh and vl:
                rows.append({"feature": int(m.group(1)), "alpha": m.group(2), "idx": i, "VH": vh, "VL": vl})
    if not rows:
        raise SystemExit(f"no VH|VL pairs in {a.input_dir}")
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print(f"wrote {a.out_csv}: {len(rows)} pairs")


if __name__ == "__main__":
    main()
