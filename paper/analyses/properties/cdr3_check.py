"""Check of the CDRH3 parser (paper.analyses.properties.props) against the IMGT CDR3 (positions 105-117 with
insertions), Supplementary Section A.12, over all 30,754 validation heavy chains: the IMGT CDR3 comes from the aligned
heavy chains of the IMGT table (paper.analyses.identification.imgt.build_alignment_map), or from a fresh ANARCI run
(IMGT scheme) where the aligned residues differ from the sequence.
Compared: parsed CDRH3 of 3-30 residues against IMGT CDR3 of 3-30 residues flanked by C and W in the sequence, over the
chains for which both exist; for the mismatches, the share whose IMGT CDR3 contains a cysteine. Also prints the share of
the 3,000 validation sequences of ident_simple (random_state 0) with each property.
Usage: python -m paper.analyses.properties.cdr3_check   (NPROC, default 8)"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.analyses.identification.imgt import CANONICAL_NUMBERING, build_alignment_map
from paper.analyses.properties.props import cdr3, props

NUM = np.array([int("".join(c for c in l if c.isdigit())) for l in CANONICAL_NUMBERING])


def anarci_cdr3(seq):
    from anarci import anarci
    r = anarci([("q", seq)], scheme="imgt", output=False)
    for (n, _, _), d in zip(r[0][0] or [], r[1][0] or []):
        if d["chain_type"] == "H":
            return "".join(a for (pos, _), a in n if 105 <= pos <= 117 and a != "-")
    return None


def aligned_cdr3(al):
    a = np.array(list(al))
    return "".join(a[(NUM >= 105) & (NUM <= 117) & (a != "-")])


def main():
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["orig_row", "VHorVHH"])
    seqs = val.VHorVHH.astype(str).tolist()
    amap = build_alignment_map()[0]
    al = [amap.get(int(r)) for r in val.orig_row]
    ok = [isinstance(a, str) and len(a) == 200 and a.replace("-", "") == s for a, s in zip(al, seqs)]
    redo = [i for i, o in enumerate(ok) if not o]
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        fresh = dict(zip(redo, p.map(anarci_cdr3, [seqs[i][:160] for i in redo], chunksize=8)))
    im = pd.Series([aligned_cdr3(al[i]) or None if ok[i] else fresh[i] for i in range(len(seqs))])
    rx = pd.Series([cdr3(s) for s in seqs])
    rx3 = rx.where(rx.str.len().between(3, 30))
    flank = np.array([isinstance(a, str) and ("C" + a + "W") in s[:160] for a, s in zip(im, seqs)])
    im3 = im.where(flank & im.str.len().between(3, 30))
    both = rx3.notna() & im3.notna()
    same = (rx3 == im3) & both
    mm = both & ~same
    print(f"{len(seqs)} validation heavy chains ({len(redo)} renumbered with ANARCI); both give a CDR3: {both.sum()}; "
          f"identical: {same.sum()} ({100 * same.sum() / both.sum():.1f}%); of the {mm.sum()} others, IMGT CDR3 contains C: "
          f"{im3[mm].str.contains('C').sum()} ({100 * im3[mm].str.contains('C').mean():.1f}%)")
    pr = pd.DataFrame([p for p in (props(s) for s in val.VHorVHH.astype(str).sample(3000, random_state=0)) if p])
    print(f"3,000 validation sequences, {len(pr)} with a parsed CDRH3: Y-rich {100 * (pr.Y >= 0.2).mean():.1f}%, "
          f"positive {100 * (pr.charge >= 1).mean():.1f}%, long {100 * (pr.len >= 18).mean():.1f}%")


if __name__ == "__main__":
    main()
