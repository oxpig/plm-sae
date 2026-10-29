"""IMGT alignment of numbered chains for the steered generation analysis: each ANARCI-numbered domain is written onto the
200 IMGT positions of paper.analyses.identification.imgt (insertions included), so that Hamming distances compare the
same positions."""
from paper.analyses.identification.imgt import CANONICAL_NUMBERING

POS_INDEX = {lab.strip(): i for i, lab in enumerate(CANONICAL_NUMBERING)}


def chain_to_aligned(domain_numbering):
    """200-character aligned chain ('-' for positions without a residue) of one ANARCI domain numbering."""
    row = ["-"] * 200
    for (pos, ins), aa in domain_numbering:
        if aa in ("-", None):
            continue
        ins = ins.strip() if isinstance(ins, str) else ins
        lab = f"{pos}{ins}" if ins else f"{pos}"
        if lab in POS_INDEX:
            row[POS_INDEX[lab]] = aa
    return "".join(row)
