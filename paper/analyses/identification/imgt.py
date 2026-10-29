"""IMGT numbering of aligned heavy chains, shared by the localisation analyses and figures.

CANONICAL_NUMBERING   the 200 IMGT positions (with insertions) of an aligned heavy chain (aligned_VHorVHH of the IMGT
                      table, datasets/master_Dec13_10.31.csv)
in_jregion(label)     IMGT 115-128 and 119A, the J region
build_alignment_map   orig_row of a validation table -> its aligned heavy chain, joined on VH+VL
write_subset          the validation rows of one J gene (ANARCI call), the input of the IgLM and ProGen2-OAS IGHJ4
                      token encodes (Supplementary Figure 8)"""
import os

import pandas as pd

from plm_sae import paths as P

CANONICAL_NUMBERING = ["1 ", "2 ",
                       "3 ", "3A",
                       "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10 ", "11 ", "12 ", "13 ", "14 ", "15 ", "16 ", "17 ", "18 ",
                       "19 ", "20 ", "21 ", "22 ", "23 ", "24 ", "25 ", "26 ", "27 ", "28 ", "29 ", "30 ", "31 ",
                       "32 ", "32A", "32B", '33C', "33B", "33A", "33 ",
                       "34 ", "35 ", "36 ", "37 ", "38 ", "39 ",
                       "40 ", '40A',
                       "41 ", "42 ", "43 ",
                       "44 ", "44A",
                       "45 ", "45A",
                       "46 ", "46A",
                       "47 ", "47A",
                       "48 ", "48A", "48B",
                       "49 ", "49A",
                       "50 ",
                       "51 ", "51A",
                       "52 ", "53 ", "54 ", "55 ", "56 ", "57 ", "58 ", "59 ",
                       "60 ", "60A", "60B", "60C", '60D', "61E", '61D', "61C", "61B", "61A", "61 ",
                       "62 ", "63 ", "64 ", "65 ", "66 ",
                       "67 ", '67A', "67B",
                       "68 ", "68A", "68B",
                       "69 ", "69A", "69B",
                       "70 ",
                       "71 ", "71A", "71B",
                       "72 ",
                       "73 ", '73A', "73B",
                       "74 ", "75 ", "76 ", "77 ", "78 ", "79 ",
                       "80 ", "80A",
                       "81 ", "81A", "81B", "81C",
                       "82 ", "82A",
                       "83 ", "83A", "83B",
                       "84 ",
                       "85 ", "85A", "85B", "85C", "85D",
                       "86 ", "86A",
                       "87 ", "88 ", "89 ", "90 ", "91 ", "92 ", "93 ", "94 ", "95 ",
                       "96 ", "96A",
                       "97 ", "98 ", "99 ", "100 ", "101 ", "102 ", "103 ", "104 ", "105 ", "106 ", "107 ", "108 ",
                       "109 ", "110 ",
                       "111 ", "111A", "111B", "111C", "111D", "111E", "111F", "111G", '111H', '111I', '111J',
                       "111K", "111L", "112L",
                       '112K', '112J', '112I', '112H', "112G", "112F", "112E", "112D", "112C", "112B", "112A", "112 ",
                       "113 ","114 ","115 ","116 ","117 ","118 ",
                       "119 ", "119A",
                       "120 ","121 ","122 ","123 ","124 ","125 ", "126 ","127 ","128 "
                        ]
assert len(CANONICAL_NUMBERING) == 200


def in_jregion(label):
    s = label.strip()
    if s == "119A":
        return True
    num = "".join(ch for ch in s if ch.isdigit())
    return num != "" and 115 <= int(num) <= 128


def build_alignment_map(val=None):
    """({orig_row: aligned heavy chain (200 characters)}, rows, rows with an alignment) for a validation table
    (default datasets/vj_val_anarci.csv), joined to the IMGT table on VH+VL (first row per VH+VL)."""
    val = pd.read_csv(P.dataset("vj_val_anarci.csv")) if val is None else val
    master = pd.read_csv(P.dataset("master_Dec13_10.31.csv"), usecols=["VH+VL", "aligned_VHorVHH"])
    master = master.dropna(subset=["VH+VL", "aligned_VHorVHH"]).drop_duplicates(subset="VH+VL", keep="first")
    m = val[["orig_row", "VH+VL"]].merge(master, on="VH+VL", how="left")
    ok = m.aligned_VHorVHH.notna()
    return dict(zip(m.orig_row[ok], m.aligned_VHorVHH[ok])), len(m), int(ok.sum())


def subset_csv(gene="IGHJ4"):
    return P.work("identification", "imgt", f"vj_val_anarci_{gene.lower()}.csv")


def write_subset(gene="IGHJ4"):
    """The validation rows of one gene (ANARCI J call), index reset, orig_row kept."""
    v = pd.read_csv(P.dataset("vj_val_anarci.csv"))
    sub = v[v["j_gene_anarci_heavy"].astype(str).str.split("*").str[0] == gene].reset_index(drop=True)
    os.makedirs(subset_csv(gene).parent, exist_ok=True)
    sub.to_csv(subset_csv(gene), index=False)
    print(f"wrote {subset_csv(gene)}: {len(sub)} rows")


if __name__ == "__main__":
    write_subset()
