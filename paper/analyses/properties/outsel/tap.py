"""FlashTAP scores (PSH, PPC, PNC, SFvCSP) of the steered sequences of the developability runs (GPU, FlashTAP
environment). A sequence is scored if ANARCI found a valid heavy chain (chain H, species and J species human, J identity
>= 0.5) and a numbered light chain after it and both contain only the 20 standard amino acids; it is scored as 'VH|VL'
with VH = seq[query_start:query_end] of the H domain and VL likewise. Runs whose tasks are not all done and numbered
are skipped (run again later).
Usage: "$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.outsel.tap SHARD NSHARDS
       runs of tap/runs.lst (one per line), shard k takes runs k, k+n, ...
Output: <work_dir>/outsel_properties/tap/<run>/a<alpha>_s<seed>.csv (idx, PSH, PPC, PNC, SFvCSP), skipped if present."""
import os
import re
import sys
from pathlib import Path

import pandas as pd
from flash_abb import pretrained_tap

from plm_sae import paths as P

METRICS = ["PSH", "PPC", "PNC", "SFvCSP"]
AA = set("ACDEFGHIKLMNPQRSTVWY|")


def kv(p):
    return dict(re.findall(r'^(\w+)="?(.*?)"?$', Path(p).read_text(), re.M))


def pairs(d, lat, g, a, s):
    """(row in the library, 'VH|VL') of every scorable sequence of one library."""
    txt = f"{d}/steered_l3_f{lat}_a{a}_seed{s}_seq1000.txt"
    fm = f"{d}/VH_alignment_layer_3_gene_{g}_features_[{lat}]_alpha_{a}_seed_{s}_sequences_1000_formatted.csv"
    if not os.path.exists(txt) or not os.path.exists(fm) or os.path.getsize(fm) == 0:
        return []
    seqs = [l.strip() for l in open(txt) if l.strip()]
    df = pd.read_csv(fm, usecols=["query_name", "chain_type", "species", "j_gene_species", "j_gene_score",
                                  "query_start", "query_end"])
    H = df[(df.chain_type == "H") & (df.species == "human") & (df.j_gene_species == "human") & (df.j_gene_score >= 0.5)]
    L, out = df[df.chain_type.isin(["K", "L"])], []
    for h in H.drop_duplicates("query_name").itertuples():
        l = L[(L.query_name == h.query_name) & (L.query_start >= h.query_end)]
        if len(l):
            l, sq = l.iloc[0], seqs[int(h.query_name)]
            out.append((int(h.query_name), sq[h.query_start:h.query_end] + "|" + sq[int(l.query_start):int(l.query_end)]))
    return out


def main():
    S = P.work("outsel_properties")
    k, n = int(sys.argv[1]), int(sys.argv[2])
    tap = pretrained_tap(device="cuda")
    for run in (S / "tap" / "runs.lst").read_text().split()[k::n]:
        kids = [t for t in os.listdir(S / "tasks/done") if "." not in t and (t == run or t.endswith("_" + run))]
        queued = [t for t in os.listdir(S / "tasks") if "." not in t and (S / "tasks" / t).is_file()
                  and (t == run or t.endswith("_" + run))]
        if queued or not all((S / "anarci" / t / "HI_DONE").exists() for t in kids):
            print("skip (not finished or not numbered)", run, flush=True)
            continue
        od = S / "tap" / run
        od.mkdir(parents=True, exist_ok=True)
        units = sorted({tuple(l.split()) for t in kids for l in open(kv(S / "tasks/done" / t)["UNITS"]) if l.strip()})
        for d, lat, g, a, s in units:
            o = od / f"a{a}_s{s}.csv"
            if o.exists():
                continue
            ok = [(i, p) for i, p in pairs(d, lat, g, a, s) if set(p) <= AA]
            sc = []
            for j in range(0, len(ok), 1000):
                sc += tap([p for _, p in ok[j:j + 1000]], batch_size=50).scores
            D = pd.DataFrame(sc, columns=METRICS)
            D.insert(0, "idx", [i for i, _ in ok])
            D.to_csv(f"{o}.tmp", index=False)
            os.replace(f"{o}.tmp", o)
            print(run, a, s, len(D), flush=True)


if __name__ == "__main__":
    main()
