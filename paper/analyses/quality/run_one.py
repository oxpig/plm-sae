"""One library of the steered generation analysis: ANARCI numbering (IMGT; the first heavy and the first light domain of
each sequence; p-IgGen sequences need both chains), the metrics of paper.analyses.quality.common, and the tests against
the natural sequences (ref natural; <work_dir>/quality/natural_{paired,heavy}.npz) and, for steered libraries, against
the model's unsteered library (ref unsteered). Writes <work_dir>/quality/out/<tag>/{steered_aligned.csv.gz, per_seq.npz,
metrics.csv}; skips a row whose metrics.csv exists.
Usage: python -m paper.analyses.quality.run_one ROW   (ROW of <work_dir>/quality/list.csv; GPU for ESM-2 and IgLM)"""
import argparse
import os

import numpy as np
import pandas as pd
from anarci import anarci

from paper.analyses.quality.common import B, chain_to_aligned, mwu, per_sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("row", type=int)
    a = ap.parse_args()
    r = pd.read_csv(B / "list.csv").iloc[a.row]
    od = B / "out" / r.tag
    if (od / "metrics.csv").exists():
        print("exists", od)
        return
    os.makedirs(od, exist_ok=True)
    paired = r.model == "piggen"
    seqs = [l.strip() for l in open(r.lib) if l.strip()]
    num, det, _ = anarci([(f"q{i}", s) for i, s in enumerate(seqs)], scheme="imgt", output=False)
    rows = []
    for ch, dt, s in zip(num, det, seqs):
        if ch is None:
            continue
        c = {}
        for (dn, st, en), d in zip(ch, dt):
            t = "H" if d["chain_type"] == "H" else ("L" if d["chain_type"] in ("K", "L") else None)
            if t and t not in c:
                c[t] = (s[st:en + 1], chain_to_aligned(dn))
        if "H" in c and (not paired or "L" in c):
            rows.append(dict(VH=c["H"][0], VHal=c["H"][1], VL=c["L"][0] if paired else None, VLal=c["L"][1] if paired else None))
    S = pd.DataFrame(rows)
    S.to_csv(od / "steered_aligned.csv.gz", index=False)
    meta = {c: r[c] for c in ("tag", "model", "arch", "set", "rank", "target", "latent", "alpha", "both", "lib")}
    if not len(S):   # no sequence numbered with both chains: recorded, not compared
        pd.DataFrame([dict(metric="none", ref="none", n_steered=0, **meta)]).to_csv(od / "metrics.csv", index=False)
        return
    d = per_sequence(S.VHal.tolist(), S.VLal.tolist() if paired else None, S.VH.tolist(), S.VL.tolist() if paired else None)
    np.savez(od / "per_seq.npz", **d)
    refs = {"natural": np.load(B / f"natural_{'paired' if paired else 'heavy'}.npz")}
    if r.set != "unsteered":
        refs["unsteered"] = np.load(B / "out" / f"{r.model}_unsteered_a0" / "per_seq.npz")
    out = [dict(metric=k, ref=ref, **mwu(d[k], R[k])) for ref, R in refs.items() for k in d]
    out.append(dict(metric="ANARCI_numbered_count", ref="", n_steered=len(S), n_ref=len(seqs), mean_steered=len(S),
                    mean_ref=len(seqs)))
    O = pd.DataFrame(out).assign(**meta)
    O.to_csv(od / "metrics.csv.tmp", index=False)
    os.replace(od / "metrics.csv.tmp", od / "metrics.csv")


if __name__ == "__main__":
    main()
