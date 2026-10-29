"""ANARCI numbering of steered libraries (IMGT scheme, germline assignment; ANARCI 1.3 with HMMER 3.3.2, README).

For a library steered_l3_f<LAT>_a<ALPHA>_seed<S>_seq1000.txt it writes, in the same directory,
  VH_alignment_layer_3_gene_<GENE>_features_[<LAT>]_alpha_<ALPHA>_seed_<S>_sequences_1000.csv            ANARCI hits
  VH_alignment_layer_3_gene_<GENE>_features_[<LAT>]_alpha_<ALPHA>_seed_<S>_sequences_1000_formatted.csv  one row per
      numbered chain with its germline V and J calls (read by every scorer)
GENE is a label used only in the file name.

Many libraries share sequences, so the command line numbers each unique sequence once (an ANARCI result depends only
on the sequence; query_name, the row in the library, is restored per library). Units files have lines
"dir latent gene alpha seed".
  collect  UNITS WORKDIR                 unique sequences of the unfinished libraries -> WORKDIR/uniq.txt
  number   WORKDIR TASK NTASKS NPROC     ANARCI on the sequences with id % NTASKS == TASK -> WORKDIR/res_TASK.pkl
  assemble UNITS WORKDIR NPROC           write each library's two CSV files
Run: python -m plm_sae.numbering.number collect|number|assemble ...  (scripts/numbering/number_units.sh runs all three)
hmmscan runs single-threaded (HMMER_NCPU=1); anarci.get_identity is replaced by an equivalent numpy version."""
import ast
import contextlib
import csv
import io
import os
import pickle
import re
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd

os.environ.setdefault("HMMER_NCPU", "1")


@lru_cache(maxsize=None)
def _germ(g):
    b = np.frombuffer(g.encode(), dtype=np.uint8)
    mask = b != ord("-")
    return b[mask], mask, int(mask.sum())


@lru_cache(maxsize=4096)
def _state(s):
    return np.frombuffer(s.upper().encode(), dtype=np.uint8)


def get_identity(state_sequence, germline_sequence):
    """anarci.anarci.get_identity (identity over the germline's non-gap positions), vectorised."""
    assert len(state_sequence) == len(germline_sequence) == 128
    g, mask, n = _germ(germline_sequence)
    if not n:
        return 0
    return float(int((_state(state_sequence)[mask] == g).sum())) / n


def _anarci():
    import anarci.anarci as A
    A.get_identity = get_identity
    return A.anarci


def paths(d, lat, gene, alpha, seed, features=None, total=1000):
    """(library .txt, ANARCI .csv, formatted .csv); features defaults to [lat]."""
    stem = (f"{d}/VH_alignment_layer_3_gene_{gene}_features_{features or [int(lat)]}_alpha_{alpha}_seed_{seed}"
            f"_sequences_{total}")
    return f"{d}/steered_l3_f{lat}_a{alpha}_seed{seed}_seq{total}.txt", stem + ".csv", stem + "_formatted.csv"


def write_alignment(results, savepath):
    """ANARCI alignment details -> CSV, one row per hit, 'key: value' cells."""
    valid_results = []
    for item in results:
        if item is not None:
            if isinstance(item, list) and all(isinstance(x, dict) for x in item):
                for entry in item:
                    valid_results.append([f"{k}: {v}" for k, v in entry.items()])
            else:
                valid_results.append([str(item)])
    if valid_results:
        with open(savepath, "w", newline="") as file:
            csv.writer(file).writerows(valid_results)
    return valid_results


def format_germlines(input_file, output_file):
    """ANARCI CSV -> one row per hit with v_gene_species, v_gene, v_gene_score, j_gene_species, j_gene, j_gene_score."""
    records = []
    with open(input_file, "r") as file:
        for row in csv.reader(file):
            if not row:
                continue
            try:
                record = {}
                for item in row:
                    if ": " in item and not item.startswith("germlines"):
                        key, value = item.split(": ", 1)
                        record[key] = value.strip('"')
                germline_match = re.search(r"germlines: (\{.*\})$", ",".join(row))
                if germline_match:
                    germlines = ast.literal_eval(germline_match.group(1))
                    v_gene_info = germlines.get("v_gene", [("unknown", "unknown"), 0])
                    j_gene_info = germlines.get("j_gene", [("unknown", "unknown"), 0])
                    record.update({"v_gene_species": v_gene_info[0][0], "v_gene": v_gene_info[0][1],
                                   "v_gene_score": v_gene_info[1], "j_gene_species": j_gene_info[0][0],
                                   "j_gene": j_gene_info[0][1], "j_gene_score": j_gene_info[1]})
                records.append(record)
            except Exception as e:
                print(f"Error processing row: {row}\nError details: {e}")
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    return df


def number_library(txt, csv_path, formatted_path):
    """Number one library without deduplication (the ablation libraries)."""
    seqs = read_seqs(txt)
    _, ali, _ = _anarci()(list(enumerate(seqs)), scheme="imgt", assign_germline=True, output=False)
    write_alignment(ali, csv_path)
    format_germlines(csv_path, formatted_path)


def units(path):
    for line in open(path):
        if line.strip():
            yield tuple(line.split())


def done(f):
    return os.path.exists(f) and os.path.getsize(f) > 0


def read_seqs(txt):
    with open(txt) as f:
        return [l.strip() for l in f if l.strip()]


def _number(chunk):
    anarci = _anarci()
    out = {}
    try:
        _, ali, _ = anarci(chunk, scheme="imgt", assign_germline=True, output=False)
        out.update({i: a for (i, _), a in zip(chunk, ali)})
    except Exception:   # per sequence, so one bad sequence fails only the libraries that contain it
        for i, sq in chunk:
            try:
                out[i] = anarci([(i, sq)], scheme="imgt", assign_germline=True, output=False)[1][0]
            except Exception as e:
                out[i] = ("ERROR", repr(e))
    return out


RES, IDX = {}, {}


def _assemble(u):
    txt, csv_path, fmt = paths(*u)
    if done(fmt):
        return None
    if not os.path.exists(txt):
        return f"missing input {u}"
    ali = []
    for row, sq in enumerate(read_seqs(txt)):
        r = RES[IDX[sq]]
        if isinstance(r, tuple):
            return f"FAILED {u} {r[1]}"
        ali.append(None if r is None else [{**e, "query_name": row} if "query_name" in e else e for e in r])
    with contextlib.redirect_stdout(io.StringIO()):
        write_alignment(ali, csv_path)
        format_germlines(csv_path, fmt)
    return None if done(fmt) else f"FAILED {u}"


def main():
    mode = sys.argv[1]
    if mode == "collect":
        U, O = sys.argv[2], sys.argv[3]
        os.makedirs(O, exist_ok=True)
        uniq, n_units, n_seq = {}, 0, 0
        for u in units(U):
            txt, _, fmt = paths(*u)
            if done(fmt) or not os.path.exists(txt):
                continue
            n_units += 1
            for sq in read_seqs(txt):
                n_seq += 1
                uniq.setdefault(sq, len(uniq))
        with open(f"{O}/uniq.txt", "w") as f:
            f.write("\n".join(uniq) + ("\n" if uniq else ""))
        print(f"collect: {n_units} units, {n_seq} sequences, {len(uniq)} unique", flush=True)
    elif mode == "number":
        O, T, NT, NP = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        mine = [(i, sq) for i, sq in enumerate(read_seqs(f"{O}/uniq.txt")) if i % NT == T]
        res = {}
        with Pool(NP) as p:
            for r in p.imap_unordered(_number, [mine[k:k + 200] for k in range(0, len(mine), 200)]):
                res.update(r)
        pickle.dump(res, open(f"{O}/res_{T}.pkl.tmp", "wb"))
        os.replace(f"{O}/res_{T}.pkl.tmp", f"{O}/res_{T}.pkl")
        print(f"number: task {T} numbered {len(res)} sequences", flush=True)
    elif mode == "assemble":
        U, O, NP = sys.argv[2], sys.argv[3], int(sys.argv[4])
        IDX.update({sq: i for i, sq in enumerate(read_seqs(f"{O}/uniq.txt"))})
        for fn in sorted(os.listdir(O)):
            if fn.startswith("res_") and fn.endswith(".pkl"):
                RES.update(pickle.load(open(f"{O}/{fn}", "rb")))
        assert len(RES) == len(IDX), (len(RES), len(IDX))
        todo = list(units(U))
        with Pool(NP) as p:   # fork: the workers share RES and IDX without pickling them
            msgs = [m for m in p.imap_unordered(_assemble, todo, chunksize=4) if m]
        for m in msgs:
            print(m, flush=True)
        print(f"assemble: {len(todo)} units checked, {len(msgs)} problems", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
