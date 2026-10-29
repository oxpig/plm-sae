"""Ladder extension and refinement of the property runs, with the steering benchmark's rules (p-IgGen base maximum A =
120): extension while the pooled coverage at the last alpha (numbered human heavy chains with J identity >= 0.5, three
seeds / 3,000) is >= 0.8, 5 alphas A_k + 24 * (1..5), at most two blocks (tasks x<block>_<run>); refinement once not
extended, when fewer than five alphas are in the valid range and it ends before the last alpha (a_hi * j / 8,
j = 1..7, minus alphas already run; task r_<run>). A refined ladder is not extended again.
Writes the new tasks into tasks/ and their names into tasks/<PHASE>.lst, and ladder_<PHASE>.csv (per run: alphas,
coverage, decision; PENDING while a task of the run is unfinished or unnumbered).
Usage: python -m paper.analyses.properties.outsel.ladder PHASE   (NPROC)"""
import os, re, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from plm_sae import paths as P
from paper.benchmark import analyze_final as af
S = P.work("outsel_properties")
A, NBLOCK, MAXBLOCK = 120.0, 5, 2


def kv(p): return dict(re.findall(r'^(\w+)="?(.*?)"?$', Path(p).read_text(), re.M))


def nhj(u): return af.lib_stats(u)["n_Hj"]


def main():
    phase = sys.argv[1]; new_names = []; log = []
    parents = (S / "tasks" / "base.lst").read_text().split()
    alltasks = {t: q for q in ("", "done") for t in os.listdir(S / "tasks" / q) if "." not in t and (S / "tasks" / q / t).is_file()}
    info = {}
    for p in parents:
        kids = [t for t in alltasks if t == p or t.endswith("_" + p)]
        unfinished = [t for t in kids if alltasks[t] != "done" or not (S / "anarci" / t / "HI_DONE").exists()]
        if unfinished: info[p] = ("PENDING", unfinished); continue
        units = [line.split() for t in kids for line in open(kv(S / "tasks/done" / t)["UNITS"]) if line.strip()]
        info[p] = ("OK", kids, units)
    jobs = sorted({(d, int(lat), g, float(a), int(s)) for v in info.values() if v[0] == "OK" for d, lat, g, a, s in v[2]})
    with Pool(int(os.environ.get("NPROC", 8))) as pool: N = dict(zip(jobs, pool.map(nhj, jobs, chunksize=16)))
    for p in parents:
        if info[p][0] == "PENDING": print(p, "PENDING", info[p][1]); log.append(dict(run=p, decision="PENDING")); continue
        _, kids, units = info[p]; d, lat, g = units[0][0], int(units[0][1]), units[0][2]
        al = np.array(sorted({float(u[3]) for u in units}))
        cov = np.array([sum(N[(d, lat, g, a, s)] for s in (0, 1, 2)) / 3000 for a in al])
        block = sum(bool(re.match(r"^x\d_", t)) for t in kids); refined = any(t.startswith("r_") for t in kids)
        new, name = [], None
        if cov[-1] >= 0.8 and block < MAXBLOCK and not refined:
            new = [round(x, 2) for x in al.max() + A / NBLOCK * np.arange(1, NBLOCK + 1)]; name = f"x{block + 1}_{p}"
        elif not refined:
            bad = np.where(cov < 0.8)[0]; k = bad[0] if len(bad) else len(al)
            if 0 < k < 5 and k < len(al):
                a_hi = al[k]; new = [round(a_hi * j / 8, 2) for j in range(1, 8)]; new = [a for a in new if not np.any(np.abs(al - a) < 1e-6)]
                name = f"r_{p}"
        log.append(dict(run=p, alphas=" ".join(f"{a:g}" for a in al), coverage=" ".join(f"{c:.3f}" for c in cov), decision=name or "FINAL",
                        new_alphas=" ".join(f"{a:g}" for a in new)))
        if not new: continue
        k0 = kv(S / "tasks/done" / p); st = int(open(k0["CFG"]).readline().split()[2]); tag = "/".join(Path(d).parts[-2:])
        (S / "cfg" / f"{name}.txt").write_text("".join(f"{lat} {a:.2f} {st} {tag}\n" for a in new))
        (S / "units" / f"{name}.txt").write_text("".join(f"{d} {lat} {g} {a:.2f} {s}\n" for a in new for s in (0, 1, 2)))
        assert not (S / "tasks" / name).exists() and not (S / "tasks/done" / name).exists(), name
        (S / "tasks" / name).write_text(f'MODEL={k0["MODEL"]}\nCK={k0["CK"]}\nCFG="{S}/cfg/{name}.txt"\nOUT={k0["OUT"]}\n'
                                        f'UNITS={S}/units/{name}.txt\nNLIB={3 * len(new)}\n')
        new_names.append(name)
    (S / "tasks" / f"{phase}.lst").write_text("\n".join(new_names) + ("\n" if new_names else ""))
    L = pd.DataFrame(log); L.to_csv(S / f"ladder_{phase}.csv", index=False)
    print(L.decision.str.extract(r"^(x1|x2|r|FINAL|PENDING)")[0].value_counts().to_string())
    print("NEW", len(new_names), "tasks,", sum(3 * len(open(S / "cfg" / f"{n}.txt").readlines()) for n in new_names), "libraries")


if __name__ == "__main__":
    main()
