"""Ladder extension (Supplementary Section A.2): a latent whose pooled coverage at the last alpha of its ladder is still
>= 80% gets an extension block of 5 alphas spanning (A_k, A_k + A] in steps of A/5, where A is the base maximum of its
model (p-IgGen 120, IgLM 60, ProGen2-OAS 30; doubled at the block output) and A_k the current maximum; at most two
blocks. Coverage: numbered human heavy chains with a germline J segment (identity >= 0.5) / 3,000 (J-gene and
developability targets); numbered light chains relative to alpha 0 (kappa, lambda).
Same start, direction, SAE, seeds and generation settings as the parent task; libraries land in the parent's
directories. Tasks are named x<block>_<parent>; refinement tasks (r_*, make_refine) are not extended.

Run repeatedly by paper/scripts/benchmark/chain.sh: a finished task is evaluated once its ANARCI numbering is complete
(anarci/<task>/assemble.log has HI_DONE). State: ext/evaluated.txt, ext/decisions.csv. Prints "PENDING n", the
finished tasks still being numbered. Usage: python -m paper.benchmark.extend   (state <work_dir>/benchmark; optional
environment PLM_SAE_SLURM_CLUSTER for squeue -M)"""
import os, re, subprocess, getpass
from pathlib import Path
import numpy as np, pandas as pd
from plm_sae import paths as P
from paper.benchmark.analyze_final import lib_stats
F = P.work("benchmark"); X = F / "ext"; X.mkdir(parents=True, exist_ok=True)
BASE = {"piggen": 120.0, "IgLM": 60.0}; NBLOCK, MAXBLOCK = 5, 2
REFINED = re.compile(r"^r_")


def kv(path):
    return dict(re.findall(r'^(\w+)="?(.*?)"?$', Path(path).read_text(), re.M))


def numbered(t):
    log = F / "anarci" / t / "assemble.log"
    if log.exists() and "HI_DONE" in log.read_text(): return True
    # numbering that ended without HI_DONE (failed job): evaluate anyway once nothing of it is left in the queue
    if (F / "anarci" / f"{t}.queued").exists():
        tag = f"anarci_{t[:14]}_"   # job-name prefix of anarci_array.sh (common.sh number_task)
        cl = os.environ.get("PLM_SAE_SLURM_CLUSTER", "")
        try:
            q = subprocess.run(["squeue"] + (["-M", cl] if cl else []) + ["-u", os.environ.get("USER") or getpass.getuser(), "-h", "-o", "%j"],
                               capture_output=True, text=True).stdout
        except FileNotFoundError:   # no SLURM: numbering ran in the foreground (scripts/numbering/number_units.sh)
            q = ""
        if tag not in q:
            print(f"WARNING {t}: ANARCI finished without HI_DONE; evaluating with what exists", flush=True); return True
    return False


def coverage(d, lat, g, a):
    r = [lib_stats((d, lat, g, a, s)) for s in (0, 1, 2)]
    if g in ("kappa", "lambda"):
        r0 = [lib_stats((d, lat, g, 0.0, s)) for s in (0, 1, 2)]
        return sum(x["n_light"] for x in r) / max(sum(x["n_light"] for x in r0), 1)
    return sum(x["n_Hj"] for x in r) / 3000   # coverage as analyze_final's germline-J readout


def main():
    done_eval = set((X / "evaluated.txt").read_text().split()) if (X / "evaluated.txt").exists() else set()
    dec = pd.read_csv(X / "decisions.csv").to_dict("records") if (X / "decisions.csv").exists() else []
    pending, new_tasks = 0, []
    for t in sorted(os.listdir(F / "tasks/done")):
        if "." in t or REFINED.match(t) or t in done_eval: continue
        if not numbered(t): pending += 1; continue
        k = kv(F / "tasks/done" / t); cfg_path, *flags = k["CFG"].split(" ", 1); flags = (" " + flags[0]) if flags else ""
        starts = {}
        for line in open(cfg_path):
            if line.strip():
                lat, a, st, tag = line.split(); starts[(int(lat), tag.split("/")[1])] = (int(st), tag.split("/")[0])
        keys = {}
        for line in open(k["UNITS"]):
            if line.strip():
                d, lat, g, a, s = line.split(); keys.setdefault((d, int(lat), g), set()).add(float(a))
        block = int(t[1]) if re.match(r"^x\d_", t) else 0
        parent = t.split("_", 1)[1] if block else t
        A = BASE[k["MODEL"]] * (2 if "--site blockout" in flags else 1)
        rows = []
        for (d, lat, g), al in sorted(keys.items()):
            amax = max(al); cov = coverage(d, lat, g, amax)
            ext = cov >= 0.8 and block < MAXBLOCK
            dec.append(dict(task=t, dir=d, latent=lat, target=g, alpha_max=amax, cov_at_max=round(cov, 3), block=block, extend=ext,
                            capped=cov >= 0.8 and block >= MAXBLOCK))
            if ext:
                st, rule = starts[(lat, g)]; rows.append((d, lat, g, st, rule, amax + A / NBLOCK * np.arange(1, NBLOCK + 1)))
        if rows:
            name = f"x{block + 1}_{parent}"; q = "tasks/todo"
            cfg, units = [], []
            for d, lat, g, st, rule, als in rows:
                for a in als:
                    cfg.append(f"{lat} {a:.2f} {st} {rule}/{g}"); units += [f"{d} {lat} {g} {a:.2f} {s}" for s in (0, 1, 2)]
            # the generator writes OUT/<rule>/<gene>/; OUT is the parent's (same library directories)
            (F / "cfg" / f"{name}.txt").write_text("\n".join(cfg) + "\n"); (F / "units" / f"{name}.txt").write_text("\n".join(units) + "\n")
            (F / q / name).write_text(f'MODEL={k["MODEL"]}\nCK={k["CK"]}\nCFG="{F}/cfg/{name}.txt{flags}"\nOUT={k["OUT"]}\n'
                                      f'UNITS={F}/units/{name}.txt\nNLIB={len(units)}\n')
            new_tasks.append(f"{name} ({len(rows)} latents, {len(units)} libraries)")
        done_eval.add(t)
    (X / "evaluated.txt").write_text("\n".join(sorted(done_eval)) + "\n")
    if dec: pd.DataFrame(dec).to_csv(X / "decisions.csv", index=False)
    for n in new_tasks: print("queued", n, flush=True)
    print("PENDING", pending, flush=True)


if __name__ == "__main__":
    main()
