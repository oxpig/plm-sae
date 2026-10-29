"""Ladder extension and refinement of the ProGen2-OAS benchmark, with the benchmark's rules: extension when the pooled
coverage at the last alpha is >= 80% (5 alphas spanning (A_k, A_k + 30], at most two blocks; tasks t10x<block>_<parent>);
refinement once a latent is no longer extended, when fewer than five alphas are in the valid range and the range ends
before the last alpha (seven alphas a_hi * k / 8, k = 1..7, minus those already run; task t10r_<parent>).
A latent is evaluated only when all its tasks are done and numbered. Run repeatedly by
paper/scripts/benchmark/progen2_t10_ladder.sh; prints QUEUED / PENDING / FINAL. Usage: python -m paper.benchmark.progen2_t10.ladder"""
import os
from pathlib import Path
import numpy as np
from paper.benchmark.progen2_t10.score import experiments, kv, af, B
A, NBLOCK, MAXBLOCK, NLAT = 30.0, 5, 2, 162


def pooled_cov(d, lat, g, alphas):
    return np.array([sum(af.lib_stats((d, lat, g, a, s))["n_Hj"] for s in (0, 1, 2)) / 3000 for a in alphas])


def main():
    E = experiments(); queued, pending, final = [], 0, 0
    for e in E.itertuples():
        if e.pending or not e.alphas: pending += 1; continue
        al = np.array(e.alphas); cov = pooled_cov(e.dir, e.latent, e.target, al)
        kids = [t for q in ("done", "claimed", "todo", "failed") for t in os.listdir(B / "tasks" / q) if "." not in t and t.endswith("_" + e.task)]
        block = sum(t.startswith("t10x") for t in kids); refined = any(t.startswith("t10r") for t in kids)
        new, name = [], None
        if cov[-1] >= 0.8 and block < MAXBLOCK:
            new = list(al.max() + A / NBLOCK * np.arange(1, NBLOCK + 1)); name = f"t10x{block + 1}_{e.task}"
        elif not refined:
            bad = np.where(cov < 0.8)[0]; k = bad[0] if len(bad) else len(al)
            if 0 < k < 5 and k < len(al):
                a_hi = al[k]; new = [round(a_hi * j / 8, 2) for j in range(1, 8)]; new = [a for a in new if not np.any(np.abs(al - a) < 1e-6)]
                name = f"t10r_{e.task}"
        if not new: final += 1; continue
        k0 = kv(B / "tasks/done" / e.task); tag = "/".join(Path(e.dir).parts[-2:])
        (B / "cfg" / f"{name}.txt").write_text("".join(f"{e.latent} {a:.2f} {e.start} {tag}\n" for a in new))
        units = [f"{e.dir} {e.latent} {e.target} {a:.2f} {s}" for a in new for s in (0, 1, 2)]
        (B / "units" / f"{name}.txt").write_text("\n".join(units) + "\n")
        cfg_flags = k0["CFG"].split(" ", 1)[1]
        (B / "tasks/todo" / name).write_text(f'MODEL={k0["MODEL"]}\nCK={k0["CK"]}\nCFG="{B}/cfg/{name}.txt {cfg_flags}"\nOUT={k0["OUT"]}\n'
                                                 f'UNITS={B}/units/{name}.txt\nNLIB={len(units)}\n')
        queued.append(name); print("queued", name, [float(a) for a in new], flush=True)
    print(f"QUEUED {len(queued)} PENDING {pending} FINAL {final} of {len(E)}", flush=True)
    assert len(E) == NLAT


if __name__ == "__main__":
    main()
