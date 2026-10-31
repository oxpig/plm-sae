"""Joint steering of co-activating latent groups (Supplementary Section A.8, paragraph "Redundancy at the sequence level
does not translate into joint control"): the numbers of the paragraph, from benchmark/final_latents.csv (set joint).
Prints key: value lines and writes <work_dir>/tables/joint_steering.json:
  joint_n / joint_corr / joint_mag      groups steered / passing the correlation / both criteria
  joint_pass_<i>                        passing groups: arch|gene|phi|group size|baseline %|achieved %|candidate latent
  joint_733                             groups around Batch TopK latent 733: corr / both / groups
  joint_<phi>_n, _mag, _corr            per phi threshold: SAE-gene pairs with a group, groups passing
  joint_<phi>_cand_mag, _cand_corr      the group's candidate latent (the pair's highest-F1 latent) steered alone
  joint_<phi>_vs_cand_p                 exact McNemar p (two-sided, paired by SAE-gene pair), group vs candidate, both criteria
  joint_<phi>_bl1_mag, _vs_bl1_p        the same pairs' baseline latent (draw 1), McNemar p
Usage: python -m paper.tables.joint_steering"""
import json

import numpy as np
import pandas as pd

from plm_sae import paths as P
from paper.benchmark import rsets
from paper.benchmark._inputs import inp
from paper.tables.common import mcnemar, ok, pfmt


def paired(x, y, crit):
    """McNemar p for x and y (one latent or group per SAE-gene pair each) on criterion crit."""
    z = x[["arch", "target", crit]].merge(y[["arch", "target", crit]], on=["arch", "target"], suffixes=("_x", "_y"))
    assert len(z) == len(x) == len(y)
    return mcnemar(ok(z[crit + "_x"]), ok(z[crit + "_y"]))


def main():
    FX = {}

    def put(k, v):
        FX[k] = v
        print(f"- {k}: {v}")

    R = pd.read_csv(inp("benchmark", "final_latents.csv"))
    R = R[R.model == "piggen"]
    POS, RD, _ = rsets.rank_tables()
    first = lambda s, T: R[R.set == s].drop(columns=["rank"]).merge(T[T["rank"] == 1], on=rsets.KEY)
    C1, T1 = first("baseline", RD), first("f1", POS)   # baseline draw 1, highest-F1 rank 1
    assert len(C1) == len(T1) == 18, (len(C1), len(T1))
    JT = R[R.set == "joint"]
    for i, e in enumerate(JT[ok(JT.paper_mag)].sort_values(["phi", "arch", "target"]).itertuples()):
        put(f"joint_pass_{i}", f"{e.arch}|{e.target}|{e.phi}|{len(str(e.members).split('+'))}|{100 * e.paper_base:.1f}|"
                               f"{100 * e.paper_ach:.1f}|{int(e.latent)}")
    c733 = JT[JT.latent == 733]
    put("joint_733", f"{ok(c733.paper_corr).sum()}/{ok(c733.paper_mag).sum()}/{len(c733)}")
    put("joint_n", str(len(JT)))
    put("joint_corr", str(ok(JT.paper_corr).sum()))
    put("joint_mag", str(ok(JT.paper_mag).sum()))
    for ph in ("phi05", "phi07"):
        g = JT[JT.phi == ph]
        pairs = set(zip(g.arch, g.target))
        in_pairs = lambda T: T[np.array([(a, t) in pairs for a, t in zip(T.arch, T.target)])]
        c, t = in_pairs(C1), in_pairs(T1)
        assert len(c) == len(t) == len(pairs) == len(g)
        assert set(zip(g.arch, g.target, g.latent)) == set(zip(t.arch, t.target, t.latent)), \
            "joint candidate is not the pair's highest-F1 latent"
        put(f"joint_{ph}_n", str(len(g)))
        put(f"joint_{ph}_mag", str(ok(g.paper_mag).sum()))
        put(f"joint_{ph}_corr", str(ok(g.paper_corr).sum()))
        put(f"joint_{ph}_bl1_mag", str(ok(c.paper_mag).sum()))
        put(f"joint_{ph}_vs_bl1_p", pfmt(paired(g, c, "paper_mag")))
        put(f"joint_{ph}_cand_mag", str(ok(t.paper_mag).sum()))
        put(f"joint_{ph}_cand_corr", str(ok(t.paper_corr).sum()))
        put(f"joint_{ph}_vs_cand_p", pfmt(paired(g, t, "paper_mag")))
    print("\n" + JT[["phi", "arch", "target", "latent", "members", "paper_base", "paper_ach", "paper_corr",
                     "paper_mag"]].round(3).to_string(index=False))
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    json.dump(FX, open(o / "joint_steering.json", "w"), indent=1)


if __name__ == "__main__":
    main()
