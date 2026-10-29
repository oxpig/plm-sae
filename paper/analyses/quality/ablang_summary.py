"""AbLang-2 pseudo-log-likelihood of each library of the steered generation analysis against the natural sequences
(paired for p-IgGen, heavy chains for IgLM and ProGen2-OAS), with the test of the other metrics
(paper.analyses.quality.common.mwu). Writes <work_dir>/quality/per_library.csv (tag, model, arch, set, latent, alpha,
rows, n, mean, p, r). Usage: python -m paper.analyses.quality.ablang_summary"""
import numpy as np
import pandas as pd

from paper.analyses.quality.common import B, mwu


def main():
    sc = B / "ablang" / "scores"
    nat = {"paired": np.load(sc / "natural_paired.npy"), "heavy": np.load(sc / "natural_heavy.npy")}
    rows = []
    for r in pd.read_csv(B / "list.csv").itertuples():
        base = dict(tag=r.tag, model=r.model, arch=r.arch, set=r.set, latent=r.latent, alpha=r.alpha)
        f = sc / f"{r.tag}.npy"
        v = np.load(f) if f.exists() else np.array([np.nan])
        if not np.isfinite(v).any():
            rows.append(base)
            continue
        m = mwu(v, nat["paired" if r.model == "piggen" else "heavy"])
        rows.append(dict(base, rows=len(v), n=m["n_steered"], mean=m["mean_steered"], p=m["p"], r=m["rank_biserial"]))
    D = pd.DataFrame(rows)
    D.to_csv(B / "per_library.csv", index=False)
    print(D.groupby(["model", "set"])["mean"].median().to_string())


if __name__ == "__main__":
    main()
