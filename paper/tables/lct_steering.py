"""Light-chain-type steering (Supplementary Table 15, tab:lct_steering): LaTeX table body and the counts of the
paragraph. Rows of the benchmark's final_latents.csv with set "light chain" (p-IgGen, three highest-F1 latents per class and SAE):
target-type proportion of numbered light chains at alpha = 0 and at alpha*, correlation and magnitude criteria. Reads
benchmark/final_latents.csv; prints the body and counts and writes <work_dir>/tables/lct_steering.tex.
Usage: python -m paper.tables.lct_steering"""
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

ARCH = ["TopK", "Batch TopK", "Ordered"]
ok = lambda s: s.astype(str).isin(["pass", "True"])


def main():
    R = pd.read_csv(inp("benchmark", "final_latents.csv"))
    LC = R[R.set == "light chain"].copy()
    lr = []
    for a in ARCH:
        x = LC[LC.arch == a]
        for cls in ("kappa", "lambda"):
            y = x[x.target == cls]
            for i, e in enumerate(y.itertuples()):
                lr.append(f"{a if (cls == 'kappa' and i == 0) else ''} & {cls if i == 0 else ''} & {int(e.latent)} & {e.paper_base:.3f} & {e.paper_ach:.3f} & "
                          f"{'pass' if e.paper_corr == 'pass' else '--'} & {'pass' if e.paper_mag == 'pass' else '--'} \\\\")
        lr.append(r"\midrule")
    body = "\n".join(lr[:-1])
    o = P.work("tables")
    o.mkdir(parents=True, exist_ok=True)
    (o / "lct_steering.tex").write_text(body + "\n")
    print(body)
    kp = LC[(LC.target == "kappa") & ok(LC.paper_corr)].sort_values("paper_ach", ascending=False)
    for i, e in enumerate(kp.itertuples()):
        print(f"% kappa pass {i}: {e.arch} latent {int(e.latent)} {e.paper_base:.2f} -> {e.paper_ach:.2f}, magnitude {e.paper_mag}")
    for cls in ("kappa", "lambda"):
        y = LC[LC.target == cls]
        print(f"% {cls}: correlation {ok(y.paper_corr).sum()}/{len(y)}, both {ok(y.paper_mag).sum()}/{len(y)}")


if __name__ == "__main__":
    main()
