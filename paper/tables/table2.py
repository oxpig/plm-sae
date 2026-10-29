"""Table 2 (steering benchmark of the hyperparameter-matched SAEs) from the deposited benchmark results.

For every model, SAE and set (highest F1, baseline, output-selected; three latents per heavy J gene, 18 per set):
Genes = IGHJ genes (1-6) with at least one latent passing the correlation criterion; Of 18 = latents passing the
correlation criterion, with the latents passing both criteria (correlation and magnitude) in parentheses. Sets and
protocols as in paper.benchmark.rsets (p-IgGen gated, IgLM ungated, ProGen2-OAS T 1.0 with the human FR1 prompt).
Reads benchmark (final_latents.csv, positions/, probe/) and progen2_t10/score/latents.csv; no GPU.

Usage: python -m paper.tables.table2 [--out DIR]     writes DIR/table2.csv and DIR/table2.tex (default: print only)"""
import argparse
from pathlib import Path
import pandas as pd
from paper.benchmark import rsets

MODEL = {"piggen": "p-IgGen", "IgLM": "IgLM", "progen2_oas": "ProGen2-OAS"}
SETS = ["f1", "baseline", "output"]


def table():
    T = rsets.existing_sets()
    T["corr"] = rsets.ok(T.paper_corr); T["both"] = T["corr"] & rsets.ok(T.paper_mag)
    rows = []
    for m in rsets.M:
        for a in rsets.ARCH:
            r = {"model": MODEL[m], "sae": a}
            for s in SETS:
                x = T[(T.model == m) & (T.arch == a) & (T.set == s)]
                assert len(x) == 18, (m, a, s, len(x))
                genes = sorted({int(g.replace("IGHJ", "")) for g in x[x["corr"]].target})
                r[f"{s}_genes"] = ", ".join(map(str, genes)); r[f"{s}_of18"] = f"{int(x['corr'].sum())} ({int(x['both'].sum())})"
            rows.append(r)
    return pd.DataFrame(rows)


def tex(df):
    out = []
    for i, r in df.iterrows():
        first = i % 3 == 0
        if first and i: out.append("\\addlinespace[2pt]")
        cells = [r.model if first else "", r.sae] + [r[f"{s}_{c}"] for s in SETS for c in ("genes", "of18")]
        out.append(" & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None)
    a = ap.parse_args(); df = table()
    print(df.to_string(index=False))
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        df.to_csv(o / "table2.csv", index=False); (o / "table2.tex").write_text(tex(df)); print("wrote", o / "table2.csv", o / "table2.tex")


if __name__ == "__main__":
    main()
