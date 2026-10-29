"""Supplementary Tables 20-22 (developability metrics and graded CDRH3 properties; output-selected, highest-F1 and
baseline latents of the p-IgGen SAEs) from the property results (paper.analyses.properties.outsel.analyze):
outsel_properties/results_graded.csv (Y-rich, positive and long CDRH3) and results_dev.csv (FlashTAP metrics).
Writes, in <work_dir>/tables/outsel_properties:
  tab_graded.tex          Supplementary Table 21 (latents passing per set, top 1 / 2 / 3, both criteria in parentheses)
  tab_graded_latents.tex  Supplementary Table 22 (every graded latent: base -> value at alpha*, alpha*, verdict)
  tab_dev_latents.tex     Supplementary Table 20 (every developability latent)
and prints the per-set counts quoted in the text. Usage: python -m paper.tables.outsel_properties"""
import pandas as pd

from plm_sae import paths as P
from paper.benchmark._inputs import inp

SAES = ["TopK", "Batch TopK", "Ordered"]
PR = [("Yrich", "Y-rich"), ("positive", "Positive"), ("long", "Long")]
SETS = [("id", "Highest $F_1$"), ("out", "Output-selected"), ("base", "Baseline")]
MT = ["PPC", "PNC", "PSH", "SFvCSP"]
fmt = lambda m, v: f"{v:.1f}" if m in ("PSH", "SFvCSP") else f"{v:.2f}"


def ver(r):
    return "both" if r.corr == "pass" and r.mag == "pass" else ("corr." if r.corr == "pass" else "fail")


def rows(df, s):
    x = df[df.set.str.startswith(s)].copy()
    x["rk"] = x.set.str[-1].astype(int)
    return x.sort_values("rk")


def tab_graded(G):
    out = []
    for a in SAES:
        for i, (s, lab) in enumerate(SETS):
            cells, props = [], []
            for k in (1, 2, 3):
                n = nb = 0
                for t, tl in PR:
                    x = rows(G[(G.arch == a) & (G.target == t)], s).head(k)
                    n += (x["corr"] == "pass").sum()
                    nb += ((x["corr"] == "pass") & (x.mag == "pass")).sum()
                    if k == 3 and (x["corr"] == "pass").any():
                        props.append(tl)
                cells.append(f"{n} ({nb})")
            out.append(f"{a if i == 0 else ''} & {lab} & {', '.join(props) if props else 'None'} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_graded_latents(G):
    out = []
    for a in SAES:
        for j, (t, tl) in enumerate(PR):
            for i, (s, lab) in enumerate(SETS):
                x = rows(G[(G.arch == a) & (G.target == t)], s)
                c = [f"{int(r.latent)}: {100 * r.base:.0f}$\\to${100 * r.at_astar:.0f}, $\\alpha^*$ {r.astar_med:g} ({ver(r)})" for r in x.itertuples()]
                out.append(f"{a if (i == 0 and j == 0) else ''} & {tl if i == 0 else ''} & {lab} & " + " & ".join(c) + r" \\")
            out.append(r"\cmidrule(lr){2-6}" if j < 2 else r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def tab_dev_latents(D):
    out = []
    for a in SAES:
        for j, t in enumerate(MT):
            for i, (s, lab) in enumerate(SETS[1:]):
                x = rows(D[(D.arch == a) & (D.target == t)], s)
                c = [f"{int(r.latent)}: {fmt(t, r.base)}$\\to${fmt(t, r.at_astar)}, $\\alpha^*$ {r.astar_med:g} ({'pass' if r.corr == 'pass' else 'fail'})" for r in x.itertuples()]
                out.append(f"{a if (i == 0 and j == 0) else ''} & {t if i == 0 else ''} & {lab} & " + " & ".join(c) + r" \\")
            out.append(r"\cmidrule(lr){2-6}" if j < 3 else r"\midrule")
    return "\n".join(out[:-1]) + "\n"


def summary(G, D):
    for s in ("id", "out", "base"):
        for a in SAES:
            x = G[(G.arch == a) & G.set.str.startswith(s)]
            print("graded", s, a, (x["corr"] == "pass").sum(), ((x["corr"] == "pass") & (x.mag == "pass")).sum())
    for s in ("out", "base"):
        for a in SAES:
            x = D[(D.arch == a) & D.set.str.startswith(s)]
            print("dev", s, a, {t: int((x[x.target == t]["corr"] == "pass").sum()) for t in MT})
    o = G[G.set.str.startswith("out")]
    print("graded out alpha*:", o.astar_med.min(), o.astar_med.max(), "| starts", sorted(G.start.unique()), sorted(D.start.unique()))
    print("max alpha_max", G.alpha_max.max(), D.alpha_max.max())


def main():
    G = pd.read_csv(inp("outsel_properties", "results_graded.csv"))
    D = pd.read_csv(inp("outsel_properties", "results_dev.csv"))
    o = P.work("tables", "outsel_properties")
    o.mkdir(parents=True, exist_ok=True)
    (o / "tab_graded.tex").write_text(tab_graded(G))
    (o / "tab_graded_latents.tex").write_text(tab_graded_latents(G))
    (o / "tab_dev_latents.tex").write_text(tab_dev_latents(D))
    summary(G, D)
    print("wrote", o / "tab_graded.tex", o / "tab_graded_latents.tex", o / "tab_dev_latents.tex")


if __name__ == "__main__":
    main()
