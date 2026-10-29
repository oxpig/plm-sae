"""The output-selected latents follow from the output scores alone (no GPU): among latents that fire in at least 1% of the
selection pool, the three with the highest score per SAE and target (plm_sae.steering.pick_output).
  J genes, 3 models x 3 SAEs x 6 genes = 54 targets: score ll_pred of benchmark/probe/scores_<model>.csv; steered latents
    = set "output" of benchmark/final_latents.csv (p-IgGen, IgLM) and progen2_t10/score/latents.csv (ProGen2-OAS).
  CDRH3 properties (Yrich, positive, long) and developability metrics (PPC, PNC, PSH, SFvCSP), p-IgGen, 3 SAEs: the
    per-target columns of outsel_properties/scores/out_scores.csv (oriented so that higher is the steered direction),
    seq_freq from the p-IgGen output scores; steered latents = outsel_properties/manifest.csv sets out_r1-3.
Files are read from <work_dir> where present, else <results_dir>.
Prints how many steered sets equal the top-3 sets and writes <work_dir>/output_selection/picks_check.csv.
Usage: python -m paper.benchmark.check_picks"""
import pandas as pd

from plm_sae import paths as P
from plm_sae.steering.pick_output import MIN_FREQ, top3
from paper.benchmark._inputs import inp

MODELS = ["piggen", "IgLM", "progen2_oas"]
CDRH3 = ("Yrich", "positive", "long")


def j_picks():
    FL = pd.read_csv(inp("benchmark", "final_latents.csv"))
    st = pd.concat([FL[FL.set.isin(["output", "output ungated"])][["model", "arch", "target", "latent"]],
                    pd.read_csv(inp("progen2_t10", "score", "latents.csv")).query("set == 'output'").assign(
                        model="progen2_oas")[["model", "arch", "target", "latent"]]]).drop_duplicates()
    rows = []
    for m in MODELS:
        S = pd.read_csv(inp("benchmark", "probe", f"scores_{m}.csv"))
        for (a, g), x in S.groupby(["arch", "gene"]):
            steered = st[(st.model == m) & (st.arch == a) & (st.target == g)].latent.astype(int).tolist()
            rows.append(dict(family="J", model=m, arch=a, target=g, steered=sorted(steered),
                             top3=top3(x[x.seq_freq >= MIN_FREQ], "ll_pred")))
    return rows


def property_picks():
    OS = pd.read_csv(inp("outsel_properties", "scores", "out_scores.csv"))
    M = pd.read_csv(inp("outsel_properties", "manifest.csv"))
    SF = pd.read_csv(inp("benchmark", "probe", "scores_piggen.csv")).drop_duplicates(["arch", "latent"])[
        ["arch", "latent", "seq_freq"]]
    OS, rows = OS.merge(SF, on=["arch", "latent"]), []
    for (a, t), x in M[M.set.str.startswith("out_r")].groupby(["arch", "target"]):
        o = OS[(OS.arch == a) & (OS.seq_freq >= MIN_FREQ)]
        rows.append(dict(family="CDRH3" if t in CDRH3 else "developability", model="piggen", arch=a, target=t,
                         steered=sorted(x.latent.astype(int).tolist()), top3=top3(o, t)))
    return rows


def check():
    C = pd.DataFrame(j_picks() + property_picks())
    C["same_set"] = [s == sorted(t) for s, t in zip(C.steered, C.top3)]
    return C


def main():
    C = check()
    o = P.work("output_selection")
    o.mkdir(parents=True, exist_ok=True)
    C.to_csv(o / "picks_check.csv", index=False)
    print(C.groupby(["family", "model"], sort=False).agg(targets=("same_set", "size"), same_set=("same_set", "sum")).to_string())
    if not C.same_set.all():
        print("DIFFERENT:\n" + C[~C.same_set].to_string(index=False))


if __name__ == "__main__":
    main()
