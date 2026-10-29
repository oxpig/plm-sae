"""Inputs of the steering benchmark that other stages produce or that come with the data record.

Each file is looked up first in the pipeline outputs (<work_dir>/<stage>/...) and, if absent there, in the deposited
results (<results_dir>/<stage>/...), so the benchmark runs either after the earlier stages or from the data record.

  identification/pairs/pairs_<sae>_top5.txt      top-5 latents per J gene by validation F1 (lines "IGHJ<k>:<latent>"
                                                 in rank order), one file per matched SAE (make_pair_lists.sh)
  identification/pairs/pairs_<sae>_lct.txt       light-chain type (kappa / lambda) latents of the p-IgGen SAEs
  identification/lowidx/selection.csv            Ordered latents of lowest index, steered in the benchmark (they are
                                                 excluded from the baseline draw)
  identification/gates/piggen_topk_r32_layer3/feature_f1_scores_class_ighj<k>.csv
                                                 F1 ranking of the TopK SAE of Section 4.1
  redundancy/coactivation_groups.txt             the 22 co-activation groups, lines "phi sae gene m1+m2+..." (first
                                                 member = the highest-F1 latent)
  libraries/piggen_alpha0_reference/             3,000 unsteered p-IgGen sequences on which the start positions of the
                                                 Section 4.1 latents were measured
<sae> = an SAE name of the data record, e.g. piggen_topk_r8, iglm_ordered_r8, progen2oas_batchtopk_r8."""
import pandas as pd

from plm_sae import paths as P

GENES = [f"IGHJ{i}" for i in range(1, 7)]
SECTION41_CASE = [("IGHJ4", 463)]   # case-study latent of Supplementary Figures 6 and 7


def inp(stage, *p):
    s = P.work(stage, *p)
    return s if s.exists() else P.results(stage, *p)


def ranking(fn):
    """fn = 'pairs_<sae>_top5.txt' or 'pairs_<sae>_lct.txt'."""
    return inp("identification", "pairs", fn)


def lowidx_selection():
    return inp("identification", "lowidx", "selection.csv")


def section41_f1(gene):
    """Per-latent F1 table of the Section 4.1 gate for gene 'IGHJ1'..'IGHJ6'."""
    return inp("identification", "gates", "piggen_topk_r32_layer3", f"feature_f1_scores_class_{gene.lower()}.csv")


def section41_latents():
    """(gene, latent, F1 rank) of the steered latents of the TopK SAE of Section 4.1: the three highest-F1 latents of
    each gene and the case-study latent 463 (IGHJ4, F1 rank 5)."""
    out = []
    for g in GENES:
        ids = pd.read_csv(section41_f1(g)).neuron_id.astype(int).tolist()
        out += [(g, l, k + 1) for k, l in enumerate(ids[:3])]
        out += [(g, l, ids.index(l) + 1) for gg, l in SECTION41_CASE if gg == g]
    return out


def coactivation_groups():
    """(phi, sae, gene, [latents]) of the 22 co-activation groups, in file order."""
    for line in open(inp("redundancy", "coactivation_groups.txt")):
        if line.strip():
            phi, sae, gene, lats = line.split()
            yield phi, sae, gene, [int(x) for x in lats.split("+")]


def alpha0_reference():
    """The three unsteered p-IgGen libraries (seeds 0-2), sorted by file name."""
    return sorted(str(f) for f in P.libraries("piggen_alpha0_reference").glob("steered_l3_f1516_a0.00_seed*_seq1000.txt"))
