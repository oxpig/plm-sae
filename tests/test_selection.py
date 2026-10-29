"""Output-selected latents from the output scores on the selection pool, and the pool draw itself."""
import pandas as pd


def test_output_picks_reproduce():
    """Top 3 by score among latents firing in >= 1% of the selection pool = the steered output-selected latents:
    54 of 54 J-gene, 9 of 9 CDRH3 and 12 of 12 developability sets (fixtures: the top 50 latents per SAE and target of
    the deposited scores)."""
    from paper.benchmark import check_picks
    C = check_picks.check()
    n = C.groupby("family").same_set.agg(["size", "sum"])
    assert n.loc["J"].tolist() == [54, 54]
    assert n.loc["CDRH3"].tolist() == [9, 9]
    assert n.loc["developability"].tolist() == [12, 12]


def test_pool_is_one_draw_per_gene():
    """The pool function draws each gene's sample with the configured seed; synthetic data, no dataset needed."""
    from plm_sae.steering import selection_pool as pool
    val = pd.DataFrame({"VHorVHH": [f"S{i}" for i in range(4000)],
                        "gene": [f"IGHJ{1 + i % 6}" for i in range(3990)] + ["IGHJ7"] * 5 + ["nan"] * 5})
    a = pool.validation_pool(val.copy())
    b = val[val.gene.str.match(r"IGHJ[1-6]$")].groupby("gene", group_keys=False).apply(
        lambda d: d.sample(n=min(pool.PER_GENE, len(d)), random_state=pool.SEED))
    assert a.index.equals(b.index) and len(a) == 6 * pool.PER_GENE and pool.SEED == 1 and pool.PER_GENE == 300
