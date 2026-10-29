"""Regenerate manuscript tables from the small result files in tests/fixtures/results and compare them with the
manuscript table bodies (tests/fixtures/expected, the rows between the header rule and the bottom rule)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from conftest import FIXTURES

EXPECTED = FIXTURES / "expected"
REPO = Path(__file__).resolve().parents[1]


def expected(name):
    return (EXPECTED / name).read_text()


def run(module, work):
    env = dict(os.environ, PLM_SAE_WORK_DIR=str(work))
    subprocess.run([sys.executable, "-m", module], check=True, cwd=REPO, capture_output=True, env=env)
    return Path(work) / "tables"


def test_table2_benchmark():
    from paper.tables import table2
    assert table2.tex(table2.table()) == expected("table2.tex")


def test_output_sets_are_the_picks():
    """Every output-selected set of the result tables is the top three of plm_sae.steering.pick_output."""
    from paper.benchmark import rsets
    T = rsets.existing_sets()
    _, _, OR = rsets.rank_tables()
    o = T[T.set == "output"].groupby(["model", "arch", "target"]).latent.apply(set)
    p = OR[OR["rank"] <= 3].groupby(["model", "arch", "target"]).latent.apply(set)
    assert len(o) == 54 and o.equals(p.loc[o.index])


def test_supplementary_tables_20_21_22(tmp_path):
    from paper.tables import outsel_properties as op
    U = FIXTURES / "results" / "outsel_properties"
    G, D = pd.read_csv(U / "results_graded.csv"), pd.read_csv(U / "results_dev.csv")
    assert op.tab_graded(G) == expected("tab_graded.tex")                  # Supplementary Table 21
    assert op.tab_graded_latents(G) == expected("tab_graded_latents.tex")  # Supplementary Table 22
    assert op.tab_dev_latents(D) == expected("tab_dev_latents.tex")        # Supplementary Table 20
    out = run("paper.tables.outsel_properties", tmp_path) / "outsel_properties"
    for f in ("tab_graded.tex", "tab_graded_latents.tex", "tab_dev_latents.tex"):
        assert (out / f).read_text() == expected(f), f


def test_supplementary_tables_7_8(tmp_path):
    """Supplementary Table 7 (per-pair rows) and 8 (passing latents by depth), written to <work_dir>/tables."""
    assert (run("paper.tables.rl_perpair", tmp_path) / "rl_perpair.tex").read_text() == expected("rl_perpair.tex")
    assert (run("paper.tables.rl_depth", tmp_path) / "rl_depth.tex").read_text() == expected("rl_depth.tex")


def test_supplementary_table_19(tmp_path):
    """Developability latent counts (live latents: active at a residue of at least one validation sequence)."""
    assert (run("paper.tables.dev_counts", tmp_path) / "dev_counts.tex").read_text() == expected("dev_counts.tex")
