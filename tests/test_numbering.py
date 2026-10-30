"""ANARCI numbering (plm_sae.numbering.number): the ANARCI entry point loads with the README's ANARCI, two libraries are
numbered end to end through collect, number and assemble, and a library without any numbered chain gets an empty
formatted table instead of stopping the run. Each test returns early (and says so) when ANARCI or hmmscan is not
installed."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYSSGWYFDY"
      "WGQGTLVTVSS")


def _have_anarci():
    if importlib.util.find_spec("anarci") is None:
        print("skipped: ANARCI not installed")
        return False
    if shutil.which("hmmscan") is None and not (Path(sys.executable).parent / "hmmscan").exists():
        print("skipped: hmmscan not installed")
        return False
    return True


def test_anarci_entry_point():
    if not _have_anarci():
        return
    from plm_sae.numbering import number as N
    _, ali, _ = N._anarci()([(0, VH)], scheme="imgt", assign_germline=True, output=False)
    assert ali[0][0]["germlines"]["j_gene"][0][1].startswith("IGHJ4")


def test_number_units_with_empty_library(tmp_path):
    if not _have_anarci():
        return
    from plm_sae.numbering.number import paths
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "steered_l3_f7_a0.00_seed0_seq1000.txt").write_text(VH + "\n" + VH + "\n")
    (lib / "steered_l3_f7_a8.00_seed0_seq1000.txt").write_text("AAAAAAAAAA\n")   # nothing to number
    units = tmp_path / "units.txt"
    units.write_text(f"{lib} 7 IGHJ4 0.00 0\n{lib} 7 IGHJ4 8.00 0\n")
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
               HMMER_NCPU="1", PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    work = tmp_path / "work"
    for args in (["collect", str(units), str(work)], ["number", str(work), "0", "1", "1"],
                 ["assemble", str(units), str(work), "1"]):
        r = subprocess.run([sys.executable, "-m", "plm_sae.numbering.number", *args], env=env, capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr
    assert "0 problems" in r.stdout, r.stdout
    fmt_ok, fmt_empty = paths(lib, 7, "IGHJ4", "0.00", 0)[2], paths(lib, 7, "IGHJ4", "8.00", 0)[2]
    df = pd.read_csv(fmt_ok)
    assert len(df) == 2 and df.j_gene.str.startswith("IGHJ4").all()
    assert os.path.getsize(fmt_empty) > 0                      # written (counts as done), with no numbered chain
    try:
        pd.read_csv(fmt_empty)
        raise AssertionError("expected an empty formatted table")
    except pd.errors.EmptyDataError:
        pass
