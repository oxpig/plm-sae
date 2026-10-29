"""Point every root at the test fixtures before plm_sae.paths is imported (it reads the configuration once)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
os.environ["PLM_SAE_RESULTS_DIR"] = str(FIXTURES / "results")
os.environ["PLM_SAE_WORK_DIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / f"plm_sae_tests_{os.getpid()}")
os.environ.setdefault("MPLBACKEND", "Agg")
