"""Run the tests without pytest:  python tests/run_tests.py [name-substring ...]

Imports conftest (fixture roots), then every tests/test_*.py, and calls each test_* function; a `tmp_path` argument
gets a fresh temporary directory. `python -m pytest` runs the same tests."""
import importlib
import inspect
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import conftest  # noqa: E402,F401  (sets the PLM_SAE_* roots before plm_sae is imported)


def main(filters):
    failed = passed = 0; t0 = time.time()
    for f in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(f.stem)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__: continue
            if filters and not any(s in f"{f.stem}.{name}" for s in filters): continue
            t = time.time()
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d: fn(Path(d))
                else:
                    fn()
                passed += 1; print(f"PASS {f.stem}.{name} ({time.time() - t:.1f}s)", flush=True)
            except Exception:
                failed += 1; print(f"FAIL {f.stem}.{name}\n{traceback.format_exc()}", flush=True)
    print(f"{passed} passed, {failed} failed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
