"""Where the data, the outputs and the external code are. The locations are set in configs/paths.yaml; an environment
variable PLM_SAE_<KEY> (key in upper case, e.g. PLM_SAE_DATA_DIR) overrides a key. Relative paths are relative to the
repository root.

  dataset(*p)      <data_dir>/datasets/...        sequence datasets of the data record
  checkpoints()    <data_dir>/checkpoints         SAE checkpoints of the data record (<name>.safetensors + <name>.json)
  libraries(*p)    <data_dir>/libraries/...       steered libraries of the data record
  results(*p)      <results_dir>/...              result tables read by the table and figure scripts
  work(*p)         <work_dir>/...                 outputs of pipeline runs
  activations(*p)  <work_dir>/activations/...     activation caches written by plm_sae.activations
  progen2_dir()    the progen2/ directory of the salesforce/progen clone
  python(env)      interpreter of the main, flashtap or ablang environment

`python -m plm_sae.paths` prints the resolved locations; `python -m plm_sae.paths --sh` prints them as shell exports
(scripts/env.sh)."""
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(REPO / "configs" / "paths.yaml")) or {}


def get(key):
    """Value of a paths.yaml key after the PLM_SAE_<KEY> override (None if unset)."""
    v = os.environ.get("PLM_SAE_" + key.upper(), CFG.get(key))
    return None if v in (None, "") else v


def _dir(key, default=None):
    v = get(key)
    if v is None:
        return default
    p = Path(os.path.expanduser(str(v)))
    return p if p.is_absolute() else REPO / p


def data_dir(): return _dir("data_dir")
def work_dir(): return _dir("work_dir")
def results_dir(): return _dir("results_dir", data_dir() / "results")
def progen2_dir(): return _dir("progen2_dir")


def dataset(*p): return data_dir().joinpath("datasets", *map(str, p))
def checkpoints(): return data_dir() / "checkpoints"
def libraries(*p): return data_dir().joinpath("libraries", *map(str, p))
def results(*p): return results_dir().joinpath(*map(str, p))
def work(*p): return work_dir().joinpath(*map(str, p))
def activations(*p): return work_dir().joinpath("activations", *map(str, p))


def add_progen2_to_syspath():
    """Make `from models.progen.modeling_progen import ProGenForCausalLM` import the ProGen2 code of progen2_dir."""
    p = str(progen2_dir())
    if p not in sys.path:
        sys.path.insert(0, p)


def python(env="main"):
    """Interpreter of an environment: main (this one), flashtap (envs/flashtap.yml) or ablang (envs/ablang2.yml)."""
    v = None if env == "main" else get(f"python_{env}")
    return str(v) if v else sys.executable


def _sh():
    out = {"PLM_SAE_REPO": REPO, "PLM_SAE_DATA_DIR": data_dir(), "PLM_SAE_WORK_DIR": work_dir(),
           "PLM_SAE_RESULTS_DIR": results_dir(), "PLM_SAE_ACTIVATIONS": activations(),
           "PLM_SAE_PYTHON": python(), "PLM_SAE_PYTHON_FLASHTAP": python("flashtap"),
           "PLM_SAE_PYTHON_ABLANG": python("ablang")}
    return "\n".join(f"export {k}='{v}'" for k, v in out.items())


if __name__ == "__main__":
    if "--sh" in sys.argv:
        print(_sh())
    else:
        for k, v in (("data_dir", data_dir()), ("results_dir", results_dir()), ("work_dir", work_dir()),
                     ("progen2_dir", progen2_dir()), ("python_flashtap", python("flashtap")),
                     ("python_ablang", python("ablang"))):
            print(f"{k:16s} {v}")
