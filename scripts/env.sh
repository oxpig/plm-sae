# Sourced by the shell scripts of this repository. Exports the locations of configs/paths.yaml as PLM_SAE_* variables
# (python -m plm_sae.paths --sh), puts the repository and third_party/ on PYTHONPATH and sets PY, the main interpreter
# (PLM_SAE_PYTHON if set, otherwise `python`).
_PLM_SAE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(cd "$_PLM_SAE_REPO" && "${PLM_SAE_PYTHON:-python}" -m plm_sae.paths --sh)" \
  || { echo "env.sh: cannot run plm_sae.paths with ${PLM_SAE_PYTHON:-python}" >&2; return 1 2>/dev/null || exit 1; }
export PYTHONPATH="$PLM_SAE_REPO:$PLM_SAE_REPO/third_party${PYTHONPATH:+:$PYTHONPATH}"
PY="$PLM_SAE_PYTHON"
log(){ echo "[$(date '+%F %T')] $*"; }
