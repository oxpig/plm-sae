#!/bin/bash
# GPU analyses of the TopK SAE of Section 4.1 (p-IgGen block 3):
#   cdr      CDR residue-identity probe (Section 4.1, accuracy 0.96)            -> identification/cdr_probe/
#   residue  IMGT position of IGHJ4 latents 463, 4720, 6276 (Supplementary A.5)  -> identification/residue_location/
#   stats    FVU, L0, never-fired share (Supplementary Table 4 caption)          -> identification/reconstruction/
#   paper/scripts/identification/section41_analyses.sh {cdr|residue|stats}
# One GPU; cdr about 1 h, residue and stats under 1 h each (ANARCI uses 8 CPUs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
case "${1:?usage: section41_analyses.sh cdr|residue|stats}" in
  cdr)     "$PY" -m paper.analyses.identification.cdr_identity_probe --n_train 600 --n_val 1000 --bs 16 ;;
  residue) "$PY" -m paper.analyses.identification.residue_location measure 4000
           "$PY" -m paper.analyses.identification.residue_location report ;;
  stats)   "$PY" -m paper.analyses.identification.section41_stats ;;
  *) echo "unknown analysis $1" >&2; exit 2 ;;
esac
