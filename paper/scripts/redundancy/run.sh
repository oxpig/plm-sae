#!/bin/bash
# Dictionary redundancy of the nine hyperparameter-matched SAEs (Supplementary Table 16, Section 4.2.2, Supplementary
# Section A.8), CPU. Needs the identification gates, the firing counts of the reconstruction step and the validation
# activations of the nine SAEs. Writes <work_dir>/redundancy/redundancy.csv.
# Usage: paper/scripts/redundancy/run.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
"$PY" -u -m paper.analyses.redundancy.redundancy
