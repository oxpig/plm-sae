#!/bin/bash
# Check of the CDRH3 parser against the IMGT CDR3 (paper.analyses.properties.cdr3_check; CPU).
#   examples/slurm/submit.sh paper/slurm/properties/cdr3_check.slurm
#SBATCH --job-name=properties-cdr3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
NPROC=${SLURM_CPUS_PER_TASK:-8} "$PY" -u -m paper.analyses.properties.cdr3_check
