#!/bin/bash
# One CPU step (python -m MODULE ARGS, NPROC = allocated cores), e.g. the analysis of the property runs:
#   examples/slurm/submit.sh paper/slurm/properties/cpu.slurm --export=ALL,MODULE=paper.analyses.properties.outsel.analyze
#SBATCH --job-name=properties-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
export NPROC=${SLURM_CPUS_PER_TASK:-8}
# ARGS is deliberately unquoted: zero or more arguments
"$PY" -u -m "$MODULE" ${ARGS:-}
