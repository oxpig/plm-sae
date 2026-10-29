#!/bin/bash
# Natural references of the steered generation analysis (paper.analyses.quality.natural).
#   examples/slurm/submit.sh paper/slurm/quality/natural.slurm
#SBATCH --job-name=quality-natural
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$PY" -m paper.analyses.quality.natural
