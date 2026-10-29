#!/bin/bash
# Dictionary redundancy of the nine hyperparameter-matched SAEs (CPU).
#   examples/slurm/submit.sh paper/slurm/ablation/redundancy.slurm
#SBATCH --job-name=redundancy
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/redundancy/run.sh"
