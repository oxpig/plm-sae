#!/bin/bash
# GPU analyses of the TopK SAE of Section 4.1 (paper/scripts/identification/section41_analyses.sh), one array task each:
# 0 CDR identity probe, 1 residue location, 2 FVU / L0 / never-fired share.
#   examples/slurm/submit.sh paper/slurm/identification/section41_analyses.slurm
#SBATCH --job-name=identification-section41
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
WHAT=(cdr residue stats)
bash "$PLM_SAE_REPO/paper/scripts/identification/section41_analyses.sh" "${WHAT[${SLURM_ARRAY_TASK_ID:-0}]}"
