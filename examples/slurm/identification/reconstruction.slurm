#!/bin/bash
# FVU, L0 and dead fraction of the hyperparameter-matched SAEs (scripts/identification/run_reconstruction.sh), one
# array task per model: 0 p-IgGen, 1 IgLM, 2 ProGen2-OAS, 3 p-IgGen expansion-32 TopK.
# Submit from the repository root:
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD" examples/slurm/identification/reconstruction.slurm
#SBATCH --job-name=identification-reconstruction
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
KEYS=(piggen IgLM progen2_oas piggen_exp32)
bash "$PLM_SAE_REPO/scripts/identification/run_reconstruction.sh" "${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
