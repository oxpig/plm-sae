#!/bin/bash
# FlashTAP scoring of the developability steering libraries of latents 980 (index 0) and 1088 (index 1), one GPU each.
#   examples/slurm/submit.sh paper/slurm/properties/dev_score.slurm
#SBATCH --job-name=dev-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/properties/dev_score.sh" "${SLURM_ARRAY_TASK_ID:-0}"
