#!/bin/bash
# FlashTAP scoring of the matched random-latent controls (dev_ctrl_list.csv), dealt to 6 array tasks, one GPU each.
#   examples/slurm/submit.sh paper/slurm/properties/dev_ctrl.slurm
#SBATCH --job-name=dev-ctrl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/properties/dev_ctrl.sh" "${SLURM_ARRAY_TASK_ID:-0}"
