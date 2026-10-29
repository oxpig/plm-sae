#!/bin/bash
# Single-latent (indices 0-17) and co-activation group (18-39) ablation, one GPU per task, after
# `python -m paper.analyses.ablation.make_tasks`.
#   examples/slurm/submit.sh paper/slurm/ablation/ablate.slurm
#SBATCH --job-name=ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-39
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/ablation/ablate.sh" "${SLURM_ARRAY_TASK_ID:-0}"
