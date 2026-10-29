#!/bin/bash
#SBATCH --job-name=plm-sae-lr-screen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
# Learning-rate screen of one SAE: one 10,000-step run per learning rate (1e-7 1e-6 1e-5 1e-4 1e-3).
# Submit from a shell that has sourced scripts/env.sh:
#   sbatch --export=ALL,NAME=piggen_ordered_r8 examples/slurm/training/lr_screen.slurm
# then, on a GPU node: scripts/training/lr_screen.sh rank piggen_ordered_r8
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
LRS=(1e-7 1e-6 1e-5 1e-4 1e-3)
bash "$PLM_SAE_REPO/scripts/training/lr_screen.sh" train "${NAME:?set NAME}" "${LRS[$SLURM_ARRAY_TASK_ID]}"
