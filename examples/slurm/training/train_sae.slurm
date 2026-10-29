#!/bin/bash
#SBATCH --job-name=plm-sae-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-9
# The ten hyperparameter-matched SAEs of the paper, one per array task (order of SAE_RUNS in scripts/training/sae_runs.sh).
# Submit from a shell that has sourced scripts/env.sh:  sbatch --export=ALL examples/slurm/training/train_sae.slurm
# One SAE: add --array=<index>, or run scripts/training/train_sae.sh NAME on a GPU node.
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
source "$PLM_SAE_REPO/scripts/training/sae_runs.sh"
read -r -a NAMES <<< "$SAE_RUNS"
bash "$PLM_SAE_REPO/scripts/training/train_sae.sh" "${NAMES[$SLURM_ARRAY_TASK_ID]}"
