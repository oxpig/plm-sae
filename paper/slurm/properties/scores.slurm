#!/bin/bash
# Latent positions, selection-pool firing statistics and output scores of the p-IgGen SAEs (one GPU).
#   examples/slurm/submit.sh paper/slurm/properties/scores.slurm
#SBATCH --job-name=properties-scores
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=03:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
"$PY" -u -m paper.analyses.properties.outsel.scores
