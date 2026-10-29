#!/bin/bash
#SBATCH --job-name=plm-sae-sparsify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Expansion-32 TopK SAEs of Section 4.1 (sparsify; p-IgGen layers 0-3 in one run).
# Submit from a shell that has sourced scripts/env.sh:  sbatch --export=ALL examples/slurm/training/train_sparsify.slurm
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
bash "$PLM_SAE_REPO/scripts/training/train_sparsify.sh"
