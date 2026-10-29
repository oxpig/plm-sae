#!/bin/bash
# Binder probe and binder-label feature gate of the hyperparameter-matched p-IgGen TopK and Ordered SAEs
# (Supplementary Section A.11; CPU).   examples/slurm/submit.sh paper/slurm/properties/binder_probe.slurm
#SBATCH --job-name=binder-probe
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/binder/probe.sh"
