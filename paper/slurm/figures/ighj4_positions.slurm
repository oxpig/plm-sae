#!/bin/bash
#SBATCH --job-name=plm-sae-ighj4-positions
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Supplementary Figure 6 (table, encode, plot; idempotent).   examples/slurm/submit.sh paper/slurm/figures/ighj4_positions.slurm
# Figures go to <work_dir>/figures (pass another directory with --export=ALL,OUTDIR=...).
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
bash "$PLM_SAE_REPO/paper/scripts/figures/ighj4_positions.sh" "${OUTDIR:-$PLM_SAE_WORK_DIR/figures}"
