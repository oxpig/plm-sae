#!/bin/bash
# FlashTAP scores of the 30,754 validation pairs, the labels of the developability gate (one GPU).
#   examples/slurm/submit.sh paper/slurm/properties/flashtap_val.slurm
#SBATCH --job-name=flashtap-val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/properties/flashtap_val.sh"
