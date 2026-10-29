#!/bin/bash
# FlashTAP scores of the developability runs, shard SLURM_ARRAY_TASK_ID of NSHARDS (paper/scripts/properties/submit_tap.sh).
#SBATCH --job-name=properties-tap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
"$PLM_SAE_PYTHON_FLASHTAP" -u -m paper.analyses.properties.outsel.tap "${SLURM_ARRAY_TASK_ID:-0}" "${NSHARDS:-1}"
