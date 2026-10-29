#!/bin/bash
# Generation array: index i runs task line i+1 of LIST in STATE (scripts/generation/run_queued_task.sh), one GPU each.
#   examples/slurm/submit.sh paper/slurm/properties/gen.slurm --array=0-<n-1>%10 --export=ALL,STATE=<dir>,LIST=<file>
#SBATCH --job-name=properties-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
t=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$LIST")
[ -n "$t" ] || { echo "no task ${SLURM_ARRAY_TASK_ID:-0} in $LIST"; exit 0; }
bash "$PLM_SAE_REPO/scripts/generation/run_queued_task.sh" "$STATE" "$t"
