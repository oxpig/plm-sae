#!/bin/bash
# ANARCI array: index i numbers task line i+1 of LIST in STATE (scripts/numbering/number_task.sh).
#   examples/slurm/submit.sh paper/slurm/properties/anarci.slurm --array=0-<n-1>%6 --export=ALL,STATE=<dir>,LIST=<file>
#SBATCH --job-name=properties-anarci
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
t=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$LIST")
[ -n "$t" ] || { echo "no task ${SLURM_ARRAY_TASK_ID:-0} in $LIST"; exit 0; }
bash "$PLM_SAE_REPO/scripts/numbering/number_task.sh" "$STATE" "$t" "${SLURM_CPUS_PER_TASK:-32}"
