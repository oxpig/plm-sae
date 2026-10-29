#!/bin/bash
#SBATCH --job-name=plm-sae-script
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Runs one repository shell script as a CPU job (short steps between array jobs, e.g. the ranking lists after the gates).
#   examples/slurm/submit.sh examples/slurm/run_script.slurm --export=ALL,SCRIPT=paper/scripts/identification/make_pair_lists.sh
# SCRIPT is relative to the repository root; ARGS (optional) are passed to it.
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
[ -n "${SCRIPT:-}" ] || { echo "set SCRIPT"; exit 1; }
bash "$PLM_SAE_REPO/$SCRIPT" ${ARGS:-}
