#!/bin/bash
#SBATCH --job-name=plm-sae-script-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Runs one repository shell script on a GPU node (for example the single-machine stage driver
# paper/scripts/quality/run_all.sh, which is idempotent, so a requeued job resumes).
#   examples/slurm/submit.sh examples/slurm/run_script_gpu.slurm --export=ALL,SCRIPT=paper/scripts/quality/run_all.sh
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
[ -n "${SCRIPT:-}" ] || { echo "set SCRIPT"; exit 1; }
bash "$PLM_SAE_REPO/$SCRIPT" ${ARGS:-}
