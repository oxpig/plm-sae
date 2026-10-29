#!/bin/bash
# AbLang-2 pseudo-log-likelihood of every library of the steered generation analysis and of the natural references
# (paper.analyses.quality.ablang_scores; interpreter $PLM_SAE_PYTHON_ABLANG). Array tasks share the work through lock
# files. After the per-library rows; then python -m paper.analyses.quality.ablang_summary.
#   examples/slurm/submit.sh paper/slurm/quality/ablang.slurm
#SBATCH --job-name=quality-ablang
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$PLM_SAE_PYTHON_ABLANG" -u -m paper.analyses.quality.ablang_scores run --fp16
