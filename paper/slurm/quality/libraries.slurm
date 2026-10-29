#!/bin/bash
# One row of <work_dir>/quality/list.csv (python -m paper.analyses.quality.libraries) per array task
# (paper.analyses.quality.run_one). Submit rows 0-2 (the unsteered references) first, then the rest:
#   examples/slurm/submit.sh paper/slurm/quality/libraries.slurm --array=0-2
#   examples/slurm/submit.sh paper/slurm/quality/libraries.slurm --array=3-<rows-1>%16
#SBATCH --job-name=quality-library
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$PY" -m paper.analyses.quality.run_one "$SLURM_ARRAY_TASK_ID"
