#!/bin/bash
# IMGT localisation (paper/scripts/identification/run_imgt.sh). CPU only, 40 processes.
#   examples/slurm/submit.sh paper/slurm/identification/imgt.slurm
#SBATCH --job-name=identification-imgt
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/identification/run_imgt.sh"
