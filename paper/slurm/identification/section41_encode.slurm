#!/bin/bash
# Pooled latents of the expansion-32 TopK SAE of Section 4.1 (paper/scripts/identification/section41_encode.sh), one
# array task per shard: 0-7 validation, 8-31 training. LAYER (default 3) selects the SAE.
#   examples/slurm/submit.sh paper/slurm/identification/section41_encode.slurm
#   examples/slurm/submit.sh paper/slurm/identification/section41_encode.slurm --export=ALL,LAYER=0
#SBATCH --job-name=identification-s41-encode
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/identification/section41_encode.sh" "$SLURM_ARRAY_TASK_ID" "${LAYER:-3}"
