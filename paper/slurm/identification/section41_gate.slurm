#!/bin/bash
# Feature identification gate of the expansion-32 TopK SAE of Section 4.1 (paper/scripts/identification/section41_gate.sh),
# one array task per layer (3: Table 1; 0-3: Supplementary Table 12). Layers 0-2 need up to 340 GB.
#   examples/slurm/submit.sh paper/slurm/identification/section41_gate.slurm --array=3
#SBATCH --job-name=identification-s41-gate
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=3
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
bash "$PLM_SAE_REPO/paper/scripts/identification/section41_gate.sh" "${SLURM_ARRAY_TASK_ID:-3}"
