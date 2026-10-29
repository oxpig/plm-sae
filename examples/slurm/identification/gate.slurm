#!/bin/bash
# Feature identification gates of the hyperparameter-matched SAEs (scripts/identification/run_gate.sh), one array
# task per gate: tasks 0-8 J-gene gates of the nine SAEs, 9-11 light-chain-type gates of the p-IgGen SAEs.
# Submit from the repository root:
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD" examples/slurm/identification/gate.slurm
#SBATCH --job-name=identification-gate
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
NAMES=(piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8 iglm_topk_r8 iglm_batchtopk_r8 iglm_ordered_r8
       progen2oas_topk_r8 progen2oas_batchtopk_r8 progen2oas_ordered_r8 piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8)
T=${SLURM_ARRAY_TASK_ID:-0}
LABEL=j; [ "$T" -ge 9 ] && LABEL=lct
bash "$PLM_SAE_REPO/scripts/identification/run_gate.sh" "${NAMES[$T]}" "$LABEL"
