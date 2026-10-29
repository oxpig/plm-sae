#!/bin/bash
# Developability gate of the nine matched SAEs, one array task per SAE (CPU; after flashtap_val.slurm).
#   examples/slurm/submit.sh paper/slurm/properties/dev_gate.slurm
#SBATCH --job-name=properties-devgate
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-8
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
SAES=(piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8 iglm_topk_r8 iglm_batchtopk_r8 iglm_ordered_r8
      progen2oas_topk_r8 progen2oas_batchtopk_r8 progen2oas_ordered_r8)
bash "$PLM_SAE_REPO/paper/scripts/properties/developability_gate.sh" "${SAES[${SLURM_ARRAY_TASK_ID:-0}]}"
