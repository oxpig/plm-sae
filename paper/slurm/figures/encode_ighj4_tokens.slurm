#!/bin/bash
#SBATCH --job-name=plm-sae-encode-ighj4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
# Per-token latents of the IgLM (task 0) and ProGen2-OAS (task 1) TopK SAEs for the IGHJ4 validation sequences.
# Submit from a shell that has sourced scripts/env.sh:  sbatch --export=ALL paper/slurm/figures/encode_ighj4_tokens.slurm
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
NAMES=(iglm_topk_r8 progen2oas_topk_r8)
bash "$PLM_SAE_REPO/paper/scripts/figures/encode_ighj4_tokens.sh" "${NAMES[$SLURM_ARRAY_TASK_ID]}"
