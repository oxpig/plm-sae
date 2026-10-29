#!/bin/bash
#SBATCH --job-name=plm-sae-encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --array=0-17
# Latents of the nine hyperparameter-matched SAEs for the training and validation splits (scripts/extraction/encode.sh;
# task = 2 x SAE + split). The p-IgGen light-chain-type encodes (Supplementary Section A.7) are tasks 0-5 with VARIANT=vhvl.
#   examples/slurm/submit.sh examples/slurm/extraction/encode.slurm
#   examples/slurm/submit.sh examples/slurm/extraction/encode.slurm --array=0-5 --time=03:00:00 --export=ALL,VARIANT=vhvl
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
NAMES=(piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8 iglm_topk_r8 iglm_batchtopk_r8 iglm_ordered_r8
       progen2oas_topk_r8 progen2oas_batchtopk_r8 progen2oas_ordered_r8)
SPLITS=(train val)
T=$SLURM_ARRAY_TASK_ID
bash "$PLM_SAE_REPO/scripts/extraction/encode.sh" "${NAMES[$((T / 2))]}" "${SPLITS[$((T % 2))]}" ${VARIANT:+"$VARIANT"}
