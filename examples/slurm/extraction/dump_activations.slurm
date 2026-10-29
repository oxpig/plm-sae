#!/bin/bash
#SBATCH --job-name=plm-sae-dump
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
# Base-model activation cache for SAE training (scripts/extraction/dump_activations.sh), one model per job:
#   examples/slurm/submit.sh examples/slurm/extraction/dump_activations.slurm --export=ALL,MODEL=piggen
#   (MODEL = piggen | IgLM | progen2_oas)
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
bash "$PLM_SAE_REPO/scripts/extraction/dump_activations.sh" "${MODEL:?set MODEL}"
